"""Semantic ad scoring for ASR/OCR text spans.

This module keeps Whisper VAD enabled for speed, then merges nearby short text
spans into longer context windows before scoring them with a local sentence
embedding model.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np

from schemas.modality import SpeechSpan


MODELS_ROOT = Path(__file__).resolve().parent / "models"
DEFAULT_SEMANTIC_MODEL = "mini"
SEMANTIC_MODELS = {
    "mini": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}

AD_PROMPTS = [
    "This text is from an advertisement or commercial break.",
    "This segment promotes a sponsor, product, service, discount, coupon, or sale.",
    "This is a paid sponsorship, brand promotion, product pitch, or call to buy.",
    "This text asks viewers to subscribe, sign up, download, shop, or use a promo code.",
]
CONTENT_PROMPTS = [
    "This text is normal main video content.",
    "This is a lecture, discussion, narration, interview, gameplay, or story.",
    "This text explains the main topic of the video and is not an advertisement.",
]
INTRO_PROMPTS = [
    "This text is an opening introduction or title sequence at the start of a video.",
    "This segment introduces the video, speaker, topic, episode, film, or program.",
    "This is opening credits, title card, or introductory narration.",
]
OUTRO_PROMPTS = [
    "This text is a closing segment, ending, sign-off, or conclusion at the end of a video.",
    "This segment thanks the audience, wraps up the video, or says goodbye.",
    "This is closing credits, end card, or final takeaway.",
]


def get_semantic_model_dir(model_name: str = DEFAULT_SEMANTIC_MODEL) -> Path:
    validate_semantic_model(model_name)
    return MODELS_ROOT / f"sentence-transformer-{model_name}"


def validate_semantic_model(model_name: str) -> None:
    if model_name not in SEMANTIC_MODELS:
        valid = ", ".join(SEMANTIC_MODELS)
        raise ValueError(f"Unknown semantic model '{model_name}'. Choose one of: {valid}")


def is_semantic_model_available(model_dir: Path) -> bool:
    return (model_dir / "modules.json").exists() and (model_dir / "config_sentence_transformers.json").exists()


def download_semantic_model(
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    model_dir: Path | None = None,
) -> Path:
    validate_semantic_model(model_name)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers first: python -m pip install -r requirements-semantic.txt") from exc

    model_dir = model_dir if model_dir is not None else get_semantic_model_dir(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(SEMANTIC_MODELS[model_name])
    model.save(str(model_dir))
    return model_dir


def ensure_semantic_model(
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    model_dir: Path | None = None,
    *,
    allow_download: bool = False,
) -> Path:
    validate_semantic_model(model_name)
    model_dir = model_dir if model_dir is not None else get_semantic_model_dir(model_name)
    if is_semantic_model_available(model_dir):
        return model_dir
    if allow_download:
        return download_semantic_model(model_name, model_dir)
    raise RuntimeError(
        f"Local semantic model not found in {model_dir}. "
        f"Run: python -c \"from semantic.analyze import download_semantic_model; "
        f"download_semantic_model('{model_name}')\""
    )


@lru_cache(maxsize=4)
def _load_sentence_model(model_dir: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Install sentence-transformers first: python -m pip install -r requirements-semantic.txt") from exc
    return SentenceTransformer(model_dir, device=device)


def merge_text_spans(
    speech_spans: Iterable[SpeechSpan],
    *,
    max_gap_sec: float = 4.0,
    max_window_sec: float = 60.0,
    min_chars: int = 25,
) -> list[SpeechSpan]:
    clean_spans = [
        span
        for span in speech_spans
        if span.text.strip()
        and (span.model_extra or {}).get("source") not in {"semantic", "semantic_structure"}
    ]
    clean_spans.sort(key=lambda span: (span.t0, span.t1))

    merged: list[SpeechSpan] = []
    start: float | None = None
    end: float | None = None
    chunks: list[str] = []

    def flush() -> None:
        nonlocal start, end, chunks
        if start is None or end is None:
            return
        text = " ".join(chunks).strip()
        if len(text) >= min_chars:
            merged.append(SpeechSpan(t0=start, t1=end, text=text, source="merged_text"))
        start = None
        end = None
        chunks = []

    for span in clean_spans:
        text = span.text.strip()
        if start is None or end is None:
            start = float(span.t0)
            end = float(span.t1)
            chunks = [text]
            continue

        gap = float(span.t0) - end
        merged_duration = float(span.t1) - start
        if gap > max_gap_sec or merged_duration > max_window_sec:
            flush()
            start = float(span.t0)
            end = float(span.t1)
            chunks = [text]
        else:
            end = max(end, float(span.t1))
            chunks.append(text)

    flush()
    return merged


def _score_against_prompts(
    model_dir: Path,
    texts: list[str],
    positive_prompts: list[str],
    negative_prompts: list[str],
    *,
    device: str = "cpu",
) -> list[tuple[float, float]]:
    if not texts:
        return []

    model = _load_sentence_model(str(model_dir), device)
    text_embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    positive_embeddings = model.encode(positive_prompts, normalize_embeddings=True, show_progress_bar=False)
    negative_embeddings = model.encode(negative_prompts, normalize_embeddings=True, show_progress_bar=False)

    text_matrix = np.asarray(text_embeddings, dtype=np.float32)
    positive_matrix = np.asarray(positive_embeddings, dtype=np.float32)
    negative_matrix = np.asarray(negative_embeddings, dtype=np.float32)

    positive_scores = text_matrix @ positive_matrix.T
    negative_scores = text_matrix @ negative_matrix.T

    results: list[tuple[float, float]] = []
    for i in range(len(texts)):
        positive_sim = float(positive_scores[i].max())
        negative_sim = float(negative_scores[i].max())
        margin = positive_sim - negative_sim
        score = max(0.0, min(1.0, 0.5 + 2.0 * margin))
        results.append((score, margin))
    return results


def _score_texts(model_dir: Path, texts: list[str], *, device: str = "cpu") -> list[tuple[float, float]]:
    return _score_against_prompts(
        model_dir,
        texts,
        AD_PROMPTS,
        CONTENT_PROMPTS,
        device=device,
    )


def build_semantic_ad_spans(
    speech_spans: Iterable[SpeechSpan],
    *,
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    model_dir: Path | None = None,
    allow_download: bool = False,
    threshold: float = 0.60,
    max_gap_sec: float = 4.0,
    max_window_sec: float = 60.0,
    min_chars: int = 25,
    device: str = "cpu",
    emit_all_scores: bool = False,
) -> list[SpeechSpan]:
    local_model_dir = ensure_semantic_model(model_name, model_dir, allow_download=allow_download)
    merged_spans = merge_text_spans(
        speech_spans,
        max_gap_sec=max_gap_sec,
        max_window_sec=max_window_sec,
        min_chars=min_chars,
    )
    scores = _score_texts(local_model_dir, [span.text for span in merged_spans], device=device)

    semantic_spans: list[SpeechSpan] = []
    for span, (score, margin) in zip(merged_spans, scores):
        if score < threshold and not emit_all_scores:
            continue
        semantic_spans.append(
            SpeechSpan(
                t0=span.t0,
                t1=span.t1,
                text=span.text,
                source="semantic",
                semantic_ad_score=round(score, 4),
                semantic_margin=round(margin, 4),
                semantic_is_ad=score >= threshold,
            )
        )
    return semantic_spans


def build_semantic_structure_spans(
    speech_spans: Iterable[SpeechSpan],
    *,
    model_name: str = DEFAULT_SEMANTIC_MODEL,
    model_dir: Path | None = None,
    allow_download: bool = False,
    threshold: float = 0.58,
    max_gap_sec: float = 4.0,
    max_window_sec: float = 60.0,
    min_chars: int = 20,
    device: str = "cpu",
    emit_all_scores: bool = False,
) -> list[SpeechSpan]:
    local_model_dir = ensure_semantic_model(model_name, model_dir, allow_download=allow_download)
    merged_spans = merge_text_spans(
        speech_spans,
        max_gap_sec=max_gap_sec,
        max_window_sec=max_window_sec,
        min_chars=min_chars,
    )
    texts = [span.text for span in merged_spans]
    intro_scores = _score_against_prompts(
        local_model_dir,
        texts,
        INTRO_PROMPTS,
        CONTENT_PROMPTS + AD_PROMPTS + OUTRO_PROMPTS,
        device=device,
    )
    outro_scores = _score_against_prompts(
        local_model_dir,
        texts,
        OUTRO_PROMPTS,
        CONTENT_PROMPTS + AD_PROMPTS + INTRO_PROMPTS,
        device=device,
    )

    semantic_spans: list[SpeechSpan] = []
    for span, (intro_score, intro_margin), (outro_score, outro_margin) in zip(
        merged_spans,
        intro_scores,
        outro_scores,
    ):
        if max(intro_score, outro_score) < threshold and not emit_all_scores:
            continue
        semantic_spans.append(
            SpeechSpan(
                t0=span.t0,
                t1=span.t1,
                text=span.text,
                source="semantic_structure",
                semantic_intro_score=round(intro_score, 4),
                semantic_intro_margin=round(intro_margin, 4),
                semantic_outro_score=round(outro_score, 4),
                semantic_outro_margin=round(outro_margin, 4),
            )
        )
    return semantic_spans


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download or test the local semantic ad text model.")
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download the selected sentence-transformer model for offline semantic scoring.",
    )
    parser.add_argument(
        "--model",
        choices=tuple(SEMANTIC_MODELS),
        default=DEFAULT_SEMANTIC_MODEL,
        help="Semantic model to use or download.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device for semantic inference when called from code. Download does not require CUDA.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.download_model:
        model_dir = download_semantic_model(args.model)
        print(f"Semantic model downloaded to: {model_dir}")
        return 0

    parser.error("Use --download-model to download a local semantic model.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
