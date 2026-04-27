from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from schemas.modality import SpeechSpan


WHISPER_MODEL_REPO_ID = "Systran/faster-whisper-small"
DEFAULT_WHISPER_MODEL_DIR = Path(__file__).resolve().parent / "models" / "faster-whisper-small"


def is_whisper_model_available(model_dir: Path = DEFAULT_WHISPER_MODEL_DIR) -> bool:
    return (model_dir / "model.bin").exists() and (model_dir / "config.json").exists()


def download_whisper_model(model_dir: Path = DEFAULT_WHISPER_MODEL_DIR) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub first: python -m pip install huggingface_hub") from exc

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=WHISPER_MODEL_REPO_ID,
        local_dir=str(model_dir),
        max_workers=1,
    )
    return model_dir


def ensure_whisper_model(model_dir: Path = DEFAULT_WHISPER_MODEL_DIR, *, allow_download: bool = False) -> Path:
    if is_whisper_model_available(model_dir):
        return model_dir
    if allow_download:
        return download_whisper_model(model_dir)
    raise RuntimeError(
        f"Local faster-whisper model not found in {model_dir}. "
        "Run: python Final_project/segment_text_analyzer.py --download-model"
    )


def build_speech_spans(
    video_path: Path,
    *,
    model_dir: Path = DEFAULT_WHISPER_MODEL_DIR,
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
) -> list[SpeechSpan]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install faster-whisper first: python -m pip install faster-whisper") from exc

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    local_model_dir = ensure_whisper_model(model_dir)
    model = WhisperModel(str(local_model_dir), device=device, compute_type=compute_type)
    whisper_segments, _ = model.transcribe(str(video_path), language=language)

    speech_spans: list[SpeechSpan] = []
    for segment in whisper_segments:
        text = segment.text.strip()
        if not text:
            continue
        speech_spans.append(
            SpeechSpan(
                t0=float(segment.start),
                t1=float(segment.end),
                text=text,
            )
        )
    return speech_spans


def speech_spans_to_payload(speech_spans: list[SpeechSpan]) -> list[dict[str, Any]]:
    return [span.model_dump() for span in speech_spans]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build timestamped SpeechSpan records with a local faster-whisper model.")
    parser.add_argument("video", nargs="?", help="Local video path to transcribe")
    parser.add_argument("--download-model", action="store_true", help="Download the local faster-whisper model")
    parser.add_argument(
        "--model-dir",
        "--whisper-model-dir",
        dest="whisper_model_dir",
        default=str(DEFAULT_WHISPER_MODEL_DIR),
        help="Local faster-whisper model directory",
    )
    parser.add_argument("--device", default="cpu", help="Whisper inference device, usually cpu or cuda")
    parser.add_argument("--compute-type", default="int8", help="Whisper compute type, usually int8 for CPU")
    parser.add_argument("--language", default=None, help="Optional language code, such as en or zh")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.download_model:
        whisper_dir = download_whisper_model(Path(args.whisper_model_dir))
        print(f"Whisper model downloaded to: {whisper_dir}")
        return 0

    if not args.video:
        parser.error("video is required unless --download-model is used")

    speech_spans = build_speech_spans(
        Path(args.video),
        model_dir=Path(args.whisper_model_dir),
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )
    print(json.dumps(speech_spans_to_payload(speech_spans), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
