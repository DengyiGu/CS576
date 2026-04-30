from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from schemas.modality import SpeechSpan


MODELS_ROOT = Path(__file__).resolve().parent / "models"
DEFAULT_WHISPER_MODEL = "small"
WHISPER_MODELS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}
CPU_DEVICE = "cpu"
CPU_COMPUTE_TYPE = "int8"


def get_whisper_model_dir(model_name: str = DEFAULT_WHISPER_MODEL) -> Path:
    validate_whisper_model(model_name)
    return MODELS_ROOT / f"faster-whisper-{model_name}"


def validate_whisper_model(model_name: str) -> None:
    if model_name not in WHISPER_MODELS:
        valid = ", ".join(WHISPER_MODELS)
        raise ValueError(f"Unknown Whisper model '{model_name}'. Choose one of: {valid}")


def is_whisper_model_available(model_dir: Path) -> bool:
    return (model_dir / "model.bin").exists() and (model_dir / "config.json").exists()


def download_whisper_model(
    model_name: str = DEFAULT_WHISPER_MODEL,
    model_dir: Path | None = None,
) -> Path:
    validate_whisper_model(model_name)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub first: python -m pip install huggingface_hub") from exc

    model_dir = model_dir if model_dir is not None else get_whisper_model_dir(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=WHISPER_MODELS[model_name],
        local_dir=str(model_dir),
        max_workers=1,
    )
    return model_dir


def ensure_whisper_model(
    model_name: str = DEFAULT_WHISPER_MODEL,
    model_dir: Path | None = None,
    *,
    allow_download: bool = False,
) -> Path:
    validate_whisper_model(model_name)
    model_dir = model_dir if model_dir is not None else get_whisper_model_dir(model_name)
    if is_whisper_model_available(model_dir):
        return model_dir
    if allow_download:
        return download_whisper_model(model_name, model_dir)
    raise RuntimeError(
        f"Local faster-whisper model not found in {model_dir}. "
        f"Run: python Automatic_speech_recognition/segment_text_analyzer.py --download-model --model {model_name}"
    )


def build_speech_spans(
    video_path: Path,
    *,
    model_name: str = DEFAULT_WHISPER_MODEL,
    model_dir: Path | None = None,
    compute_type: str = CPU_COMPUTE_TYPE,
    language: str | None = "en",
    vad: bool = False,
) -> list[SpeechSpan]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Install faster-whisper first: python -m pip install faster-whisper") from exc

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    local_model_dir = ensure_whisper_model(model_name, model_dir)
    model = WhisperModel(str(local_model_dir), device=CPU_DEVICE, compute_type=compute_type)
    whisper_segments, _ = model.transcribe(
        str(video_path),
        language=language,
        vad_filter=vad,
        vad_parameters=dict(min_silence_duration_ms=500) if vad else None,
    )

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
    parser.add_argument("--download-model", action="store_true", help="Download the selected local faster-whisper model")
    parser.add_argument(
        "--model",
        choices=tuple(WHISPER_MODELS),
        default=DEFAULT_WHISPER_MODEL,
        help="Whisper model size to use or download",
    )
    parser.add_argument(
        "--model-dir",
        "--whisper-model-dir",
        dest="whisper_model_dir",
        default=None,
        help="Optional custom local faster-whisper model directory",
    )
    parser.add_argument("--compute-type", default=CPU_COMPUTE_TYPE, help="Whisper CPU compute type")
    parser.add_argument("--language", default="en", help="Language code, such as en or zh")
    parser.add_argument("--vad", action="store_true", help="Enable voice activity detection to skip non-speech sections")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.download_model:
        model_dir = Path(args.whisper_model_dir) if args.whisper_model_dir else None
        whisper_dir = download_whisper_model(args.model, model_dir)
        print(f"Whisper model downloaded to: {whisper_dir}")
        return 0

    if not args.video:
        parser.error("video is required unless --download-model is used")

    speech_spans = build_speech_spans(
        Path(args.video),
        model_name=args.model,
        model_dir=Path(args.whisper_model_dir) if args.whisper_model_dir else None,
        compute_type=args.compute_type,
        language=args.language,
        vad=args.vad,
    )
    print(json.dumps(speech_spans_to_payload(speech_spans), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())