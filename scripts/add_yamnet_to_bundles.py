"""Add YAMNet per-window features to existing analysis bundles.

For each cached ``data/output/<test>_analysis_bundle.json`` we
  1. Load the bundle and resolve its source video,
  2. Extract a fresh 16 kHz mono PCM via the same ffmpeg recipe ``audio/analyze.py`` uses,
  3. Run YAMNet on the waveform, aggregate to the bundle's existing
     ``AudioWindow`` cadence,
  4. Write the four ``yamnet_*_score`` fields into each window's extras,
  5. Persist the bundle in place.

This avoids re-running the slow visual + audio + Whisper passes; only
the audio extraction (a few seconds) and YAMNet inference (~tens of
seconds for a 24-min video on CPU) need to run.

Usage
-----
    PYTHONPATH=. python scripts/add_yamnet_to_bundles.py
    PYTHONPATH=. python scripts/add_yamnet_to_bundles.py --tests test_005 test_010
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audio.analyze import _ffmpeg_extract_wav, _read_wav_mono  # noqa: E402
from audio.yamnet_features import (  # noqa: E402
    ALL_YAMNET_KEYS,
    compute_yamnet_per_window,
    default_model_path,
)
from schemas.modality import AnalysisBundle  # noqa: E402

DEFAULT_TESTS = ["test_001", "test_002", "test_003", "test_004", "test_005", "test_010"]


def _resolve_video(bundle: AnalysisBundle, fallback_dir: Path) -> Path:
    """Pick the video file matching the bundle's ``video_path``, falling back
    to ``<fallback_dir>/<stem>.mp4`` when the recorded path is no longer
    valid (common after moving the workspace)."""
    raw = bundle.video_path or ""
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate
    if candidate.name:
        guess = fallback_dir / candidate.name
        if guess.is_file():
            return guess
    raise FileNotFoundError(f"Could not resolve video for bundle (video_path={raw!r})")


def _add_yamnet_to_bundle(bundle_path: Path, videos_dir: Path, model_path: Path) -> dict:
    bundle = AnalysisBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    if not bundle.audio_windows:
        return {"path": str(bundle_path), "n_windows": 0, "skipped": "no audio windows"}

    video_path = _resolve_video(bundle, videos_dir)

    t_start = time.time()
    with tempfile.TemporaryDirectory(prefix="yamnet_extract_") as tmpdir:
        wav_path = Path(tmpdir) / f"{video_path.stem}.wav"
        _ffmpeg_extract_wav(video_path, wav_path, sample_rate=16_000)
        samples, sr = _read_wav_mono(wav_path)
    t_extract = time.time() - t_start

    bounds: list[tuple[float, float]] = [(float(w.t0), float(w.t1)) for w in bundle.audio_windows]

    t_start = time.time()
    yamnet_scores = compute_yamnet_per_window(
        samples,
        audio_window_bounds=bounds,
        model_path=model_path,
    )
    t_inference = time.time() - t_start

    for idx, window in enumerate(bundle.audio_windows):
        extra = window.model_extra
        if extra is None:
            extra = {}
            window.__pydantic_extra__ = extra  # type: ignore[attr-defined]
        for key in ALL_YAMNET_KEYS:
            extra[key] = float(yamnet_scores[key][idx])

    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    return {
        "path": str(bundle_path),
        "n_windows": len(bundle.audio_windows),
        "samples": int(samples.size),
        "sample_rate": int(sr),
        "extract_sec": round(t_extract, 2),
        "inference_sec": round(t_inference, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="add_yamnet_to_bundles", description=__doc__)
    parser.add_argument(
        "--tests",
        nargs="+",
        default=DEFAULT_TESTS,
        help="List of test stems (default: all 6 cached bundles).",
    )
    parser.add_argument(
        "--bundles-dir",
        type=Path,
        default=ROOT / "data" / "output",
        help="Directory holding <test>_analysis_bundle.json files.",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=ROOT / "videos_with_ad",
        help="Directory holding the source .mp4 files.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model_path(),
        help="Path to the YAMNet ONNX model.",
    )
    args = parser.parse_args(argv)

    if not args.model.is_file():
        print(f"Error: YAMNet model not found at {args.model}", file=sys.stderr)
        return 2

    summary: list[dict] = []
    for test in args.tests:
        bundle_path = args.bundles_dir / f"{test}_analysis_bundle.json"
        if not bundle_path.is_file():
            print(f"[skip] {test}: bundle missing at {bundle_path}", file=sys.stderr)
            continue
        print(f"[{test}] running YAMNet on {bundle_path.name} …", flush=True)
        try:
            result = _add_yamnet_to_bundle(bundle_path, args.videos_dir, args.model)
            summary.append({"test": test, **result})
            print(
                f"[{test}] wrote yamnet_* fields to {result['n_windows']} windows "
                f"(extract {result.get('extract_sec', 0)}s, "
                f"inference {result.get('inference_sec', 0)}s)",
                flush=True,
            )
        except Exception as exc:
            print(f"[{test}] ERROR: {exc}", file=sys.stderr, flush=True)
            summary.append({"test": test, "error": str(exc)})

    print()
    print("=" * 60)
    print(" YAMNet integration summary ")
    print("=" * 60)
    for row in summary:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
