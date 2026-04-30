"""CLI for the audio modality.

Examples
--------
Stand-alone audio track JSON::

    python -m audio_analyze \\
        --video data/input/test_001.mp4 \\
        --out data/output/test_001_audio_track.json

Re-run the audio analysis on a pre-extracted WAV (no ffmpeg)::

    python -m audio_analyze \\
        --audio-in data/intermediate/test_001.wav \\
        --out data/output/test_001_audio_track.json

Merge audio windows into an existing AnalysisBundle (visual already inside)::

    python -m audio_analyze \\
        --video data/input/test_001.mp4 \\
        --bundle-in data/output/test_001_analysis_bundle.json \\
        --bundle-out data/output/test_001_analysis_bundle.json

End-to-end (audio + speech) using ``video_info`` descriptors::

    python -m audio_analyze \\
        --video-info video_info/test_001.json \\
        --videos-root data/input \\
        --bundle-in data/output/test_001_analysis_bundle.json \\
        --bundle-out data/output/test_001_analysis_bundle.json \\
        --with-speech --model small --language en

Faster speech pass with VAD on (skips silent regions)::

    python -m audio_analyze \\
        --video data/input/test_001.mp4 \\
        --bundle-in data/output/test_001_analysis_bundle.json \\
        --bundle-out data/output/test_001_analysis_bundle.json \\
        --with-speech --model small --vad

Speech recognition uses ``Automatic_speech_recognition/segment_text_analyzer.py``;
download the selected model first with::

    python Automatic_speech_recognition/segment_text_analyzer.py \\
        --download-model --model small
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audio.analyze import analyze_audio, audio_windows_to_payload, write_audio_track_json
from schemas.modality import AnalysisBundle, AudioWindow, SpeechSpan


def _resolve_input_video(args: argparse.Namespace) -> Path | None:
    """Return the resolved video path, or None when running from --audio-in."""
    if args.audio_in is not None:
        return None

    if args.video is not None:
        path = Path(args.video).expanduser().resolve(strict=False)
        if not path.is_file():
            print(f"Error: video file not found: {path}", file=sys.stderr)
            raise SystemExit(2)
        return path

    from visual.video_info_dataset import load_doc_and_resolve_video

    info_path = Path(args.video_info).expanduser().resolve(strict=False)
    if not info_path.is_file():
        print(f"Error: --video-info file not found: {info_path}", file=sys.stderr)
        raise SystemExit(2)
    videos_root = args.videos_root.expanduser().resolve()
    _doc, video, searched = load_doc_and_resolve_video(info_path, videos_root)
    if video is None or not video.is_file():
        print(
            f"Error: stitched video not found for {info_path.name}. Searched:",
            file=sys.stderr,
        )
        for entry in searched:
            print(f"  {entry}", file=sys.stderr)
        raise SystemExit(2)
    return video


def _resolve_audio_in(args: argparse.Namespace) -> Path | None:
    if args.audio_in is None:
        return None
    path = Path(args.audio_in).expanduser().resolve(strict=False)
    if not path.is_file():
        print(f"Error: --audio-in file not found: {path}", file=sys.stderr)
        raise SystemExit(2)
    return path


def _maybe_load_bundle(bundle_in: Path | None) -> AnalysisBundle | None:
    if bundle_in is None:
        return None
    path = bundle_in.expanduser().resolve(strict=False)
    if not path.is_file():
        print(f"Error: --bundle-in not found: {path}", file=sys.stderr)
        raise SystemExit(2)
    return AnalysisBundle.model_validate_json(path.read_text(encoding="utf-8"))


def _run_speech_to_text(
    video: Path,
    *,
    model_name: str,
    model_dir: Path | None,
    compute_type: str,
    language: str | None,
    vad: bool,
) -> list[SpeechSpan]:
    try:
        from Automatic_speech_recognition.segment_text_analyzer import build_speech_spans
    except ImportError as exc:
        raise RuntimeError(
            "Speech extraction requires Automatic_speech_recognition/. Ensure the package "
            "is on PYTHONPATH and faster-whisper is installed (see "
            "Automatic_speech_recognition/SETUP.md)."
        ) from exc

    return build_speech_spans(
        video,
        model_name=model_name,
        model_dir=model_dir,
        compute_type=compute_type,
        language=language,
        vad=vad,
    )


def _summarize(windows: list[AudioWindow]) -> str:
    if not windows:
        return "no audio windows"
    counts: dict[str, int] = {}
    for w in windows:
        label = str(w.model_extra.get("audio_label", "unknown")) if w.model_extra else "unknown"
        counts[label] = counts.get(label, 0) + 1
    parts = [f"{label}={counts[label]}" for label in sorted(counts)]
    return f"{len(windows)} windows ({', '.join(parts)})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m audio_analyze",
        description="Extract per-window audio features and (optionally) speech for the fusion pipeline.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, default=None, help="Input video file path.")
    src.add_argument(
        "--video-info",
        type=Path,
        default=None,
        help="Path to video_info/*.json (uses --videos-root to resolve the .mp4).",
    )
    src.add_argument(
        "--audio-in",
        type=Path,
        default=None,
        help=(
            "Pre-extracted mono PCM WAV (skip ffmpeg). Cannot be combined with "
            "--with-speech, which still requires the source video."
        ),
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=Path("data/input"),
        help="Directory containing the stitched .mp4 when using --video-info.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for stand-alone audio_track.json.",
    )
    parser.add_argument(
        "--bundle-in",
        type=Path,
        default=None,
        help="Existing AnalysisBundle to extend in place (visual + audio + speech).",
    )
    parser.add_argument(
        "--bundle-out",
        type=Path,
        default=None,
        help="Write the merged AnalysisBundle here (defaults to overwriting --bundle-in).",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.0,
        help="Audio analysis window length in seconds (must match visual window for fusion).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Sample rate for ffmpeg extraction (default 16 kHz, matches Whisper).",
    )
    parser.add_argument(
        "--keep-wav",
        type=Path,
        default=None,
        help="Optional path to retain the extracted mono WAV for inspection or reuse.",
    )
    parser.add_argument(
        "--with-speech",
        action="store_true",
        help="Also transcribe with faster-whisper and fill speech_spans on the bundle.",
    )
    parser.add_argument(
        "--model",
        choices=("tiny", "base", "small", "medium", "large-v3"),
        default="small",
        help="faster-whisper model size used when --with-speech is set (default: small).",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="Enable faster-whisper voice activity detection to skip non-speech audio.",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper CPU compute type (default: int8). Common: int8, int16, float32.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code passed to faster-whisper (default: en).",
    )
    parser.add_argument(
        "--whisper-model-dir",
        "--model-dir",
        dest="whisper_model_dir",
        type=Path,
        default=None,
        help="Override the local faster-whisper model directory (defaults to "
             "Automatic_speech_recognition/models/faster-whisper-<model>).",
    )
    args = parser.parse_args(argv)

    video = _resolve_input_video(args)
    audio_in = _resolve_audio_in(args)

    if args.with_speech and video is None:
        print(
            "Error: --with-speech requires a source video (use --video or --video-info, "
            "not --audio-in alone).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if audio_in is not None and args.keep_wav is not None:
        print(
            "Error: --keep-wav has no effect with --audio-in (no extraction is performed).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    source_label = video if video is not None else audio_in
    source_stem = (video or audio_in).stem  # type: ignore[union-attr]

    if args.out is None and args.bundle_out is None and args.bundle_in is None:
        default_dir = Path("data/output")
        default_dir.mkdir(parents=True, exist_ok=True)
        args.out = default_dir / f"{source_stem}_audio_track.json"

    if audio_in is not None:
        print(f"[audio] Reading pre-extracted WAV {audio_in} …", file=sys.stderr)
    else:
        print(f"[audio] Extracting audio features from {video} …", file=sys.stderr)
    windows, duration_sec = analyze_audio(
        video_path=video,
        audio_in=audio_in,
        window_sec=args.window_sec,
        sample_rate=args.sample_rate,
        keep_wav=args.keep_wav,
    )
    print(f"[audio] {_summarize(windows)}; duration={duration_sec:.2f}s", file=sys.stderr)

    speech_spans: list[SpeechSpan] = []
    if args.with_speech:
        assert video is not None  # guarded above
        print(
            f"[audio] Transcribing with faster-whisper (model={args.model}, "
            f"compute={args.compute_type}, vad={args.vad}, language={args.language})…",
            file=sys.stderr,
        )
        speech_spans = _run_speech_to_text(
            video,
            model_name=args.model,
            model_dir=args.whisper_model_dir,
            compute_type=args.compute_type,
            language=args.language,
            vad=args.vad,
        )
        print(f"[audio] Got {len(speech_spans)} speech spans", file=sys.stderr)

    if args.out is not None:
        write_audio_track_json(
            windows,
            args.out.expanduser().resolve(),
            video_path=source_label or "",
            duration_sec=duration_sec,
            window_sec=args.window_sec,
        )
        print(f"[audio] Wrote audio track → {args.out.resolve()}")

    if args.bundle_in is not None or args.bundle_out is not None:
        bundle = _maybe_load_bundle(args.bundle_in)
        if bundle is None:
            bundle = AnalysisBundle(
                video_path=str(source_label or ""),
                duration_sec=duration_sec,
                visual=None,
                audio_windows=[],
                speech_spans=[],
            )
        else:
            if not bundle.video_path and source_label is not None:
                bundle.video_path = str(source_label)
            if not bundle.duration_sec:
                bundle.duration_sec = duration_sec

        bundle.audio_windows = list(windows)
        if args.with_speech:
            bundle.speech_spans = list(speech_spans)

        target = (args.bundle_out or args.bundle_in)
        target = target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        print(f"[audio] Updated bundle → {target}")

    if args.with_speech and args.bundle_out is None and args.bundle_in is None:
        speech_dump = [span.model_dump() for span in speech_spans]
        print(json.dumps(speech_dump, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
