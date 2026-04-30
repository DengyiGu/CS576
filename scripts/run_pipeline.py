"""End-to-end pipeline runner: visual -> audio (+ speech) -> fusion -> segments.

Usage
-----
Run from the repo root with PYTHONPATH set so the local packages are importable::

    PYTHONPATH=. python scripts/run_pipeline.py videos_with_ad/test_001.mp4

That single command:
  1. Runs the visual analyzer and writes ``<stem>_analysis_bundle.json``.
  2. Runs the audio analyzer (and Whisper, unless ``--skip-speech``) and merges
     audio_windows + speech_spans into the same bundle.
  3. Runs fusion and writes ``<stem>_segments.json``.

After it finishes, opening the video in the player will load those segments
automatically (player_fusion.py looks them up in ``data/output/``).

Common options
--------------
  --skip-speech          Skip Whisper transcription (faster; reduces accuracy).
  --skip-audio           Skip the audio modality entirely.
  --model MODEL          Whisper model size (default: small).
  --vad                  Enable Whisper VAD (skips silent regions).
  --sample-fps F         Visual frame sample rate (default: 1.0).
  --window-sec W         Visual window length in seconds (default: 2.0).
  --audio-window-sec W   Audio window length in seconds (default: 1.0).
  --min-segment-sec S    Minimum segment length used by fusion smoothing
                         (default: 20.0).
  --out-dir DIR          Output directory (default: data/output).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _print_step(idx: int, total: int, msg: str) -> None:
    print(f"\n[{idx}/{total}] {msg}", flush=True)


def _run_visual(video: Path, bundle_out: Path, sample_fps: float, window_sec: float) -> None:
    from visual.analyze import (
        analyze_visual,
        build_analysis_bundle,
        write_analysis_bundle_json,
    )

    track = analyze_visual(video, sample_fps=sample_fps, window_sec=window_sec)
    bundle = build_analysis_bundle(video, track=track)
    write_analysis_bundle_json(bundle, bundle_out)
    print(f"     Wrote bundle -> {bundle_out}")


def _run_audio(
    video: Path,
    bundle_path: Path,
    *,
    window_sec: float,
    with_speech: bool,
    model_name: str,
    vad: bool,
    language: str,
) -> None:
    from audio.analyze import analyze_audio
    from schemas.modality import AnalysisBundle

    bundle = AnalysisBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    windows, _duration = analyze_audio(video_path=video, window_sec=window_sec)
    bundle.audio_windows = list(windows)
    print(f"     Audio: {len(windows)} windows")

    if with_speech:
        from Automatic_speech_recognition.segment_text_analyzer import build_speech_spans

        bundle.speech_spans = build_speech_spans(
            video,
            model_name=model_name,
            language=language,
            vad=vad,
        )
        print(f"     Speech: {len(bundle.speech_spans)} spans")
    else:
        print("     Speech: skipped")

    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    print(f"     Updated bundle -> {bundle_path}")


def _run_fusion(bundle_path: Path, segments_out: Path, min_segment_sec: float) -> None:
    from fusion.fuse import fuse_bundle_to_segments, load_bundle, write_segments_json

    bundle = load_bundle(bundle_path)
    segments = fuse_bundle_to_segments(bundle, min_segment_seconds=min_segment_sec)
    write_segments_json(segments, segments_out)
    print(f"     Wrote {len(segments)} segments -> {segments_out}")
    if segments:
        for seg in segments:
            dur = seg["end"] - seg["start"]
            print(
                f"       {seg['start']:>8.1f}s - {seg['end']:>8.1f}s "
                f"({dur:>6.1f}s)  {seg['label']}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/run_pipeline.py",
        description="Run visual + audio + speech + fusion end-to-end on a video.",
    )
    parser.add_argument("video", type=Path, help="Input video file (.mp4 / .mov / ...).")
    parser.add_argument("--out-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--skip-speech", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="en")
    parser.add_argument("--vad", action="store_true")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--audio-window-sec", type=float, default=1.0)
    parser.add_argument("--min-segment-sec", type=float, default=20.0)
    args = parser.parse_args(argv)

    video = args.video.expanduser().resolve(strict=False)
    if not video.is_file():
        print(f"Error: video not found: {video}", file=sys.stderr)
        return 2

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / f"{video.stem}_analysis_bundle.json"
    segments_path = out_dir / f"{video.stem}_segments.json"

    print(f"Pipeline: {video.name}")
    print(f"  Bundle  -> {bundle_path}")
    print(f"  Segments-> {segments_path}")

    total_steps = 1 + (0 if args.skip_audio else 1) + 1
    step = 0

    step += 1
    t0 = time.monotonic()
    _print_step(step, total_steps, "Visual analysis")
    _run_visual(video, bundle_path, args.sample_fps, args.window_sec)
    print(f"     Took {time.monotonic() - t0:.1f}s")

    if not args.skip_audio:
        step += 1
        t0 = time.monotonic()
        _print_step(
            step,
            total_steps,
            "Audio analysis" + ("" if args.skip_speech else " + speech"),
        )
        _run_audio(
            video,
            bundle_path,
            window_sec=args.audio_window_sec,
            with_speech=not args.skip_speech,
            model_name=args.model,
            vad=args.vad,
            language=args.language,
        )
        print(f"     Took {time.monotonic() - t0:.1f}s")

    step += 1
    t0 = time.monotonic()
    _print_step(step, total_steps, "Fusion")
    _run_fusion(bundle_path, segments_path, args.min_segment_sec)
    print(f"     Took {time.monotonic() - t0:.1f}s")

    print()
    print(f"Done. Launch the player with:")
    print(f"  PYTHONPATH=. python -m player.player")
    print(f"  (open '{video}' in the file dialog; segments load from {segments_path.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
