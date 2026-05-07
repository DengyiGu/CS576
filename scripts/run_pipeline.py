"""End-to-end pipeline runner: visual -> audio (+ speech) -> fusion -> segments.

Usage
-----
Run from the repo root with PYTHONPATH set so the local packages are importable.

Exact command (run from repository root):

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
    --skip-analysis        Reuse the existing <stem>_analysis_bundle.json in
                                                 data/output/ and only run fusion.
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

def _run_ocr(video: Path, bundle_path: Path) -> None:
    from ocr.analyze import build_ocr_spans
    from schemas.modality import AnalysisBundle

    bundle = AnalysisBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))

    print("     Running OCR (this can be slow)...")
    ocr_spans = build_ocr_spans(
        video,
        sample_every_sec=3.0,
        span_sec=4.0,
        min_confidence=0.35,
        allow_download=False,  # set to False after first run to avoid re-downloading models
        # gpu=True,          # uncomment if you have GPU + CUDA
    )

    # Merge OCR spans with existing speech spans
    if not hasattr(bundle, 'speech_spans') or bundle.speech_spans is None:
        bundle.speech_spans = []
    
    bundle.speech_spans.extend(ocr_spans)
    print(f"     OCR: added {len(ocr_spans)} text spans")

    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    print(f"     Updated bundle with OCR -> {bundle_path}")


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


def _bundle_and_segments_paths(video: Path, out_dir: Path) -> tuple[Path, Path]:
    bundle_path = out_dir / f"{video.stem}_analysis_bundle.json"
    segments_path = out_dir / f"{video.stem}_segments.json"
    return bundle_path, segments_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/run_pipeline.py",
        description="Run visual + audio + speech + fusion end-to-end on a video.",
    )
    parser.add_argument("video", type=Path, nargs="?", help="Input video file (.mp4 / .mov / ...).")
    parser.add_argument("--input-dir", type=Path, help="Process all videos in the specified directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/output"))
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument(
        "--force-analysis",
        action="store_true",
        help=(
            "Force re-running visual/audio analysis even if an existing "
            "analysis bundle is present in the output directory."
        ),
    )
    parser.add_argument("--skip-speech", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--model", default="small")
    parser.add_argument("--language", default="en")
    parser.add_argument("--vad", action="store_true")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--audio-window-sec", type=float, default=1.0)
    parser.add_argument("--min-segment-sec", type=float, default=20.0)
    parser.add_argument("--ocr", action="store_true", help="Run OCR on frames for brand/logo detection")
    args = parser.parse_args(argv)

    # Determine which videos to process
    if args.input_dir:
        videos_dir = args.input_dir.expanduser().resolve()
        if not videos_dir.is_dir():
            print(f"Error: videos directory not found: {videos_dir}", file=sys.stderr)
            return 2
        # Find all video files (common video formats)
        video_files = sorted(set(
            v for ext in ["*.mp4", "*.mov", "*.avi", "*.mkv"]
            for v in videos_dir.glob(ext)
            if v.is_file()
        ))
        if not video_files:
            print(f"Error: no video files found in {videos_dir}", file=sys.stderr)
            return 2
        videos_to_process = video_files
    else:
        if args.video is None:
            parser.print_help()
            print(f"\nError: either provide a video file or use --input-dir", file=sys.stderr)
            return 2
        video = args.video.expanduser().resolve(strict=False)
        if not video.is_file():
            print(f"Error: video not found: {video}", file=sys.stderr)
            return 2
        videos_to_process = [video]

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process each video
    return_code = 0
    for video_idx, video in enumerate(videos_to_process, 1):
        print(f"\n{'='*60}")
        print(f"Processing video {video_idx}/{len(videos_to_process)}: {video.name}")
        print(f"{'='*60}")

        bundle_path, segments_path = _bundle_and_segments_paths(video, out_dir)

        print(f"Pipeline: {video.name}")
        print(f"  Bundle  -> {bundle_path}")
        print(f"  Segments-> {segments_path}")

        has_existing_bundle = bundle_path.is_file()

        if args.skip_analysis:
            if not has_existing_bundle:
                print(f"Error: analysis bundle not found: {bundle_path}", file=sys.stderr)
                return_code = 2
                continue
            reuse_bundle = True
        elif has_existing_bundle and not args.force_analysis:
            reuse_bundle = True
        else:
            reuse_bundle = False

        total_steps = 1 if reuse_bundle else 1 + (0 if args.skip_audio else 1) + (1 if args.ocr else 0) + 1
        step = 0

        if not reuse_bundle:
            step += 1
            t0 = time.monotonic()
            _print_step(step, total_steps, "Visual analysis")
            _run_visual(video, bundle_path, args.sample_fps, args.window_sec)
            print(f"     Took {time.monotonic() - t0:.1f}s")

            if args.ocr:
                step += 1
                t0 = time.monotonic()
                _print_step(step, total_steps, "OCR analysis")
                _run_ocr(video, bundle_path)
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
        else:
            print("  Reusing existing analysis bundle")

        step += 1
        t0 = time.monotonic()
        _print_step(step, total_steps, "Fusion")
        _run_fusion(bundle_path, segments_path, args.min_segment_sec)
        print(f"     Took {time.monotonic() - t0:.1f}s")

        print()
        print(f"Done with {video.name}")

    print(f"\n{'='*60}")
    print(f"All videos processed.")
    print(f"{'='*60}")
    print(f"To launch the player with:")
    print(f"  PYTHONPATH=. python -m player.player")
    print(f"  (open any video file in the file dialog; segments load from data/output/)")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
