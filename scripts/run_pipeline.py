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


def _run_ocr(video: Path, bundle_path: Path, *, gpu: bool = False) -> None:
    """Optional: extract on-screen text from visually suspicious windows.

    OCR spans are appended to ``bundle.speech_spans`` with ``source="ocr"`` so
    the existing brand-name / phrase matching in fusion picks them up.
    """
    try:
        from ocr.analyze import build_ocr_spans
    except Exception as exc:
        print(f"     OCR: skipped ({exc}) — install with `pip install -r requirements-ocr.txt`")
        return

    from schemas.modality import AnalysisBundle

    bundle = AnalysisBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    candidate_times: list[float] = []
    if bundle.visual is not None:
        for window in bundle.visual.windows:
            if (
                window.high_text_density
                or window.visual_hypothesis in {"graphics_heavy", "static"}
                or window.palette_delta > 0.35
            ):
                candidate_times.append(0.5 * (window.t0 + window.t1))

    try:
        ocr_spans = build_ocr_spans(
            video,
            candidate_times=candidate_times if candidate_times else None,
            sample_every_sec=10.0,
            max_frames=260,
            gpu=gpu,
        )
    except Exception as exc:
        print(f"     OCR: failed during analysis ({exc})")
        return

    bundle.speech_spans = list(bundle.speech_spans) + list(ocr_spans)
    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    print(f"     OCR: {len(ocr_spans)} text spans (candidate windows={len(candidate_times)})")


def _run_semantic(bundle_path: Path, *, device: str = "cpu") -> None:
    """Optional: score ASR/OCR text spans for ad / intro / outro semantics.

    Adds ``source="semantic"`` and ``source="semantic_structure"`` SpeechSpans
    onto the bundle.  These extra fields are visible to fusion but only fully
    utilised once ``fusion/fuse.py`` is updated to consume them.
    """
    try:
        from semantic.analyze import build_semantic_ad_spans, build_semantic_structure_spans
    except Exception as exc:
        print(f"     Semantic: skipped ({exc}) — install with `pip install -r requirements-semantic.txt`")
        return

    from schemas.modality import AnalysisBundle

    bundle = AnalysisBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
    if not bundle.speech_spans:
        print("     Semantic: skipped (no ASR/OCR text spans to score)")
        return

    try:
        ad_spans = build_semantic_ad_spans(bundle.speech_spans, device=device)
        bundle.speech_spans = list(bundle.speech_spans) + list(ad_spans)
        struct_spans = build_semantic_structure_spans(bundle.speech_spans, device=device)
        bundle.speech_spans = list(bundle.speech_spans) + list(struct_spans)
    except Exception as exc:
        print(f"     Semantic: failed during analysis ({exc})")
        return

    bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    print(f"     Semantic: ad={len(ad_spans)} structure={len(struct_spans)} new spans")


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
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        help="Input video file (.mp4 / .mov / ...). Omit when using --input-dir.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Process all video files in a directory instead of a single file.",
    )
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
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip visual analysis and use existing analysis bundle in the output dir.",
    )
    parser.add_argument(
        "--with-ocr",
        action="store_true",
        help="Run on-frame OCR on visually suspicious windows (slow; needs `pip install -r requirements-ocr.txt`).",
    )
    parser.add_argument(
        "--with-semantic",
        action="store_true",
        help="Score ASR/OCR text with semantic ad/intro/outro embeddings (needs `pip install -r requirements-semantic.txt`).",
    )
    parser.add_argument(
        "--text-device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device for OCR + semantic models when those flags are enabled.",
    )
    args = parser.parse_args(argv)

    def _process_single(video_path: Path) -> int:
        video = video_path.expanduser().resolve(strict=False)
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

        total_steps = (
            (0 if args.skip_analysis else 1)
            + (0 if args.skip_audio else 1)
            + (1 if args.with_ocr else 0)
            + (1 if args.with_semantic else 0)
            + 1
        )
        step = 0

        if not args.skip_analysis:
            step += 1
            t0 = time.monotonic()
            _print_step(step, total_steps, "Visual analysis")
            _run_visual(video, bundle_path, args.sample_fps, args.window_sec)
            print(f"     Took {time.monotonic() - t0:.1f}s")
        else:
            step += 1
            _print_step(step, total_steps, "Skip visual analysis — using existing bundle")
            if not bundle_path.is_file():
                print(f"Error: analysis bundle not found: {bundle_path}", file=sys.stderr)
                return 3
            print(f"     Using bundle -> {bundle_path}")

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

        if args.with_ocr:
            step += 1
            t0 = time.monotonic()
            _print_step(step, total_steps, f"OCR ({args.text_device})")
            _run_ocr(video, bundle_path, gpu=(args.text_device == "cuda"))
            print(f"     Took {time.monotonic() - t0:.1f}s")

        if args.with_semantic:
            step += 1
            t0 = time.monotonic()
            _print_step(step, total_steps, f"Semantic text scoring ({args.text_device})")
            _run_semantic(bundle_path, device=args.text_device)
            print(f"     Took {time.monotonic() - t0:.1f}s")

        step += 1
        t0 = time.monotonic()
        _print_step(step, total_steps, "Fusion")
        _run_fusion(bundle_path, segments_path, args.min_segment_sec)
        print(f"     Took {time.monotonic() - t0:.1f}s")

        print()
        print(f"Done for {video.name}. Launch the player with:")
        print(f"  PYTHONPATH=. python -m player.player")
        print(f"  (open '{video}' in the file dialog; segments load from {segments_path.name})")
        return 0

    # decide between single video or directory processing
    if args.input_dir and args.video:
        print("Error: provide either a single video or --input-dir, not both", file=sys.stderr)
        return 4

    if args.input_dir:
        input_dir = args.input_dir.expanduser().resolve(strict=False)
        if not input_dir.is_dir():
            print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
            return 5

        video_exts = {".mp4", ".mov", ".mkv", ".avi"}
        videos = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in video_exts and p.is_file()])
        if not videos:
            print(f"No video files found in: {input_dir}", file=sys.stderr)
            return 6

        overall_failed = 0
        for vid in videos:
            rc = _process_single(vid)
            if rc != 0:
                overall_failed = 1
        return overall_failed

    if not args.video:
        print("Error: no input specified. Provide a video path or --input-dir.", file=sys.stderr)
        return 1

    return _process_single(args.video)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
