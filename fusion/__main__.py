"""
CLI entry point for the fusion pipeline.

Two modes:
  1. Bundle mode  (fast, for iteration)
     python -m fusion --bundle data/output/test_001_analysis_bundle.json

  2. End-to-end mode  (runs visual analysis first, then fuses)
     python -m fusion --video data/input/test_001.mp4

In both modes --out controls where segments.json is written.
If --out is omitted, the file is written next to the bundle / video.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fusion.fuse import fuse_bundle_to_segments, load_bundle, write_segments_json


def _run_visual_and_fuse(
    video: Path,
    *,
    out: Path,
    bundle_out: Path | None,
    sample_fps: float,
    window_sec: float,
    min_segment_seconds: float,
) -> int:
    """Run the visual analyzer then immediately fuse. Used in end-to-end mode."""
    # Import here to avoid requiring cv2 when running in bundle-only mode
    try:
        from visual.analyze import analyze_visual, build_analysis_bundle, write_analysis_bundle_json
    except ImportError as exc:
        print(
            f"Error: visual analysis dependencies not available: {exc}\n"
            "Install opencv-python and scenedetect, or pre-compute a bundle with\n"
            "  PYTHONPATH=. python -m visual_analyze --video ... --bundle-out ...\n"
            "then run:\n"
            "  PYTHONPATH=. python -m fusion --bundle <bundle.json>",
            file=sys.stderr,
        )
        return 1

    print(f"[fusion] Running visual analysis on {video.name} …")
    bundle = build_analysis_bundle(video, track=analyze_visual(video, sample_fps=sample_fps, window_sec=window_sec))

    if bundle_out is not None:
        write_analysis_bundle_json(bundle, bundle_out.resolve())
        print(f"[fusion] Wrote analysis bundle → {bundle_out.resolve()}")

    print(f"[fusion] Fusing {len(bundle.visual.windows)} windows …")
    segments = fuse_bundle_to_segments(bundle, min_segment_seconds=min_segment_seconds)
    write_segments_json(segments, out)
    _print_summary(segments, out)
    return 0


def _print_summary(segments: list[dict], out: Path) -> None:
    content = [s for s in segments if s["kind"] == "content"]
    non_content = [s for s in segments if s["kind"] != "content"]
    content_sec = sum(s["end"] - s["start"] for s in content)
    non_content_sec = sum(s["end"] - s["start"] for s in non_content)

    print(f"\n[fusion] Wrote {len(segments)} segments → {out.resolve()}")
    print(f"         Content      : {len(content)} segments, {content_sec:.1f}s")
    print(f"         Non-content  : {len(non_content)} segments, {non_content_sec:.1f}s")
    print()

    # Print segment table
    print(f"  {'#':<4} {'Label':<20} {'Kind':<14} {'Start':>8}  {'End':>8}  {'Duration':>8}")
    print("  " + "-" * 70)
    for i, seg in enumerate(segments):
        dur = seg["end"] - seg["start"]
        print(
            f"  {i:<4} {seg['label']:<20} {seg['kind']:<14} "
            f"{seg['start']:>8.1f}s {seg['end']:>8.1f}s {dur:>8.1f}s"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fusion",
        description="Fuse multimodal analysis signals into labeled video segments.",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--bundle",
        type=Path,
        default=None,
        metavar="BUNDLE_JSON",
        help="Path to a pre-computed AnalysisBundle JSON (fast path).",
    )
    src.add_argument(
        "--video",
        type=Path,
        default=None,
        metavar="VIDEO_FILE",
        help="Path to a raw video file (runs visual analysis first).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="SEGMENTS_JSON",
        help="Output path for segments.json (default: next to bundle/video).",
    )
    parser.add_argument(
        "--bundle-out",
        type=Path,
        default=None,
        metavar="BUNDLE_JSON",
        help="(End-to-end mode only) Also write the AnalysisBundle JSON.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="Frame sample rate for visual analysis (end-to-end mode, default 2.0).",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.0,
        help="Window size in seconds for visual analysis (default 1.0).",
    )
    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=4.0,
        help="Minimum segment duration in seconds before smoothing absorbs it (default 4.0).",
    )

    args = parser.parse_args(argv)

    # End-to-end mode: video → visual analysis → fuse
    if args.video is not None:
        video = args.video.expanduser().resolve()
        if not video.is_file():
            print(f"Error: video file not found: {video}", file=sys.stderr)
            return 2
        out = args.out or video.with_name(f"{video.stem}_segments.json")
        return _run_visual_and_fuse(
            video,
            out=out,
            bundle_out=args.bundle_out,
            sample_fps=args.sample_fps,
            window_sec=args.window_sec,
            min_segment_seconds=args.min_segment_sec,
        )

    # Bundle mode: pre-computed bundle → fuse
    bundle_path = args.bundle.expanduser().resolve()
    if not bundle_path.is_file():
        print(f"Error: bundle file not found: {bundle_path}", file=sys.stderr)
        return 2

    out = args.out or bundle_path.with_name(
        bundle_path.stem.replace("_analysis_bundle", "") + "_segments.json"
    )

    print(f"[fusion] Loading bundle from {bundle_path.name} …")
    bundle = load_bundle(bundle_path)

    if bundle.visual is None or not bundle.visual.windows:
        print("Error: bundle contains no visual windows. Re-run visual_analyze first.", file=sys.stderr)
        return 1

    n_audio = len(bundle.audio_windows)
    n_speech = len(bundle.speech_spans)
    modalities_active = ["visual"]
    if n_audio > 0:
        modalities_active.append(f"audio ({n_audio} windows)")
    if n_speech > 0:
        modalities_active.append(f"speech ({n_speech} spans)")
    print(f"[fusion] Active modalities: {', '.join(modalities_active)}")
    print(f"[fusion] Fusing {len(bundle.visual.windows)} windows …")

    segments = fuse_bundle_to_segments(bundle, min_segment_seconds=args.min_segment_sec)
    write_segments_json(segments, out)
    _print_summary(segments, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
