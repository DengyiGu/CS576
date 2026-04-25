from __future__ import annotations

import argparse
import sys
from pathlib import Path

from visual.analyze import analyze_visual, build_analysis_bundle, write_analysis_bundle_json, write_visual_track_json
from visual.video_info_dataset import list_mp4s_under_data_input, load_doc_and_resolve_video, write_reference_ads_json

_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


def _list_candidate_videos_in_data_input() -> list[Path]:
    base = Path("data/input")
    if not base.is_dir():
        return []
    found: list[Path] = []
    entries = sorted(base.iterdir())
    ei = 0
    while ei < len(entries):
        path = entries[ei]
        if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS:
            found.append(path)
        ei += 1
    return found


def _resolve_video_arg(raw: Path) -> Path:
    try:
        return raw.expanduser().resolve(strict=False)
    except OSError as exc:
        print(f"Error: invalid --video path {raw}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _exit_video_not_found(video: Path) -> None:
    print(f"Error: video file not found:\n  {video}", file=sys.stderr)
    name_upper = video.name.upper()
    if "YOUR" in name_upper:
        print(
            '\nYou used the placeholder name "YOUR.mp4". Replace it with your real file, e.g.\n'
            "  PYTHONPATH=. python3 -m visual_analyze \\\n"
            '    --video "data/input/mylecture.mp4" \\\n'
            '    --out "data/output/mylecture_visual_track.json" \\\n'
            '    --bundle-out "data/output/mylecture_analysis_bundle.json"',
            file=sys.stderr,
        )
    candidates = _list_candidate_videos_in_data_input()
    if candidates:
        print("\nVideo files currently under data/input/:", file=sys.stderr)
        ci = 0
        cn = len(candidates)
        while ci < cn:
            print(f"  {candidates[ci]}", file=sys.stderr)
            ci += 1
        print(
            f'\nExample:\n  PYTHONPATH=. python3 -m visual_analyze --video "{candidates[0]}" --out data/output/track.json',
            file=sys.stderr,
        )
    else:
        print(
            "\nPut a video under data/input/ (see data/README.md), or use:\n"
            "  --video-info video_info/test_001.json --videos-root <folder_with_test_001.mp4>",
            file=sys.stderr,
        )
    raise SystemExit(2)


def _exit_video_info_json_missing(path: Path) -> None:
    print(f"Error: --video-info file not found:\n  {path}", file=sys.stderr)
    raise SystemExit(2)


def _run_analysis(
    video: Path,
    *,
    out: Path | None,
    bundle_out: Path | None,
    sample_fps: float,
    window_sec: float,
    scenedetect_threshold: float,
    resize_max_width: int,
) -> tuple[Path, int]:
    out_path = out if out is not None else video.with_name(f"{video.stem}_visual_track.json")
    track = analyze_visual(
        video,
        sample_fps=sample_fps,
        window_sec=window_sec,
        scenedetect_threshold=scenedetect_threshold,
        resize_max_width=resize_max_width,
    )
    write_visual_track_json(track, out_path)

    if bundle_out is not None:
        bundle = build_analysis_bundle(video, track=track)
        write_analysis_bundle_json(bundle, bundle_out.resolve())

    return out_path.resolve(), len(track.windows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract visual/motion features to visual_track.json (and optional AnalysisBundle).",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=Path, default=None, help="Input video file path.")
    src.add_argument(
        "--video-info",
        type=Path,
        default=None,
        help="Path to video_info/*.json (stitched video + ad timeline). Implies resolving --videos-root/<video_filename>.",
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=Path("data/input"),
        help="Directory containing test_*.mp4 when using --video-info (default: data/input).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for visual_track.json (default: next to video, or data/output/<stem>_visual_track.json with --video-info).",
    )
    parser.add_argument(
        "--bundle-out",
        type=Path,
        default=None,
        help="If set, also write full AnalysisBundle JSON for fusion (visual + empty audio/speech placeholders).",
    )
    parser.add_argument(
        "--reference-out",
        type=Path,
        default=None,
        help="With --video-info: write ground-truth ad intervals from timeline (player-shaped JSON).",
    )
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Target analysis sampling rate.")
    parser.add_argument("--window-sec", type=float, default=1.0, help="Feature window length in seconds.")
    parser.add_argument(
        "--scenedetect-threshold",
        type=float,
        default=27.0,
        help="ContentDetector threshold (lower = more cuts).",
    )
    parser.add_argument("--resize-max-width", type=int, default=320, help="Resize frames to at most this width.")
    args = parser.parse_args(argv)

    if args.video_info is not None:
        info_path = _resolve_video_arg(args.video_info)
        if not info_path.is_file():
            _exit_video_info_json_missing(info_path)

        videos_root = args.videos_root.expanduser().resolve()
        doc, video, searched = load_doc_and_resolve_video(info_path, videos_root)
        if video is None or not video.is_file():
            print(
                f"Error: stitched video not found for {info_path.name}.\n"
                f"Expected a file named `{doc.primary_video_basename()}` in one of these locations (checked in order):",
                file=sys.stderr,
            )
            si = 0
            sn = len(searched)
            while si < sn:
                print(f"  {searched[si]}", file=sys.stderr)
                si += 1
            hints = list_mp4s_under_data_input()
            if hints:
                print("\nFound these .mp4 files under data/input/ (check spelling and folder):", file=sys.stderr)
                hi = 0
                hn = len(hints)
                while hi < hn:
                    print(f"  {hints[hi]}", file=sys.stderr)
                    hi += 1
            print(
                "\nDo one of the following:\n"
                f"  • Put `{doc.primary_video_basename()}` directly in `data/input/`, or in `data/input/videos_with_ads/`, or\n"
                "  • Put all `test_*.mp4` under `videos_with_ad/` at the repo root, or\n"
                "  • Pass `--videos-root /path/to/folder` that already contains that file.\n\n"
                "Quick symlink layout (after MP4s are in videos_with_ad/):\n"
                "  bash scripts/sync_videos_with_ad_to_data_input.sh",
                file=sys.stderr,
            )
            raise SystemExit(2)

        out = args.out
        if out is None:
            out_dir = Path("data/output")
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"{info_path.stem}_visual_track.json"

        out_resolved, n_windows = _run_analysis(
            video,
            out=out,
            bundle_out=args.bundle_out,
            sample_fps=args.sample_fps,
            window_sec=args.window_sec,
            scenedetect_threshold=args.scenedetect_threshold,
            resize_max_width=args.resize_max_width,
        )

        if args.reference_out is not None:
            write_reference_ads_json(doc, args.reference_out.resolve())
            print(f"Wrote reference ads {args.reference_out.resolve()}.")

        print(f"Wrote {out_resolved} ({n_windows} windows).")
        if args.bundle_out is not None:
            print(f"Wrote bundle {args.bundle_out.resolve()}.")
        return 0

    video = _resolve_video_arg(args.video)
    if not video.is_file():
        _exit_video_not_found(video)

    out_resolved, n_windows = _run_analysis(
        video,
        out=args.out,
        bundle_out=args.bundle_out,
        sample_fps=args.sample_fps,
        window_sec=args.window_sec,
        scenedetect_threshold=args.scenedetect_threshold,
        resize_max_width=args.resize_max_width,
    )
    print(f"Wrote {out_resolved} ({n_windows} windows).")
    if args.bundle_out is not None:
        print(f"Wrote bundle {args.bundle_out.resolve()}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
