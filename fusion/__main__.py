"""
CLI entry point for the fusion pipeline.

Two modes:
  Bundle mode (fast, for iteration after visual analysis is already done):
    python -m fusion --bundle data/output/test_001_analysis_bundle.json

  End-to-end mode (runs visual analysis first, then speech, then fuses):
    python -m fusion --video videos_with_ad/test_001.mp4

In both modes --out controls where segments.json is written.
If --out is omitted, the file is written next to the bundle or video.

Recommended parameters for best results:
    --min-segment-sec 20 --sample-fps 1.0 --window-sec 2.0

Always run with conda activate cs576 first so speech recognition is available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fusion.fuse import fuse_bundle_to_segments, load_bundle, write_segments_json


def _has_text_source(bundle, source: str) -> bool:
    return any((span.model_extra or {}).get("source") == source for span in bundle.speech_spans)


def _try_add_ocr(bundle, video: Path, *, use_cuda_text_models: bool = False) -> None:
    if bundle.visual is None or not bundle.visual.windows:
        return
    if _has_text_source(bundle, "ocr"):
        print("[fusion] OCR spans already present; skipping OCR.")
        return
    try:
        from ocr.analyze import build_ocr_spans
        candidate_times = [
            0.5 * (window.t0 + window.t1)
            for window in bundle.visual.windows
            if (
                window.high_text_density
                or window.visual_hypothesis in {"graphics_heavy", "static"}
                or window.palette_delta > 0.35
            )
        ]
        device_name = "cuda" if use_cuda_text_models else "cpu"
        print(f"[fusion] Running OCR on {video.name} ({device_name}) ...")
        ocr_spans = build_ocr_spans(
            video,
            candidate_times=candidate_times,
            sample_every_sec=10.0,
            max_frames=260,
            gpu=use_cuda_text_models,
        )
        bundle.speech_spans.extend(ocr_spans)
        print(f"[fusion] Got {len(ocr_spans)} OCR text spans.")
    except Exception as e:
        print(f"[fusion] OCR unavailable, running without: {e}")


def _try_add_semantic_text_scores(bundle, *, use_cuda_text_models: bool = False) -> None:
    if not bundle.speech_spans:
        return
    try:
        from semantic.analyze import build_semantic_ad_spans, build_semantic_structure_spans

        device_name = "cuda" if use_cuda_text_models else "cpu"
        if _has_text_source(bundle, "semantic"):
            print("[fusion] Semantic ad scores already present; skipping ad semantic scoring.")
        else:
            print(f"[fusion] Running semantic ad scoring on {len(bundle.speech_spans)} text spans ({device_name}) ...")
            semantic_spans = build_semantic_ad_spans(
                bundle.speech_spans,
                device="cuda" if use_cuda_text_models else "cpu",
            )
            bundle.speech_spans.extend(semantic_spans)
            print(f"[fusion] Got {len(semantic_spans)} semantic ad spans.")

        if _has_text_source(bundle, "semantic_structure"):
            print("[fusion] Semantic structure scores already present; skipping structure semantic scoring.")
        else:
            print(f"[fusion] Running semantic structure scoring on {len(bundle.speech_spans)} text spans ({device_name}) ...")
            structure_spans = build_semantic_structure_spans(
                bundle.speech_spans,
                device="cuda" if use_cuda_text_models else "cpu",
            )
            bundle.speech_spans.extend(structure_spans)
            print(f"[fusion] Got {len(structure_spans)} semantic structure spans.")
    except Exception as e:
        print(f"[fusion] Semantic text scoring unavailable, running without: {e}")


def _run_visual_and_fuse(
    video: Path,
    *,
    out: Path,
    bundle_out: Path | None,
    sample_fps: float,
    window_sec: float,
    min_segment_seconds: float,
    skip_speech: bool = False,
    use_cuda_text_models: bool = False,
    asr_device: str = "cpu",
    asr_compute_type: str = "int8",
    asr_vad: bool = True,
) -> int:
    """
    Run visual analysis, optionally speech recognition, then fuse.
    """
    try:
        from visual.analyze import analyze_visual, build_analysis_bundle, write_analysis_bundle_json
    except ImportError as exc:
        print(
            f"Error: visual analysis dependencies not available: {exc}\n"
            "Install opencv-python and scenedetect, or pre-compute a bundle with\n"
            "  PYTHONPATH=. python -m visual_analyze --video ... --bundle-out ...\n"
            "then run fusion in bundle mode:\n"
            "  PYTHONPATH=. python -m fusion --bundle <bundle.json>",
            file=sys.stderr,
        )
        return 1

    print(f"[fusion] Running visual analysis on {video.name} ...")
    bundle = build_analysis_bundle(video, track=analyze_visual(video, sample_fps=sample_fps, window_sec=window_sec))

    # Wire in audio analysis
    try:
        from audio.analyze import analyze_audio
        print(f"[fusion] Running audio analysis on {video.name} ...")
        audio_windows, _ = analyze_audio(video_path=video, window_sec=window_sec)
        bundle.audio_windows = audio_windows
        print(f"[fusion] Got {len(audio_windows)} audio windows.")
    except Exception as e:
        print(f"[fusion] Audio analysis unavailable, running without: {e}")

    if not skip_speech:
        try:
            from Automatic_speech_recognition.segment_text_analyzer import build_speech_spans
            print(
                f"[fusion] Running speech recognition on {video.name} "
                f"({asr_device}, {asr_compute_type}, vad={asr_vad}) ..."
            )
            bundle.speech_spans = build_speech_spans(
                video,
                device=asr_device,
                compute_type=asr_compute_type,
                vad=asr_vad,
            )
            print(f"[fusion] Got {len(bundle.speech_spans)} speech spans.")
        except Exception as e:
            print(f"[fusion] Speech recognition unavailable, running without: {e}")

    _try_add_ocr(bundle, video, use_cuda_text_models=use_cuda_text_models)
    _try_add_semantic_text_scores(bundle, use_cuda_text_models=use_cuda_text_models)

    if bundle_out is not None:
        write_analysis_bundle_json(bundle, bundle_out.resolve())
        print(f"[fusion] Wrote analysis bundle -> {bundle_out.resolve()}")

    n_audio = len(bundle.audio_windows)
    n_speech = len(bundle.speech_spans)
    modalities_active = ["visual"]
    if n_audio > 0:
        modalities_active.append(f"audio ({n_audio} windows)")
    if n_speech > 0:
        modalities_active.append(f"speech ({n_speech} spans)")
    print(f"[fusion] Active modalities: {', '.join(modalities_active)}")
    print(f"[fusion] Fusing {len(bundle.visual.windows)} windows ...")

    segments = fuse_bundle_to_segments(bundle, min_segment_seconds=min_segment_seconds)
    write_segments_json(segments, out)
    _print_summary(segments, out)
    return 0


def _print_summary(segments: list[dict], out: Path) -> None:
    content = [s for s in segments if s["kind"] == "content"]
    non_content = [s for s in segments if s["kind"] != "content"]
    content_sec = sum(s["end"] - s["start"] for s in content)
    non_content_sec = sum(s["end"] - s["start"] for s in non_content)

    print(f"\n[fusion] Wrote {len(segments)} segments -> {out.resolve()}")
    print(f"         Content      : {len(content)} segments, {content_sec:.1f}s")
    print(f"         Non-content  : {len(non_content)} segments, {non_content_sec:.1f}s")
    print()

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
        help="Frame sample rate for visual analysis (default 2.0, recommend 1.0 for speed).",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=1.0,
        help="Window size in seconds for visual analysis (default 1.0, recommend 2.0 for speed).",
    )
    parser.add_argument(
        "--skip-speech",
        action="store_true",
        default=False,
        help="Skip speech recognition even if available (faster, visual-only run).",
    )
    parser.add_argument(
        "--min-segment-sec",
        type=float,
        default=4.0,
        help="Segments shorter than this get absorbed by neighbors (default 4.0, recommend 20.0).",
    )
    parser.add_argument(
        "--cuda-text-models",
        action="store_true",
        default=False,
        help="Run OCR and semantic text scoring on CUDA if your local CUDA/PyTorch setup supports it.",
    )
    parser.add_argument(
        "--asr-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Speech recognition device for faster-whisper (default cpu).",
    )
    parser.add_argument(
        "--asr-compute-type",
        default="int8",
        help="Speech recognition compute type, such as int8, int8_float16, float16, or float32.",
    )
    parser.add_argument(
        "--asr-vad",
        action="store_true",
        default=True,
        help="Enable VAD for speech recognition in end-to-end video mode (default on).",
    )
    parser.add_argument(
        "--no-asr-vad",
        action="store_false",
        dest="asr_vad",
        help="Disable VAD for speech recognition.",
    )

    args = parser.parse_args(argv)

    # End-to-end mode: video -> visual analysis -> speech -> fuse
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
            skip_speech=args.skip_speech,
            use_cuda_text_models=args.cuda_text_models,
            asr_device=args.asr_device,
            asr_compute_type=args.asr_compute_type,
            asr_vad=args.asr_vad,
        )

    # Bundle mode: load pre-computed bundle, optionally add speech, then fuse
    bundle_path = args.bundle.expanduser().resolve()
    if not bundle_path.is_file():
        print(f"Error: bundle file not found: {bundle_path}", file=sys.stderr)
        return 2

    out = args.out or bundle_path.with_name(
        bundle_path.stem.replace("_analysis_bundle", "") + "_segments.json"
    )

    print(f"[fusion] Loading bundle from {bundle_path.name} ...")
    bundle = load_bundle(bundle_path)

    if bundle.visual is None or not bundle.visual.windows:
        print("Error: bundle contains no visual windows. Re-run visual_analyze first.", file=sys.stderr)
        return 1

    # Try to add audio if the bundle doesn't already have it
    if not bundle.audio_windows:
        video_path_str = bundle.video_path
        if video_path_str:
            video_for_audio = Path(video_path_str)
            if video_for_audio.is_file():
                try:
                    from audio.analyze import analyze_audio
                    print(f"[fusion] Running audio analysis on {video_for_audio.name} ...")
                    audio_windows, _ = analyze_audio(video_path=video_for_audio, window_sec=args.window_sec)
                    bundle.audio_windows = audio_windows
                    print(f"[fusion] Got {len(audio_windows)} audio windows.")
                except Exception as e:
                    print(f"[fusion] Audio analysis unavailable, running without: {e}")

    # Try to add speech if the bundle doesn't already have it
    if not args.skip_speech and not bundle.speech_spans:
        video_path_str = bundle.video_path
        if video_path_str:
            video_for_speech = Path(video_path_str)
            if video_for_speech.is_file():
                try:
                    from Automatic_speech_recognition.segment_text_analyzer import build_speech_spans
                    print(
                        f"[fusion] Running speech recognition on {video_for_speech.name} "
                        f"({args.asr_device}, {args.asr_compute_type}, vad={args.asr_vad}) ..."
                    )
                    bundle.speech_spans = build_speech_spans(
                        video_for_speech,
                        device=args.asr_device,
                        compute_type=args.asr_compute_type,
                        vad=args.asr_vad,
                    )
                    print(f"[fusion] Got {len(bundle.speech_spans)} speech spans.")
                except Exception as e:
                    print(f"[fusion] Speech recognition unavailable, running without: {e}")

    video_path_str = bundle.video_path
    if video_path_str:
        video_for_ocr = Path(video_path_str)
        if video_for_ocr.is_file():
            _try_add_ocr(bundle, video_for_ocr, use_cuda_text_models=args.cuda_text_models)

    _try_add_semantic_text_scores(bundle, use_cuda_text_models=args.cuda_text_models)

    n_audio = len(bundle.audio_windows)
    n_speech = len(bundle.speech_spans)
    modalities_active = ["visual"]
    if n_audio > 0:
        modalities_active.append(f"audio ({n_audio} windows)")
    if n_speech > 0:
        modalities_active.append(f"speech ({n_speech} spans)")
    print(f"[fusion] Active modalities: {', '.join(modalities_active)}")
    print(f"[fusion] Fusing {len(bundle.visual.windows)} windows ...")

    segments = fuse_bundle_to_segments(bundle, min_segment_seconds=args.min_segment_sec)
    write_segments_json(segments, out)
    _print_summary(segments, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
