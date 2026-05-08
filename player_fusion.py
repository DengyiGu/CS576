"""
Drop-in replacement for run_video_segmentation() in player/player.py.

HOW TO INTEGRATE
In player/player.py, replace the existing run_video_segmentation function:

    def run_video_segmentation(video_path: Path) -> list[Segment]:
        simulated_step_delay_seconds = 0.8
        sleep(simulated_step_delay_seconds)
        ...
        segments = build_even_segments(duration_seconds)   # ← fake
        ...
        return segments

With this single import + call:

    from player_fusion_patch import run_video_segmentation

Or copy-paste the function body directly into player.py.

LOOKUP ORDER
When a video is loaded, the player looks for a pre-computed segments file in the following order:

  1. data/output/<video_stem>_segments.json   (standard fusion output path)
  2. <same_dir_as_video>/<video_stem>_segments.json

If found, segments are loaded instantly (no processing delay).
If not found, the full fusion pipeline runs live (visual analysis + fuse).
If the fusion pipeline fails for any reason, the player falls back to build_full_content_segment() so it never 
crashes.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


from player.segments import Segment, build_full_content_segment, build_segment_from_payload, probe_video_duration_seconds

TEXT_MODEL_DEVICE = os.environ.get("CS576_TEXT_MODEL_DEVICE", "cuda").strip().lower()
USE_CUDA_FOR_TEXT_MODELS = TEXT_MODEL_DEVICE == "cuda"

# Candidate locations for a pre-computed segments file

def _find_segments_file(video_path: Path) -> Path | None:
    stem = video_path.stem
    candidates = [
        Path("data/output") / f"{stem}_segments.json",
        video_path.parent / f"{stem}_segments.json",
        Path("data/output") / f"{stem.replace('_visual_track', '')}_segments.json",
    ]
    for c in candidates:
        try:
            if c.is_file():
                return c.resolve()
        except OSError:
            continue
    return None


# Load segments from a pre-computed segments JSON

def _load_segments_from_json(path: Path) -> list[Segment]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_segments: list[dict[str, Any]] = data.get("segments", [])
    segments: list[Segment] = []
    for i, payload in enumerate(raw_segments):
        seg = build_segment_from_payload(payload, i)
        if seg is not None:
            segments.append(seg)
    return segments


# Run fusion live (visual analysis + fuse)

def _run_fusion_live(video_path: Path) -> list[Segment]:
    from visual.analyze import build_analysis_bundle
    from fusion.fuse import fuse_bundle_to_segments

    print(f"[fusion] Running visual analysis on {video_path.name} ...", file=sys.stderr)
    bundle = build_analysis_bundle(video_path)

    # Wire in audio analysis
    try:
        from audio.analyze import analyze_audio
        print(f"[fusion] Running audio analysis on {video_path.name} ...", file=sys.stderr)
        audio_windows, _ = analyze_audio(video_path=video_path)
        bundle.audio_windows = audio_windows
        print(f"[fusion] Got {len(audio_windows)} audio windows.", file=sys.stderr)
    except Exception as e:
        print(f"[fusion] Audio analysis unavailable, skipping: {e}", file=sys.stderr)

    # Wire in speech recognition if the module is available
    try:
        from Automatic_speech_recognition.segment_text_analyzer import build_speech_spans
        print(f"[fusion] Running speech recognition on {video_path.name} ...", file=sys.stderr)
        bundle.speech_spans = build_speech_spans(
            video_path,
            device="cpu",
            compute_type="int8",
            vad=True,
        )
        print(f"[fusion] Got {len(bundle.speech_spans)} speech spans.", file=sys.stderr)
    except Exception as e:
        print(f"[fusion] Speech recognition unavailable, skipping: {e}", file=sys.stderr)

    # Optional OCR: sample only visually suspicious windows, so it adds text
    # evidence for silent/product ads without scanning every frame.
    try:
        from ocr.analyze import build_ocr_spans

        ocr_candidate_times = [
            0.5 * (window.t0 + window.t1)
            for window in bundle.visual.windows
            if (
                window.high_text_density
                or window.visual_hypothesis in {"graphics_heavy", "static"}
                or window.palette_delta > 0.35
            )
        ]
        text_device = "cuda" if USE_CUDA_FOR_TEXT_MODELS else "cpu"
        print(f"[fusion] Running OCR on {video_path.name} ({text_device}) ...", file=sys.stderr)
        ocr_spans = build_ocr_spans(
            video_path,
            candidate_times=ocr_candidate_times,
            sample_every_sec=10.0,
            max_frames=260,
            gpu=USE_CUDA_FOR_TEXT_MODELS,
        )
        bundle.speech_spans.extend(ocr_spans)
        print(f"[fusion] Got {len(ocr_spans)} OCR text spans.", file=sys.stderr)
    except Exception as e:
        print(f"[fusion] OCR unavailable, skipping: {e}", file=sys.stderr)

    # Optional semantic text scoring: merge short ASR/OCR spans into longer
    # context windows, then add ad and structure scores for fusion to consume.
    try:
        from semantic.analyze import build_semantic_ad_spans, build_semantic_structure_spans

        text_device = "cuda" if USE_CUDA_FOR_TEXT_MODELS else "cpu"
        print(f"[fusion] Running semantic ad scoring on {len(bundle.speech_spans)} text spans ({text_device}) ...", file=sys.stderr)
        semantic_spans = build_semantic_ad_spans(
            bundle.speech_spans,
            device="cuda" if USE_CUDA_FOR_TEXT_MODELS else "cpu",
        )
        bundle.speech_spans.extend(semantic_spans)
        print(f"[fusion] Got {len(semantic_spans)} semantic ad spans.", file=sys.stderr)
        print(f"[fusion] Running semantic structure scoring on {len(bundle.speech_spans)} text spans ({text_device}) ...", file=sys.stderr)
        structure_spans = build_semantic_structure_spans(
            bundle.speech_spans,
            device="cuda" if USE_CUDA_FOR_TEXT_MODELS else "cpu",
        )
        bundle.speech_spans.extend(structure_spans)
        print(f"[fusion] Got {len(structure_spans)} semantic structure spans.", file=sys.stderr)
    except Exception as e:
        print(f"[fusion] Semantic text scoring unavailable, skipping: {e}", file=sys.stderr)

    print(f"[fusion] Fusing {len(bundle.visual.windows)} windows ...", file=sys.stderr)
    raw_segments = fuse_bundle_to_segments(bundle)

    # Persist for next time
    out_dir = Path("data/output")
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_path.stem}_segments.json"
        from fusion.fuse import write_segments_json
        write_segments_json(raw_segments, out_path)
        print(f"[fusion] Saved segments → {out_path}", file=sys.stderr)
    except Exception as e:
        print(f"[fusion] Warning: could not save segments.json: {e}", file=sys.stderr)

    segments: list[Segment] = []
    for i, payload in enumerate(raw_segments):
        seg = build_segment_from_payload(payload, i)
        if seg is not None:
            segments.append(seg)
    return segments


# Public entry point — replaces run_video_segmentation in player.py

def run_video_segmentation(video_path: Path) -> list[Segment]:
    """
    Replacement for the stub run_video_segmentation() in player/player.py.

    Priority:
      1. Load pre-computed segments.json if it exists next to the video or in data/output/.
      2. Otherwise, run visual analysis + fusion live and cache the result.
      3. If everything fails, fall back to build_full_content_segment() so the player still works for demo 
         purposes.
    """
    # 1. Pre-computed file
    segments_file = _find_segments_file(video_path)
    if segments_file is not None:
        try:
            segments = _load_segments_from_json(segments_file)
            if segments:
                print(
                    f"[fusion] Loaded {len(segments)} segments from {segments_file.name}",
                    file=sys.stderr,
                )
                return segments
        except Exception as e:
            print(f"[fusion] Warning: could not load {segments_file}: {e}", file=sys.stderr)

    # 2. Live fusion
    try:
        return _run_fusion_live(video_path)
    except Exception as e:
        print(f"[fusion] Live fusion failed: {e}", file=sys.stderr)

    # 3. Graceful fallback — fake even segments so the player still opens
    print("[fusion] Falling back to demo segments.", file=sys.stderr)
    duration = probe_video_duration_seconds(video_path)
    return build_full_content_segment(duration)
