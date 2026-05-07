"""Boundary quality of K-selected intervals vs GT ad boundaries.

For each video, prints the K=5 chosen intervals together with their
boundary signals (palette_delta from visual, loudness_jump from
audio, both at the start and end), the GT ad set, and a TP/FP label
based on midpoint overlap.

Goal: see whether the boundary signals (palette_delta * loudness_jump
or similar) cleanly separate true ads from interior false-positive
picks like test_003's 490-552 and 1022-1126.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from fusion.fuse import (  # noqa: E402
    SMOOTH_HALF_WIN,
    _compute_edge_scores,
    _compute_foreignness_scores,
    _find_best_k_ads,
    _loudness_jump_score,
    _smooth,
    load_bundle,
)
from schemas.video_info import (  # noqa: E402
    load_video_info_doc,
    reference_ad_segments_player_shape,
)

OUT_DIR = ROOT / "data" / "output"
INFO_DIR = ROOT / "video_info"
TESTS = ["test_001", "test_002", "test_003", "test_004", "test_005", "test_010"]


def _palette_delta_at(t: float, windows: list) -> float:
    if not windows:
        return 0.0
    sec = windows[0].t1 - windows[0].t0 or 2.0
    idx = int(t / sec)
    idx = max(0, min(idx, len(windows) - 1))
    return float(windows[idx].palette_delta or 0.0)


def _midpoint_in_any(mid: float, intervals: list[tuple[float, float]]) -> int:
    for i, (s, e) in enumerate(intervals):
        if s <= mid < e:
            return i
    return -1


def main() -> int:
    for test in TESTS:
        bundle = load_bundle(OUT_DIR / f"{test}_analysis_bundle.json")
        info_path = INFO_DIR / f"{test}.json"
        info_doc = load_video_info_doc(info_path)
        gt = reference_ad_segments_player_shape(info_doc)
        gt_intervals = [(seg["start"], seg["end"]) for seg in gt]

        windows = bundle.visual.windows
        duration = bundle.visual.duration_sec or bundle.duration_sec

        raw_foreign = _compute_foreignness_scores(
            windows, bundle.audio_windows, bundle.speech_spans, duration
        )
        smooth_foreign = _smooth(raw_foreign, SMOOTH_HALF_WIN)
        raw_edge = _compute_edge_scores(
            windows, bundle.audio_windows, bundle.speech_spans, duration
        )
        smooth_edge = _smooth(raw_edge, SMOOTH_HALF_WIN)

        # Use auto-K to mirror what the pipeline actually picks.
        from fusion.fuse import _select_num_ads_auto, MAX_NUM_ADS, MIN_MARGINAL_RATIO

        K, intervals = _select_num_ads_auto(
            smooth_edge,
            smooth_foreign,
            windows,
            max_k=MAX_NUM_ADS,
            min_marginal_ratio=MIN_MARGINAL_RATIO,
        )
        sec = windows[0].t1 - windows[0].t0 or 2.0
        print(f"# auto-K picked K={K}")

        norm_edge = smooth_edge / (smooth_edge.max() + 1e-9)
        norm_foreign = smooth_foreign / (smooth_foreign.max() + 1e-9)

        print(f"=== {test}  GT={[(round(s,1), round(e,1)) for s, e in gt_intervals]} ===")
        print(f"{'kind':<3} {'start':>7} {'end':>7}  {'edge_s':>7} {'edge_e':>7}  {'pal_s':>6} {'pal_e':>6}  {'lj_s':>6} {'lj_e':>6}  {'int':>5}")
        for s_idx, e_idx in intervals:
            t0 = s_idx * sec
            t1 = e_idx * sec
            mid = 0.5 * (t0 + t1)
            tp_idx = _midpoint_in_any(mid, gt_intervals)
            kind = f"T{tp_idx}" if tp_idx >= 0 else "FP"

            edge_s = float(norm_edge[s_idx])
            edge_e = float(norm_edge[min(e_idx, len(norm_edge) - 1)])
            pal_s = _palette_delta_at(t0, windows)
            pal_e = _palette_delta_at(t1, windows)
            lj_s = _loudness_jump_score(t0, bundle.audio_windows)
            lj_e = _loudness_jump_score(t1, bundle.audio_windows)
            interior = float(norm_foreign[s_idx:e_idx].mean()) if e_idx > s_idx else 0.0

            print(
                f"{kind:<3} {t0:>7.1f} {t1:>7.1f}  "
                f"{edge_s:>7.3f} {edge_e:>7.3f}  "
                f"{pal_s:>6.3f} {pal_e:>6.3f}  "
                f"{lj_s:>6.3f} {lj_e:>6.3f}  "
                f"{interior:>5.3f}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
