"""Per-interval interior-mean foreignness for each K on each video.

Goal: see whether the *minimum* interior_mean across the K chosen ads
is a clean separator between "K is right" and "K is too many".

For each video and each K in [1, MAX], runs the DP, recovers the K
chosen intervals, and prints the per-interval interior_mean alongside
the gold F1 of that K. The hypothesis the diagnostic is testing: when
auto-K is right, every interval's interior_mean is high; when it's
overshooting, the worst interval's interior_mean drops below a floor
that the rest of the videos respect.
"""
from __future__ import annotations

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
    _smooth,
    fuse_bundle_to_segments,
    load_bundle,
    write_segments_json,
)
from scripts.evaluate import _evaluate_one  # noqa: E402

OUT_DIR = ROOT / "data" / "output"
INFO_DIR = ROOT / "video_info"
TESTS = ["test_001", "test_002", "test_003", "test_004", "test_005", "test_010"]
MAX_K = 5


def _per_interval_interior_means(
    smooth_foreign: np.ndarray,
    intervals: list[tuple[int, int]],
) -> list[float]:
    f_max = float(smooth_foreign.max())
    norm_foreign = smooth_foreign / (f_max + 1e-9)
    return [
        float(norm_foreign[s:e].mean()) if e > s else 0.0
        for s, e in intervals
    ]


def main() -> int:
    print(f"{'test':<10} {'K':>2} {'F1':>6}  {'interior_means_per_interval':<60}  {'min':>6}")
    print("-" * 100)
    for test in TESTS:
        bundle_path = OUT_DIR / f"{test}_analysis_bundle.json"
        if not bundle_path.is_file():
            continue
        bundle = load_bundle(bundle_path)

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

        for k in range(1, MAX_K + 1):
            _total, intervals = _find_best_k_ads(smooth_edge, smooth_foreign, windows, k)
            if not intervals:
                continue
            means = _per_interval_interior_means(smooth_foreign, intervals)
            min_mean = min(means) if means else 0.0

            # Evaluate F1 at this K so we can see "is this K close to optimal".
            segments = fuse_bundle_to_segments(bundle, num_ads=k)
            write_segments_json(segments, OUT_DIR / f"{test}_segments.json")
            result = _evaluate_one(test, INFO_DIR, OUT_DIR)
            f1 = float(result["f1"]) if result else 0.0

            means_str = " ".join(f"{m:.3f}" for m in means)
            print(f"{test:<10} {k:>2} {f1:>6.3f}  {means_str:<60}  {min_mean:>6.3f}")
        print()

    # Restore K=3 segment files
    for test in TESTS:
        bundle = load_bundle(OUT_DIR / f"{test}_analysis_bundle.json")
        segments = fuse_bundle_to_segments(bundle, num_ads=3)
        write_segments_json(segments, OUT_DIR / f"{test}_segments.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
