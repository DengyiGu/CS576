"""
Sweep the marginal-gain ratio for auto-K selection across all 6 videos.

For each ratio r in RATIOS, run the DP for K = 1..6 on each cached bundle,
report the K it would have chosen, and the resulting F1 / IoU vs the ground
truth. This tells us whether any ratio gives a per-video adaptive rule that
beats fixed K=3.

Usage
  PYTHONPATH=. python scripts/sweep_auto_ratio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
VIDEO_INFO_DIR = ROOT / "video_info"

TESTS = ["test_001", "test_002", "test_003", "test_004", "test_005", "test_010"]
MAX_K = 6
RATIOS = [0.55, 0.70, 0.80, 0.85, 0.90, 0.95]


def _per_k_totals(bundle):
    windows  = bundle.visual.windows
    duration = bundle.visual.duration_sec or bundle.duration_sec

    raw_foreign = _compute_foreignness_scores(
        windows, bundle.audio_windows, bundle.speech_spans, duration
    )
    smooth_foreign = _smooth(raw_foreign, SMOOTH_HALF_WIN)
    raw_edge = _compute_edge_scores(
        windows, bundle.audio_windows, bundle.speech_spans, duration
    )
    smooth_edge = _smooth(raw_edge, SMOOTH_HALF_WIN)

    totals = [0.0]
    for k in range(1, MAX_K + 1):
        total_k, _ = _find_best_k_ads(smooth_edge, smooth_foreign, windows, k)
        totals.append(float(total_k))
    return totals


def _select_k_for_ratio(totals: list[float], ratio: float) -> int:
    """Largest K in [1, MAX_K] where g_K >= ratio * g_1, contiguously."""
    if len(totals) < 2:
        return 0
    g1 = totals[1] - totals[0]
    if g1 <= 0:
        return 1
    chosen = 1
    for k in range(2, len(totals)):
        gk = totals[k] - totals[k - 1]
        if gk >= ratio * g1:
            chosen = k
        else:
            break
    return chosen


def _evaluate_at_k(test: str, bundle, k: int) -> tuple[float, float, int]:
    segments = fuse_bundle_to_segments(bundle, num_ads=k)
    out = OUT_DIR / f"{test}_segments.json"
    write_segments_json(segments, out)
    result = _evaluate_one(test, VIDEO_INFO_DIR, OUT_DIR)
    return float(result["f1"]), float(result["mean_seg_iou"]), int(result["n_pred_ads"])


def main() -> int:
    print("Computing per-K totals & ground-truth F1/IoU for each video...")
    print()

    grid: dict[str, dict] = {}
    for test in TESTS:
        bundle_path = OUT_DIR / f"{test}_analysis_bundle.json"
        if not bundle_path.is_file():
            continue
        bundle = load_bundle(bundle_path)

        totals = _per_k_totals(bundle)
        marginals = [totals[k] - totals[k - 1] for k in range(1, len(totals))]

        f1_iou_at_k: dict[int, tuple[float, float, int]] = {}
        for k in range(1, MAX_K + 1):
            f1_iou_at_k[k] = _evaluate_at_k(test, bundle, k)

        grid[test] = {
            "totals": totals, "marginals": marginals, "f1_iou_at_k": f1_iou_at_k,
        }

    print("=" * 110)
    print(" Marginal-gain series (g_k = total_k - total_{k-1}); ratios are g_k / g_1")
    print("=" * 110)
    print(f"{'test':<10} {'g1':>7} {'g2/g1':>8} {'g3/g1':>8} {'g4/g1':>8} {'g5/g1':>8} {'g6/g1':>8}")
    for test in TESTS:
        if test not in grid:
            continue
        m = grid[test]["marginals"]
        g1 = m[0] if m and m[0] > 0 else 1e-9
        ratios_for_test = [m[k - 1] / g1 for k in range(1, len(m) + 1)]
        print(
            f"{test:<10} {m[0]:>7.3f} "
            + " ".join(f"{r:>8.3f}" for r in ratios_for_test[1:])
        )
    print()

    print("=" * 110)
    print(" F1 at each K (ground truth)")
    print("=" * 110)
    print(f"{'test':<10} " + " ".join(f"K={k:<2}{'':>3}" for k in range(1, MAX_K + 1)))
    for test in TESTS:
        if test not in grid:
            continue
        f1s = [grid[test]["f1_iou_at_k"][k][0] for k in range(1, MAX_K + 1)]
        print(f"{test:<10} " + " ".join(f"{f:>6.3f}" for f in f1s))
    print()

    print("=" * 110)
    print(" Auto-K selection by ratio (chosen K + resulting F1, IoU)")
    print("=" * 110)
    header = f"{'test':<10} "
    for r in RATIOS:
        header += f"r={r:.2f}: K F1 IoU       "
    print(header)
    sums = {r: {"f1": 0.0, "iou": 0.0, "n": 0} for r in RATIOS}
    for test in TESTS:
        if test not in grid:
            continue
        line = f"{test:<10} "
        for r in RATIOS:
            k = _select_k_for_ratio(grid[test]["totals"], r)
            f1, iou, _ = grid[test]["f1_iou_at_k"].get(k, (0.0, 0.0, 0))
            line += f"K={k} {f1:.3f}/{iou:.3f}    "
            sums[r]["f1"] += f1; sums[r]["iou"] += iou; sums[r]["n"] += 1
        print(line)
    print()
    print("Mean across videos:")
    for r in RATIOS:
        n = sums[r]["n"] or 1
        print(f"  ratio={r:.2f}  meanF1={sums[r]['f1']/n:.3f}  meanIoU={sums[r]['iou']/n:.3f}")
    print()

    # Restore K=3 segment files (matches sessions notes "v2" baseline state)
    print("Restoring segments files to K=3 (match cached baseline)...")
    for test in TESTS:
        if test not in grid:
            continue
        bundle = load_bundle(OUT_DIR / f"{test}_analysis_bundle.json")
        segments = fuse_bundle_to_segments(bundle, num_ads=3)
        write_segments_json(segments, OUT_DIR / f"{test}_segments.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
