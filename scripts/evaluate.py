"""
Evaluate fusion output against the ground-truth ad intervals in video_info/*.json.

For each test_00N, it:
  - Loads the reference ad segments from video_info/test_00N.json
  - Loads the predicted segments from data/output/test_00N_segments.json
  - Computes precision, recall, F1, and IoU for advertisement detection

Usage
  # Evaluate all test cases that have been processed:
  PYTHONPATH=. python scripts/evaluate.py

  # Evaluate a specific test case:
  PYTHONPATH=. python scripts/evaluate.py --test test_001

  # Evaluate with a custom segments directory:
  PYTHONPATH=. python scripts/evaluate.py --segments-dir data/output

Metrics:
  Temporal Precision — of all the time we predicted as "ad", what fraction actually was an ad?

  Temporal Recall — of all actual ad time, what fraction did we catch?

  F1 — harmonic mean of precision and recall

  Mean Segment IoU — for each predicted ad segment, the max overlap with any reference ad, 
                     averaged across predictions. This measures how well individual segment
                     boundaries are localised, not just whether ad time was found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Loading helpers
def _load_reference_ads(video_info_path: Path) -> list[dict]:
    """Return list of {start, end} dicts for all ad segments in a video_info JSON."""
    from schemas.video_info import load_video_info_doc, reference_ad_segments_player_shape
    doc = load_video_info_doc(video_info_path)
    return reference_ad_segments_player_shape(doc)  # already {start, end, label, kind, source}


def _load_predicted_ads(segments_path: Path) -> list[dict]:
    """Return list of {start, end} dicts for all non-content segments in a segments JSON."""
    data = json.loads(segments_path.read_text(encoding="utf-8"))
    all_segs = data.get("segments", [])
    # For evaluation we focus on anything predicted as Advertisement.
    # Change this to kind == "non-content" for a broader non-content evaluation.
    return [s for s in all_segs if s.get("label") == "Advertisement"]


# Metric computation
def _overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    intersection = _overlap_seconds(a_start, a_end, b_start, b_end)
    union = (a_end - a_start) + (b_end - b_start) - intersection
    return intersection / union if union > 1e-9 else 0.0


def _temporal_precision_recall(
    predicted: list[dict],
    reference: list[dict],
    duration: float,
) -> tuple[float, float, float]:
    """
    Compute temporal precision and recall at 0.1-second resolution.
    Returns (precision, recall, f1).
    """
    if duration <= 0:
        return 0.0, 0.0, 0.0

    resolution = 0.1
    n_bins = int(duration / resolution) + 1

    pred_mask = [False] * n_bins
    ref_mask = [False] * n_bins

    for seg in predicted:
        i0 = int(seg["start"] / resolution)
        i1 = min(int(seg["end"] / resolution), n_bins - 1)
        for i in range(i0, i1 + 1):
            pred_mask[i] = True

    for seg in reference:
        i0 = int(seg["start"] / resolution)
        i1 = min(int(seg["end"] / resolution), n_bins - 1)
        for i in range(i0, i1 + 1):
            ref_mask[i] = True

    tp = sum(1 for i in range(n_bins) if pred_mask[i] and ref_mask[i])
    fp = sum(1 for i in range(n_bins) if pred_mask[i] and not ref_mask[i])
    fn = sum(1 for i in range(n_bins) if not pred_mask[i] and ref_mask[i])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _mean_segment_iou(predicted: list[dict], reference: list[dict]) -> float:
    """
    For each predicted segment, find the best-matching reference segment
    and compute their IoU.  Return the mean over all predictions.
    """
    if not predicted or not reference:
        return 0.0

    ious: list[float] = []
    for pred in predicted:
        best = max(
            _iou(pred["start"], pred["end"], ref["start"], ref["end"])
            for ref in reference
        )
        ious.append(best)
    return sum(ious) / len(ious)


# Per-test evaluation
def _evaluate_one(
    test_name: str,
    video_info_dir: Path,
    segments_dir: Path,
) -> dict | None:
    info_path = video_info_dir / f"{test_name}.json"
    seg_path = segments_dir / f"{test_name}_segments.json"

    if not info_path.is_file():
        print(f"  [{test_name}] Skipped — video_info not found: {info_path}")
        return None

    if not seg_path.is_file():
        print(f"  [{test_name}] Skipped — segments not found: {seg_path}")
        print(f"             Run: PYTHONPATH=. python -m fusion --video data/input/{test_name}.mp4")
        return None

    try:
        reference = _load_reference_ads(info_path)
    except Exception as e:
        print(f"  [{test_name}] Error loading reference: {e}")
        return None

    try:
        predicted = _load_predicted_ads(seg_path)
    except Exception as e:
        print(f"  [{test_name}] Error loading predictions: {e}")
        return None

    # Load duration from the segments JSON (or fall back to summing reference)
    seg_data = json.loads(seg_path.read_text(encoding="utf-8"))
    all_segs = seg_data.get("segments", [])
    duration = max((s["end"] for s in all_segs), default=0.0)
    if duration <= 0:
        duration = max((s["end"] for s in reference), default=300.0)

    precision, recall, f1 = _temporal_precision_recall(predicted, reference, duration)
    seg_iou = _mean_segment_iou(predicted, reference)

    ref_ad_sec = sum(s["end"] - s["start"] for s in reference)
    pred_ad_sec = sum(s["end"] - s["start"] for s in predicted)

    return {
        "test": test_name,
        "n_ref_ads": len(reference),
        "n_pred_ads": len(predicted),
        "ref_ad_sec": ref_ad_sec,
        "pred_ad_sec": pred_ad_sec,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_seg_iou": seg_iou,
    }


# Pretty printing
def _print_results(results: list[dict]) -> None:
    if not results:
        print("\nNo results to display.")
        return

    col_w = [10, 8, 8, 10, 10, 10, 8, 8, 13]
    headers = ["Test", "Ref Ads", "Pred Ads", "Ref Sec", "Pred Sec", "Precision", "Recall", "F1", "Mean Seg IoU"]

    def _row(*cols):
        return "  ".join(str(c).ljust(w) for c, w in zip(cols, col_w))

    print()
    print("=" * 90)
    print("  Advertisement Detection Evaluation")
    print("=" * 90)
    print(_row(*headers))
    print("-" * 90)
    for r in results:
        print(_row(
            r["test"],
            r["n_ref_ads"],
            r["n_pred_ads"],
            f"{r['ref_ad_sec']:.1f}s",
            f"{r['pred_ad_sec']:.1f}s",
            f"{r['precision']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['f1']:.3f}",
            f"{r['mean_seg_iou']:.3f}",
        ))

    if len(results) > 1:
        print("-" * 90)
        avg_p = sum(r["precision"] for r in results) / len(results)
        avg_r = sum(r["recall"] for r in results) / len(results)
        avg_f = sum(r["f1"] for r in results) / len(results)
        avg_iou = sum(r["mean_seg_iou"] for r in results) / len(results)
        print(_row(
            f"MEAN (n={len(results)})", "", "", "", "",
            f"{avg_p:.3f}", f"{avg_r:.3f}", f"{avg_f:.3f}", f"{avg_iou:.3f}",
        ))

    print("=" * 90)
    print()
    print("Metrics:")
    print("  Precision    — of predicted ad time, how much was truly an ad")
    print("  Recall       — of reference ad time, how much did we detect")
    print("  F1           — harmonic mean of precision and recall")
    print("  Mean Seg IoU — average best-match IoU per predicted segment")
    print()


# CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/evaluate.py",
        description="Evaluate fusion output against video_info ground truth.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=None,
        help="Single test name to evaluate, e.g. test_001. Default: all test_00* found.",
    )
    parser.add_argument(
        "--video-info-dir",
        type=Path,
        default=Path("video_info"),
        help="Directory containing test_00*.json files (default: video_info/).",
    )
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=Path("data/output"),
        help="Directory containing test_00*_segments.json files (default: data/output/).",
    )
    args = parser.parse_args(argv)

    video_info_dir = args.video_info_dir.expanduser().resolve()
    segments_dir = args.segments_dir.expanduser().resolve()

    if not video_info_dir.is_dir():
        print(f"Error: video_info directory not found: {video_info_dir}", file=sys.stderr)
        return 2

    if args.test:
        test_names = [args.test]
    else:
        # Auto-discover all test_00*.json
        test_names = sorted(p.stem for p in video_info_dir.glob("test_*.json"))
        if not test_names:
            print(f"No test_*.json files found in {video_info_dir}", file=sys.stderr)
            return 2

    print(f"\n[evaluate] Evaluating {len(test_names)} test(s) …")
    print(f"           video_info : {video_info_dir}")
    print(f"           segments   : {segments_dir}")

    results: list[dict] = []
    for name in test_names:
        result = _evaluate_one(name, video_info_dir, segments_dir)
        if result is not None:
            results.append(result)

    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
