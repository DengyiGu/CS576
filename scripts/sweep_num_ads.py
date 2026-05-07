"""
One-off sweep: re-fuse all six cached bundles under several num_ads
configurations, write each variant's segments to a tagged path, and run the
existing evaluator on each variant.

Variants:
  K3    : num_ads=3 (current default; sanity check vs cached output)
  K4    : num_ads=4 (forces 4 ads everywhere)
  auto  : num_ads=None, max_num_ads=6, default marginal-gain ratio

Usage
  PYTHONPATH=. python scripts/sweep_num_ads.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion.fuse import (  # noqa: E402
    fuse_bundle_to_segments,
    load_bundle,
    write_segments_json,
)
from scripts.evaluate import _evaluate_one  # noqa: E402

OUT_DIR = ROOT / "data" / "output"
VIDEO_INFO_DIR = ROOT / "video_info"

VARIANTS = [
    ("K3",   {"num_ads": 3}),
    ("K4",   {"num_ads": 4}),
    ("auto", {"num_ads": None, "max_num_ads": 6}),
]

TESTS = ["test_001", "test_002", "test_003", "test_004", "test_005", "test_010"]


def _round(x: float | None, n: int = 3) -> float:
    if x is None:
        return float("nan")
    return round(float(x), n)


def main() -> int:
    rows: list[dict] = []
    for test in TESTS:
        bundle_path = OUT_DIR / f"{test}_analysis_bundle.json"
        if not bundle_path.is_file():
            print(f"[skip] {test}: bundle missing")
            continue
        bundle = load_bundle(bundle_path)
        for tag, kwargs in VARIANTS:
            segments = fuse_bundle_to_segments(bundle, **kwargs)
            tagged_path = OUT_DIR / f"{test}_segments_{tag}.json"
            write_segments_json(segments, tagged_path)

            # The evaluator looks at <test>_segments.json, so symlink the variant
            # into that path temporarily and call _evaluate_one in a controlled way.
            # _evaluate_one takes (test_name, video_info_dir, segments_dir), and
            # uses segments_dir/{test}_segments.json. Redirect by reading the
            # tagged file and dropping it in place.
            (OUT_DIR / f"{test}_segments.json").write_text(
                tagged_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            result = _evaluate_one(test, VIDEO_INFO_DIR, OUT_DIR)
            if result is None:
                continue
            n_pred = result["n_pred_ads"]
            f1 = _round(result["f1"])
            iou = _round(result["mean_seg_iou"])
            prec = _round(result["precision"])
            rec = _round(result["recall"])
            rows.append({
                "test": test, "variant": tag, "n_pred": n_pred,
                "precision": prec, "recall": rec, "f1": f1, "iou": iou,
            })
            print(f"[{test}] {tag:>4}: n_pred={n_pred} P={prec:.3f} R={rec:.3f} F1={f1:.3f} IoU={iou:.3f}")

    # Summary
    summary = {variant: {"f1": [], "iou": []} for variant, _ in VARIANTS}
    for row in rows:
        summary[row["variant"]]["f1"].append(row["f1"])
        summary[row["variant"]]["iou"].append(row["iou"])

    print()
    print("=" * 88)
    print("  SUMMARY (mean over all evaluated tests)")
    print("=" * 88)
    print(f"  {'Variant':<8} {'mean F1':>10} {'mean IoU':>10} {'n':>4}")
    for variant, _ in VARIANTS:
        f1s = summary[variant]["f1"]
        ious = summary[variant]["iou"]
        n = len(f1s)
        if n == 0:
            continue
        print(f"  {variant:<8} {sum(f1s)/n:>10.3f} {sum(ious)/n:>10.3f} {n:>4}")
    print("=" * 88)

    # Per-test grid
    print()
    print("Per-test F1:")
    print(f"  {'test':<10} " + " ".join(f"{tag:>7}" for tag, _ in VARIANTS) + "  n_pred")
    for test in TESTS:
        line = f"  {test:<10} "
        n_preds = []
        for tag, _ in VARIANTS:
            r = next((r for r in rows if r["test"] == test and r["variant"] == tag), None)
            if r is None:
                line += f"  {'-':>5} "
                continue
            line += f" {r['f1']:>6.3f} "
            n_preds.append((tag, r["n_pred"]))
        line += "  " + ", ".join(f"{tag}:{n}" for tag, n in n_preds)
        print(line)

    # Persist summary alongside the existing v2 reference
    out_json = OUT_DIR / "_eval_sweep_num_ads.json"
    out_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
