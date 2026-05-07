"""Sweep EXTEND_KEEP_RATIO and EXTEND_SEARCH_SEC.

Re-fuses (auto-K only) at each (ratio, search) combo and prints the
per-test F1 plus the mean.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fusion.fuse as fuse  # noqa: E402
from fusion.fuse import (  # noqa: E402
    fuse_bundle_to_segments,
    load_bundle,
    write_segments_json,
)
from scripts.evaluate import _evaluate_one  # noqa: E402

OUT_DIR = ROOT / "data" / "output"
INFO_DIR = ROOT / "video_info"
TESTS = ["test_001", "test_002", "test_003", "test_004", "test_005", "test_010"]


def main() -> int:
    combos: list[tuple[float, float]] = [
        (0.0,  0.60),  # baseline: extension off
        (15.0, 0.60),
        (20.0, 0.70),
        (30.0, 0.60),
        (30.0, 0.70),
        (30.0, 0.80),
        (40.0, 0.70),
    ]

    print(f"{'search':>7} {'keep':>5}  " + " ".join(f"{t[-3:]:>6}" for t in TESTS) + "  mean")
    for search_sec, keep in combos:
        fuse.EXTEND_SEARCH_SEC = float(search_sec)
        fuse.EXTEND_KEEP_RATIO = float(keep)
        f1s: list[float] = []
        for test in TESTS:
            bundle = load_bundle(OUT_DIR / f"{test}_analysis_bundle.json")
            segments = fuse_bundle_to_segments(bundle)  # auto-K
            write_segments_json(segments, OUT_DIR / f"{test}_segments.json")
            result = _evaluate_one(test, INFO_DIR, OUT_DIR)
            f1s.append(float(result["f1"]) if result else 0.0)
        mean = sum(f1s) / len(f1s)
        cells = " ".join(f"{f:>6.3f}" for f in f1s)
        print(f"{search_sec:>7.1f} {keep:>5.2f}  {cells}  {mean:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
