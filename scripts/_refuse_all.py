"""Re-run only the fusion stage on every cached analysis bundle.

This is the fast path for iterating on fusion code: visual/audio/OCR/semantic
stay cached, only fuse_bundle_to_segments runs.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from fusion.fuse import fuse_bundle_to_segments, load_bundle, write_segments_json

    bundle_dir = ROOT / "data" / "output"
    bundles = sorted(bundle_dir.glob("test_*_analysis_bundle.json"))
    if not bundles:
        print("No bundles found.", file=sys.stderr)
        return 1

    print(f"{'name':<10} {'pred':>4} {'fused':>7}")
    print("-" * 24)
    for b_path in bundles:
        name = b_path.name.replace("_analysis_bundle.json", "")
        seg_path = bundle_dir / f"{name}_segments.json"
        try:
            t0 = time.monotonic()
            bundle = load_bundle(b_path)
            segments = fuse_bundle_to_segments(bundle, min_segment_seconds=20.0)
            write_segments_json(segments, seg_path)
            elapsed = time.monotonic() - t0
            n_ads = sum(1 for s in segments if s.get("label") == "Advertisement")
            print(f"{name:<10} {n_ads:>4} {elapsed*1000:>5.0f}ms")
        except Exception as e:
            print(f"{name:<10} FAIL: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
