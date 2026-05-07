"""Inside-vs-outside-ad separation on YAMNet features.

For each cached bundle:
  - Load ground-truth ad windows from video_info/<test>.json
  - For each yamnet_* field, compute mean inside vs outside
  - Print the gap (positive = field is higher inside ads, useful as foreignness)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from audio.yamnet_features import ALL_YAMNET_KEYS  # noqa: E402
from schemas.modality import AnalysisBundle  # noqa: E402
from schemas.video_info import VideoInfoDoc, reference_ad_segments_player_shape  # noqa: E402

OUT_DIR = ROOT / "data" / "output"
INFO_DIR = ROOT / "video_info"
TESTS = ["test_001", "test_002", "test_003", "test_004", "test_005", "test_010"]


def _gt_ad_intervals(test: str) -> list[tuple[float, float]]:
    info = VideoInfoDoc.model_validate_json(
        (INFO_DIR / f"{test}.json").read_text(encoding="utf-8")
    )
    return [(s["start"], s["end"]) for s in reference_ad_segments_player_shape(info)]


def _is_inside(t0: float, t1: float, intervals: list[tuple[float, float]]) -> bool:
    mid = 0.5 * (t0 + t1)
    return any(start <= mid < end for start, end in intervals)


def main() -> int:
    print(f"{'test':<10} {'field':<28} {'inside':>9} {'outside':>9} {'gap':>9}")
    print("-" * 72)
    for test in TESTS:
        bundle_path = OUT_DIR / f"{test}_analysis_bundle.json"
        if not bundle_path.is_file():
            continue
        bundle = AnalysisBundle.model_validate_json(bundle_path.read_text(encoding="utf-8"))
        gt = _gt_ad_intervals(test)
        if not gt:
            continue

        inside_vals: dict[str, list[float]] = {k: [] for k in ALL_YAMNET_KEYS}
        outside_vals: dict[str, list[float]] = {k: [] for k in ALL_YAMNET_KEYS}
        for w in bundle.audio_windows:
            extra = w.model_extra or {}
            target = inside_vals if _is_inside(w.t0, w.t1, gt) else outside_vals
            for k in ALL_YAMNET_KEYS:
                v = extra.get(k)
                if v is not None:
                    target[k].append(float(v))

        for key in ALL_YAMNET_KEYS:
            ins = float(np.mean(inside_vals[key])) if inside_vals[key] else 0.0
            out = float(np.mean(outside_vals[key])) if outside_vals[key] else 0.0
            print(f"{test:<10} {key:<28} {ins:>9.4f} {out:>9.4f} {ins - out:>9.4f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
