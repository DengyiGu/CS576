"""Consolidated triage report for a single video.

Use this BEFORE re-tuning when you need to understand what the
pipeline saw on a video. It runs in ~30 s on a cached bundle and
prints a single report covering:

  1. Modality health     - did each signal extractor actually fire?
  2. Ground truth        - GT ad set if available (skipped with --no-gt)
  3. Auto-K decision     - per-K marginal-ratio + interior trace, with the
                           rule that fires at each transition
  4. Picks                - per-pick boundary signals + TP/FP labels
  5. Diagnostic hints    - heuristic suggestions for what to retune

Examples
--------

    # Inspect one of the cached test videos (uses bundle from
    # data/output/, GT from video_info/)
    python scripts/inspect_video.py --test test_005

    # Inspect a brand-new video (will run the full pipeline first;
    # may take ~10 minutes for a 30 min video)
    python scripts/inspect_video.py --video videos_with_ad/demo.mp4 --no-gt

    # Inspect a brand-new video that DOES have a video_info file
    python scripts/inspect_video.py --video videos_with_ad/demo.mp4

The report prints to stdout and is also written to
``data/output/<name>_inspect_report.txt`` for sharing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from fusion.fuse import (  # noqa: E402
    AD_MAX_SEC,
    AD_MIN_SEC,
    EXTEND_KEEP_RATIO,
    EXTEND_SEARCH_SEC,
    MAX_NUM_ADS,
    MIN_INTERIOR_MEAN_FLOOR,
    MIN_MARGINAL_RATIO,
    MIN_NUM_ADS,
    SMOOTH_HALF_WIN,
    _compute_edge_scores,
    _compute_foreignness_scores,
    _count_brand_hits,
    _count_lexicon_hits,
    _find_best_k_ads,
    _loudness_jump_score,
    _min_interior_mean,
    _select_num_ads_auto,
    _smooth,
    fuse_bundle_to_segments,
    load_bundle,
    write_segments_json,
)
from schemas.modality import AnalysisBundle  # noqa: E402
from schemas.video_info import (  # noqa: E402
    load_video_info_doc,
    reference_ad_segments_player_shape,
)

OUT_DIR = ROOT / "data" / "output"
INFO_DIR = ROOT / "video_info"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(values, q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _median(values) -> float:
    return _percentile(values, 50.0)


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(seconds, 60.0)
    return f"{int(m):02d}:{s:05.2f}"


def _midpoint_in_intervals(mid: float, intervals: list[tuple[float, float]]) -> int:
    for i, (s, e) in enumerate(intervals):
        if s <= mid < e:
            return i
    return -1


def _temporal_pr(
    predicted: list[tuple[float, float]],
    reference: list[tuple[float, float]],
    duration: float,
) -> tuple[float, float, float]:
    """1 s-grid temporal precision/recall/F1 (mirrors scripts/evaluate.py)."""
    if duration <= 0:
        return 0.0, 0.0, 0.0
    grid = np.zeros(int(round(duration)) + 1, dtype=np.uint8)
    pred_mask = grid.copy()
    ref_mask = grid.copy()
    for s, e in predicted:
        i0 = max(0, int(round(s)))
        i1 = min(len(pred_mask), int(round(e)))
        pred_mask[i0:i1] = 1
    for s, e in reference:
        i0 = max(0, int(round(s)))
        i1 = min(len(ref_mask), int(round(e)))
        ref_mask[i0:i1] = 1
    tp = float((pred_mask & ref_mask).sum())
    fp = float((pred_mask & (1 - ref_mask)).sum())
    fn = float((ref_mask & (1 - pred_mask)).sum())
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return p, r, f1


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _section_modality_health(bundle: AnalysisBundle) -> list[str]:
    out: list[str] = ["MODALITY HEALTH"]

    # Visual
    vw = bundle.visual.windows
    n = len(vw)
    near = sum(1 for w in vw if w.shot_boundary_near)
    pal = [w.palette_delta for w in vw if w.palette_delta is not None]
    duration = bundle.duration_sec or (vw[-1].t1 if vw else 0.0)
    cuts_per_min = (near / duration * 60.0) if duration > 0 else 0.0
    out.append(
        f"  Visual    n_windows={n:<5}  shot_boundaries={near}/{n} ({100.0*near/max(n,1):.0f}%, "
        f"{cuts_per_min:.1f}/min)  palette_delta median={_median(pal):.2f} p95={_percentile(pal,95):.2f}"
    )

    # Audio
    aw = bundle.audio_windows
    if aw:
        anom = [a.anomaly_score for a in aw if a.anomaly_score is not None]
        rms = [a.rms_db for a in aw if a.rms_db is not None]
        flat = [
            a.model_extra.get("spectral_flatness") if a.model_extra else None
            for a in aw
        ]
        flat = [f for f in flat if f is not None]
        anom_str = f"median={_median(anom):.2f} p95={_percentile(anom,95):.2f}" if anom else "missing"
        rms_str = f"range=[{min(rms):.1f}, {max(rms):.1f}] dB" if rms else "missing"
        flat_str = f"median={_median(flat):.2f}" if flat else "missing"
        out.append(
            f"  Audio     n_windows={len(aw):<5}  anomaly {anom_str}  rms_db {rms_str}  "
            f"spectral_flatness {flat_str}"
        )
    else:
        out.append("  Audio     MISSING -- no audio_windows in bundle")

    # YAMNet
    if aw:
        ym = [
            (a.model_extra or {}).get("yamnet_music_score") for a in aw
        ]
        ys = [
            (a.model_extra or {}).get("yamnet_speech_score") for a in aw
        ]
        ym = [v for v in ym if v is not None]
        ys = [v for v in ys if v is not None]
        if ym or ys:
            ym_str = (
                f"music median={_median(ym):.2f} p95={_percentile(ym,95):.2f}"
                if ym
                else "music missing"
            )
            ys_str = f"speech median={_median(ys):.2f}" if ys else "speech missing"
            out.append(f"  YAMNet    {ym_str}  {ys_str}")
        else:
            out.append("  YAMNet    MISSING -- yamnet_music/speech_score not stamped")

    # Speech
    spans = bundle.speech_spans or []
    speech_time = sum(max(0.0, s.t1 - s.t0) for s in spans)
    coverage = (speech_time / duration * 100.0) if duration > 0 else 0.0
    out.append(
        f"  Speech    n_spans={len(spans):<5}  speech_time={speech_time:.1f}s ({coverage:.0f}% of video)"
    )

    # Brand / lexicon
    brand_safe = brand_ambig = lex_hits = 0
    for s in spans:
        text = s.text or ""
        sf, am, _ = _count_brand_hits(text)
        ncat, _ = _count_lexicon_hits(text)
        brand_safe += sf
        brand_ambig += am
        lex_hits += ncat
    out.append(
        f"  Brand     safe_hits={brand_safe}  ambiguous_hits={brand_ambig}"
    )
    out.append(
        f"  Lexicon   ad-keyword_categories_hit={lex_hits}"
    )
    return out


def _section_ground_truth(
    info_path: Path | None,
) -> tuple[list[str], list[tuple[float, float]]]:
    if info_path is None or not info_path.is_file():
        return ["GROUND TRUTH", "  (none -- running with --no-gt or no video_info file)"], []
    try:
        doc = load_video_info_doc(info_path)
        ref_segs = reference_ad_segments_player_shape(doc)
    except Exception as e:
        return ["GROUND TRUTH", f"  ERROR loading {info_path.name}: {e}"], []
    intervals = [(float(s["start"]), float(s["end"])) for s in ref_segs]
    out = [f"GROUND TRUTH ({info_path.name})"]
    if not intervals:
        out.append("  no ad segments")
        return out, []
    for i, (s, e) in enumerate(intervals):
        out.append(f"  ad #{i+1}  {_fmt_time(s)} - {_fmt_time(e)}  ({e-s:.1f}s)")
    return out, intervals


def _section_auto_k(
    bundle: AnalysisBundle,
) -> tuple[list[str], dict]:
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

    # Run DP for K=1..MAX
    totals: list[float] = [0.0]
    interval_sets: list[list[tuple[int, int]]] = [[]]
    interiors: list[float] = [1.0]
    for k in range(1, MAX_NUM_ADS + 1):
        total_k, intervals_k = _find_best_k_ads(
            smooth_edge, smooth_foreign, windows, k
        )
        if not intervals_k:
            break
        totals.append(total_k)
        interval_sets.append(intervals_k)
        interiors.append(_min_interior_mean(smooth_foreign, intervals_k))

    chosen_k, chosen_intervals = _select_num_ads_auto(
        smooth_edge,
        smooth_foreign,
        windows,
        max_k=MAX_NUM_ADS,
        min_marginal_ratio=MIN_MARGINAL_RATIO,
        min_k=MIN_NUM_ADS,
    )

    out: list[str] = ["AUTO-K DECISION"]
    out.append(
        f"  config: MIN_NUM_ADS={MIN_NUM_ADS} (soft) MAX_NUM_ADS={MAX_NUM_ADS} "
        f"MIN_MARGINAL_RATIO={MIN_MARGINAL_RATIO} MIN_INTERIOR_MEAN_FLOOR={MIN_INTERIOR_MEAN_FLOOR}"
    )
    out.append(
        f"  {'K':>3} {'g_K':>7} {'g_K/g_1':>9} {'min_int':>8}  {'ratio_ok':>9} {'int_ok':>7}  step"
    )
    # Walk the actual algorithm: accept K=1 unconditionally, then for each
    # subsequent K, both rules must pass. Stop at the first failure and
    # remember whether it was an interior collapse (which suppresses the
    # soft floor) or a ratio break (which doesn't).
    first_gain = totals[1] - totals[0] if len(totals) > 1 else 0.0
    rule_k = 1
    interior_collapse_at: int | None = None
    walking = True
    for k in range(1, len(totals)):
        gain_k = totals[k] - totals[k - 1]
        ratio = gain_k / first_gain if first_gain > 0 else 0.0
        ratio_ok = first_gain > 0 and gain_k >= MIN_MARGINAL_RATIO * first_gain
        interior_ok = interiors[k] >= MIN_INTERIOR_MEAN_FLOOR
        if k == 1:
            note = "accept (k=1 always)"
        elif not walking:
            note = "(skipped after stop)"
        elif not interior_ok:
            note = f"STOP -- interior {interiors[k]:.2f} < floor {MIN_INTERIOR_MEAN_FLOOR}"
            interior_collapse_at = k
            walking = False
        elif not ratio_ok:
            note = f"STOP -- ratio {ratio:.2f} < {MIN_MARGINAL_RATIO}"
            walking = False
        else:
            note = "accept"
            rule_k = k
        out.append(
            f"  {k:>3} {gain_k:>7.2f} {ratio:>9.3f} {interiors[k]:>8.3f}  "
            f"{'YES' if ratio_ok else 'NO':>9} {'YES' if interior_ok else 'NO':>7}  {note}"
        )

    # Explain how rule_k -> chosen_k via the soft floor.
    out.append(f"  >> rule walked to K={rule_k}")
    if rule_k < MIN_NUM_ADS:
        if interior_collapse_at is not None:
            out.append(
                f"  >> interior collapsed at K={interior_collapse_at} -- soft floor "
                f"MIN_NUM_ADS={MIN_NUM_ADS} suppressed; respect rule_k"
            )
        else:
            out.append(
                f"  >> no interior collapse below MIN_NUM_ADS={MIN_NUM_ADS}; "
                f"soft floor raises K from {rule_k} to {chosen_k}"
            )
    out.append(f"  >> chosen K = {chosen_k}")
    return out, {
        "chosen_k": chosen_k,
        "chosen_intervals": chosen_intervals,
        "smooth_edge": smooth_edge,
        "smooth_foreign": smooth_foreign,
        "windows": windows,
        "rule_k": rule_k,
        "interior_collapse_at": interior_collapse_at,
        "interiors": interiors,
    }


def _section_picks(
    auto_data: dict,
    bundle: AnalysisBundle,
    gt_intervals: list[tuple[float, float]],
) -> list[str]:
    chosen_k = auto_data["chosen_k"]
    intervals = auto_data["chosen_intervals"]
    smooth_edge = auto_data["smooth_edge"]
    smooth_foreign = auto_data["smooth_foreign"]
    windows = auto_data["windows"]
    if not windows or not intervals:
        return [f"PICKS (K={chosen_k})", "  (no intervals chosen)"]

    norm_edge = smooth_edge / (smooth_edge.max() + 1e-9)
    norm_foreign = smooth_foreign / (smooth_foreign.max() + 1e-9)
    sec = windows[0].t1 - windows[0].t0 or 2.0

    out: list[str] = [f"PICKS (K={chosen_k})"]
    out.append(
        f"  {'kind':<4} {'start':>8} {'end':>8}  {'edge_s':>6} {'edge_e':>6}  "
        f"{'pal_s':>6} {'pal_e':>6}  {'lj_s':>6} {'lj_e':>6}  {'int':>5}"
    )
    for s_idx, e_idx in intervals:
        t0 = s_idx * sec
        t1 = e_idx * sec
        mid = 0.5 * (t0 + t1)
        if gt_intervals:
            tp_idx = _midpoint_in_intervals(mid, gt_intervals)
            kind = f"T{tp_idx}" if tp_idx >= 0 else "FP"
        else:
            kind = "?"
        edge_s = float(norm_edge[s_idx])
        edge_e = float(norm_edge[min(e_idx, len(norm_edge) - 1)])
        # Visual palette at the boundary windows
        pal_s = float(windows[s_idx].palette_delta or 0.0)
        pal_e = float(windows[min(e_idx, len(windows) - 1)].palette_delta or 0.0)
        lj_s = _loudness_jump_score(t0, bundle.audio_windows)
        lj_e = _loudness_jump_score(t1, bundle.audio_windows)
        interior = (
            float(norm_foreign[s_idx:e_idx].mean()) if e_idx > s_idx else 0.0
        )
        out.append(
            f"  {kind:<4} {_fmt_time(t0):>8} {_fmt_time(t1):>8}  "
            f"{edge_s:>6.3f} {edge_e:>6.3f}  "
            f"{pal_s:>6.3f} {pal_e:>6.3f}  "
            f"{lj_s:>6.3f} {lj_e:>6.3f}  "
            f"{interior:>5.3f}"
        )
    return out


def _section_score(
    bundle: AnalysisBundle,
    test_name: str,
    gt_intervals: list[tuple[float, float]],
) -> list[str]:
    """Run the full fuse pipeline (with extension + smoothing) and score."""
    segments = fuse_bundle_to_segments(bundle)
    seg_path = OUT_DIR / f"{test_name}_segments.json"
    write_segments_json(segments, seg_path)
    if not gt_intervals:
        ad_segs = [s for s in segments if s.get("label") == "Advertisement"]
        out = [
            "SCORE",
            f"  no GT -- wrote {len(ad_segs)} ad segments to {seg_path.name}",
        ]
        return out
    pred = [
        (float(s["start"]), float(s["end"]))
        for s in segments
        if s.get("label") == "Advertisement"
    ]
    duration = max((s["end"] for s in segments), default=0.0)
    p, r, f1 = _temporal_pr(pred, gt_intervals, duration)
    return [
        "SCORE",
        f"  predicted ads: {len(pred)}  GT ads: {len(gt_intervals)}",
        f"  precision={p:.3f}  recall={r:.3f}  F1={f1:.3f}",
        f"  wrote {seg_path.name}",
    ]


def _section_hints(
    bundle: AnalysisBundle,
    auto_data: dict,
    gt_intervals: list[tuple[float, float]],
) -> list[str]:
    out: list[str] = ["DIAGNOSTIC HINTS"]
    n = len(bundle.visual.windows)
    duration = bundle.duration_sec or 0.0
    near = sum(1 for w in bundle.visual.windows if w.shot_boundary_near)
    near_pct = 100.0 * near / max(n, 1)

    # Brand list health
    spans = bundle.speech_spans or []
    speech_words = sum(len((s.text or "").split()) for s in spans)
    total_brand_hits = sum(
        _count_brand_hits(s.text or "")[0] + _count_brand_hits(s.text or "")[1]
        for s in spans
    )
    if speech_words > 200 and total_brand_hits == 0:
        out.append(
            f"  - Brand list found 0 hits across {speech_words} transcript "
            f"words. Edit fusion/fuse.py BRAND_NAMES if the demo show has "
            f"specific advertiser brands."
        )

    # Cut-density saturation
    if near_pct > 50:
        out.append(
            f"  - Shot-boundary density is {near_pct:.0f}% (>50%). The "
            f"scene_cut signal saturates here -- consider per-video median "
            f"subtraction or down-weighting if results look noisy."
        )

    # YAMNet missing
    aw = bundle.audio_windows
    has_yamnet = any(
        (a.model_extra or {}).get("yamnet_music_score") is not None
        for a in aw
    )
    if not has_yamnet:
        out.append(
            "  - YAMNet scores NOT stamped. Either "
            "(a) audio/models/yamnet.onnx is missing -- see SETUP.md 3b, or "
            "(b) the bundle predates YAMNet integration -- run "
            "scripts/add_yamnet_to_bundles.py."
        )

    # Auto-K interpretation
    chosen_k = auto_data["chosen_k"]
    interiors = auto_data["interiors"]
    if chosen_k == MAX_NUM_ADS:
        out.append(
            f"  - Auto-K hit MAX_NUM_ADS={MAX_NUM_ADS}; the rule never said "
            f"'stop'. If recall looks low, MAX_NUM_ADS may be a hard ceiling."
        )
    if chosen_k > 0 and chosen_k < len(interiors) - 1:
        next_collapse = interiors[chosen_k + 1] < MIN_INTERIOR_MEAN_FLOOR
        if next_collapse:
            out.append(
                f"  - Interior cleanly collapses K={chosen_k}->K={chosen_k+1} "
                f"({interiors[chosen_k]:.2f} -> {interiors[chosen_k+1]:.2f}); "
                f"K={chosen_k} is well-supported."
            )

    # Speech density vs ad density
    speech_time = sum(max(0.0, s.t1 - s.t0) for s in spans)
    coverage = speech_time / max(duration, 1.0)
    if coverage > 0.70:
        out.append(
            f"  - Speech covers {100*coverage:.0f}% of the video. The "
            f"no-speech foreignness signal is essentially noise here -- "
            f"YAMNet music + audio anomaly carry most of the load."
        )
    elif coverage < 0.20:
        out.append(
            f"  - Speech covers only {100*coverage:.0f}% of the video. "
            f"Brand/lexicon/transcript signals have limited reach; expect "
            f"audio + YAMNet to drive picks."
        )

    # GT vs picks alignment
    if gt_intervals:
        pred_intervals = auto_data["chosen_intervals"]
        sec = bundle.visual.windows[0].t1 - bundle.visual.windows[0].t0 or 2.0
        n_tp = sum(
            1 for s, e in pred_intervals
            if _midpoint_in_intervals(0.5 * (s + e) * sec, gt_intervals) >= 0
        )
        n_fp = len(pred_intervals) - n_tp
        if n_fp > 0:
            out.append(
                f"  - {n_fp}/{len(pred_intervals)} picks are FPs (midpoint "
                f"outside any GT ad). If you see consistently-located FPs "
                f"across multiple K, the foreignness signal needs new "
                f"information, not a different K."
            )
        n_missed = len(gt_intervals) - len({
            _midpoint_in_intervals(0.5 * (s + e) * sec, gt_intervals)
            for s, e in pred_intervals
            if _midpoint_in_intervals(0.5 * (s + e) * sec, gt_intervals) >= 0
        })
        if n_missed > 0:
            out.append(
                f"  - {n_missed}/{len(gt_intervals)} GT ads were missed "
                f"entirely. Likely brand/lexicon/foreignness gaps at those "
                f"timestamps; inspect transcript and audio there."
            )

    if len(out) == 1:
        out.append("  (no obvious issues)")
    return out


# ---------------------------------------------------------------------------
# Pipeline-runner fallback for fresh videos
# ---------------------------------------------------------------------------


def _ensure_bundle(video_path: Path | None, test_name: str) -> AnalysisBundle:
    """Load the cached bundle if present; otherwise run the full pipeline."""
    bundle_path = OUT_DIR / f"{test_name}_analysis_bundle.json"
    if bundle_path.is_file():
        return load_bundle(bundle_path)
    if video_path is None:
        raise FileNotFoundError(
            f"No cached bundle at {bundle_path} and no --video given -- "
            f"can't materialise an analysis bundle."
        )
    print(
        f"[inspect] no cached bundle at {bundle_path.name}; running full "
        f"pipeline on {video_path.name} -- this can take 5-15 min..."
    )
    # Reuse the existing CLI entrypoint so we benefit from any future
    # pipeline changes without duplicating its orchestration here.
    from scripts.run_pipeline import main as run_pipeline_main

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rc = run_pipeline_main([str(video_path), "--out-dir", str(OUT_DIR)])
    if rc != 0:
        raise RuntimeError(f"run_pipeline.py exited with code {rc}")
    if not bundle_path.is_file():
        raise RuntimeError(
            f"Pipeline completed but no bundle was written at {bundle_path}"
        )
    return load_bundle(bundle_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--test", help="Test name (e.g. test_005)")
    src.add_argument("--video", help="Path to a video file (will run pipeline)")
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Skip GT lookup even if a matching video_info file exists.",
    )
    args = parser.parse_args(argv)

    if args.test:
        test_name = args.test
        video_path = None
    else:
        video_path = Path(args.video).resolve()
        test_name = video_path.stem

    bundle = _ensure_bundle(video_path, test_name)

    info_path = None if args.no_gt else INFO_DIR / f"{test_name}.json"
    gt_lines, gt_intervals = _section_ground_truth(info_path)

    health_lines = _section_modality_health(bundle)
    autok_lines, auto_data = _section_auto_k(bundle)
    pick_lines = _section_picks(auto_data, bundle, gt_intervals)
    score_lines = _section_score(bundle, test_name, gt_intervals)
    hint_lines = _section_hints(bundle, auto_data, gt_intervals)

    duration = bundle.duration_sec or (bundle.visual.windows[-1].t1 if bundle.visual.windows else 0.0)
    header = f"=== {test_name}  ({duration:.1f}s, {bundle.visual.native_fps or 0:.1f} fps) ==="
    sections = [header, ""]
    for block in (
        health_lines,
        gt_lines,
        autok_lines,
        pick_lines,
        score_lines,
        hint_lines,
    ):
        sections.extend(block)
        sections.append("")
    report = "\n".join(sections)
    print(report)

    out_path = OUT_DIR / f"{test_name}_inspect_report.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[inspect] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
