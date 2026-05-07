from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from schemas.modality import AnalysisBundle, AudioWindow, SpeechSpan, VisualWindow

# Label constants
LABEL_CORE_CONTENT = "Core Content"
LABEL_INTRO = "Intro"
LABEL_OUTRO = "Outro"
LABEL_ADVERTISEMENT = "Advertisement"
KIND_CONTENT = "content"
KIND_NON_CONTENT = "non-content"

_KIND_FOR_LABEL: dict[str, str] = {
    LABEL_CORE_CONTENT: KIND_CONTENT,
    LABEL_INTRO: KIND_NON_CONTENT,
    LABEL_OUTRO: KIND_NON_CONTENT,
    LABEL_ADVERTISEMENT: KIND_NON_CONTENT,
}

# Hyper-parameters
AD_MIN_SEC = 20.0
AD_MAX_SEC = 130.0
GAP_MIN_SEC = 60.0
FIRST_AD_MIN_START_SEC = 30.0

W_AUDIO = 1.00
W_VISUAL_SEMANTIC = 0.50

SMOOTH_HALF_WIN = 2
SPEECH_CONTEXT_SEC = 6.0

EDGE_WEIGHT = 2.5
INTERIOR_WEIGHT = 1.0

# Ad-signal phrase/brand loading
def _load_ad_signals() -> tuple[list[str], dict[str, list[str]]]:
    signals_file = Path(__file__).parent / "ad_signals.json"
    if not signals_file.is_file():
        return [], {}
    data = json.loads(signals_file.read_text(encoding="utf-8"))
    brand_names: list[str] = []
    seen: set[str] = set()
    for category_brands in data.get("brands", {}).values():
        for name in category_brands:
            if name not in seen:
                seen.add(name)
                brand_names.append(name)
    return brand_names, data.get("phrases", {})

_AD_BRAND_NAMES, _AD_PHRASES = _load_ad_signals()
_SPONSORSHIP_PHRASES = _AD_PHRASES.get("sponsorship", [])
_SELF_PROMO_PHRASES = _AD_PHRASES.get("self_promotion", [])
_OUTRO_PHRASES = _AD_PHRASES.get("outro", [])
_INTRO_PHRASES = _AD_PHRASES.get("intro", [])
_RECAP_PHRASES = _AD_PHRASES.get("recap", [])

# Per-window audio helpers
def _audio_features(
    t0: float, t1: float, audio_windows: list[AudioWindow]
) -> tuple[float, float]:
    mid = 0.5 * (t0 + t1)
    best_dist = float("inf")
    anomaly = 0.0
    energy = 1.0
    for aw in audio_windows:
        d = abs(0.5 * (aw.t0 + aw.t1) - mid)
        if d < best_dist:
            best_dist = d
            extra = aw.model_extra or {}
            anomaly = float(extra.get("anomaly_score", 0.0))
            energy = float(extra.get("energy_rms", 1.0))
    return anomaly, energy


def _audio_delta(
    t_mid: float,
    audio_windows: list[AudioWindow],
    half_sec: float = 4.0,
) -> float:
    before_vals, after_vals = [], []
    for aw in audio_windows:
        mid = 0.5 * (aw.t0 + aw.t1)
        extra = aw.model_extra or {}
        a = float(extra.get("anomaly_score", 0.0))
        if t_mid - half_sec <= mid < t_mid:
            before_vals.append(a)
        elif t_mid <= mid < t_mid + half_sec:
            after_vals.append(a)
    if not before_vals or not after_vals:
        return 0.0
    return abs(np.mean(after_vals) - np.mean(before_vals))


def _speech_coverage(t0: float, t1: float, speech_spans: list[SpeechSpan]) -> float:
    dur = max(t1 - t0, 1e-6)
    covered = 0.0
    for span in speech_spans:
        ov_s = max(t0, span.t0)
        ov_e = min(t1, span.t1)
        if ov_e > ov_s:
            covered += ov_e - ov_s
    return min(1.0, covered / dur)


def _has_nearby_speech(
    t0: float, t1: float, speech_spans: list[SpeechSpan], context: float
) -> bool:
    lo, hi = t0 - context, t1 + context
    return any(s.t1 >= lo and s.t0 <= hi for s in speech_spans)


def _speech_text_ad_signal(
    t0: float, t1: float, speech_spans: list[SpeechSpan]
) -> float:
    lo, hi = t0 - 20.0, t1 + 20.0
    chunks = [
        s.text.lower() for s in speech_spans
        if s.t1 >= lo and s.t0 <= hi and s.text
    ]
    if not chunks:
        return 0.0
    combined = " ".join(chunks)
    for phrase in _SPONSORSHIP_PHRASES:
        if phrase in combined:
            return 0.9
    brand_hits = sum(1 for b in _AD_BRAND_NAMES if b in combined)
    if brand_hits >= 2:
        return 0.6
    if brand_hits == 1:
        return 0.3
    return 0.0


def _visual_semantic_ad_score(w: VisualWindow) -> float:
    score = 0.0
    if w.high_text_density:
        score += 0.35
    if w.visual_hypothesis == "graphics_heavy":
        score += 0.45 * float(w.hypothesis_confidence)
    if w.edge_density > 0.45:
        score += 0.20 * min(1.0, (float(w.edge_density) - 0.45) / 0.35)
    return min(1.0, score)

# Step 1 – per-window foreignness score
def _compute_foreignness_scores(
    windows: list[VisualWindow],
    audio_windows: list[AudioWindow],
    speech_spans: list[SpeechSpan],
    duration: float,
) -> np.ndarray:
    N = len(windows)
    scores = np.zeros(N, dtype=np.float64)
    for i, w in enumerate(windows):
        t0, t1 = w.t0, w.t1
        mid = 0.5 * (t0 + t1)
        palette_score = float(w.palette_delta)
        visual_semantic = _visual_semantic_ad_score(w)
        anomaly, energy = _audio_features(t0, t1, audio_windows)
        audio_score = float(anomaly)
        if energy < 0.015:
            audio_score = max(audio_score, 0.8)
        cov = _speech_coverage(t0, t1, speech_spans)
        nearby = _has_nearby_speech(t0, t1, speech_spans, SPEECH_CONTEXT_SEC)
        text_sig = _speech_text_ad_signal(t0, t1, speech_spans)
        nospeech_score = 0.0
        if not nearby:
            nospeech_score = 0.85
        elif cov < 0.05:
            nospeech_score = 0.55
        if text_sig > 0:
            audio_score = max(audio_score, text_sig)
            nospeech_score = max(nospeech_score, 0.4)
        if mid < FIRST_AD_MIN_START_SEC or mid > duration - 20.0:
            palette_score *= 0.1
            visual_semantic *= 0.1
            audio_score *= 0.1
            nospeech_score *= 0.1
        scores[i] = (
            W_VISUAL_SEMANTIC * visual_semantic
            + W_AUDIO * audio_score
        )
    return scores

# Step 2 – per-boundary edge score
def _compute_edge_scores(
    windows: list[VisualWindow],
    audio_windows: list[AudioWindow],
    speech_spans: list[SpeechSpan],
    duration: float,
) -> np.ndarray:
    N = len(windows)
    edge = np.zeros(N, dtype=np.float64)
    for i in range(1, N):
        t_boundary = windows[i].t0
        vis = float(windows[i].palette_delta)
        scene_cut = 1.0 if windows[i].shot_boundary_near else 0.0
        if windows[i].shot_boundary_distance_sec is not None:
            scene_cut = max(scene_cut, max(0.0, 1.0 - float(windows[i].shot_boundary_distance_sec) / 2.0))
        aud_delta = _audio_delta(t_boundary, audio_windows, half_sec=3.0)
        had_speech_before = _has_nearby_speech(
            t_boundary - 4.0, t_boundary, speech_spans, 0.5
        )
        has_speech_after = _has_nearby_speech(
            t_boundary, t_boundary + 4.0, speech_spans, 0.5
        )
        speech_transition = 1.0 if (had_speech_before != has_speech_after) else 0.0
        mid = t_boundary
        if mid < FIRST_AD_MIN_START_SEC or mid > duration - 20.0:
            vis *= 0.1
            scene_cut *= 0.1
            aud_delta *= 0.1
            speech_transition *= 0.1
        edge[i] = 0.40 * vis + 0.25 * scene_cut + 0.25 * aud_delta + 0.10 * speech_transition
    return edge


def _smooth(scores: np.ndarray, half_win: int) -> np.ndarray:
    if half_win <= 0:
        return scores.copy()
    N = len(scores)
    out = np.zeros(N, dtype=np.float64)
    for i in range(N):
        lo = max(0, i - half_win)
        hi = min(N, i + half_win + 1)
        out[i] = scores[lo:hi].mean()
    return out

# Step 3 – DP over edge pairs
def _find_best_three_ads(
    edge_scores: np.ndarray,
    foreign_scores: np.ndarray,
    windows: list[VisualWindow],
) -> list[tuple[int, int]]:
    N = len(windows)
    if N == 0:
        return []
    e_max = edge_scores.max()
    f_max = foreign_scores.max()
    norm_edge = edge_scores / (e_max + 1e-9)
    norm_foreign = foreign_scores / (f_max + 1e-9)
    norm_edge = np.append(norm_edge, 0.0)
    cum_foreign = np.concatenate([[0.0], np.cumsum(norm_foreign)])

    def interval_score(s: int, e: int) -> float:
        interior_mean = (cum_foreign[e] - cum_foreign[s]) / max(e - s, 1)
        return (EDGE_WEIGHT * (norm_edge[s] + norm_edge[e])
                + INTERIOR_WEIGHT * interior_mean)

    window_sec = windows[0].t1 - windows[0].t0 if windows else 2.0
    min_w = max(1, int(AD_MIN_SEC / window_sec))
    max_w = max(min_w + 1, int(AD_MAX_SEC / window_sec) + 1)
    gap_w = max(1, int(GAP_MIN_SEC / window_sec))

    first_start_idx = 0
    for i, w in enumerate(windows):
        if w.t0 >= FIRST_AD_MIN_START_SEC:
            first_start_idx = i
            break

    NEG_INF = float("-inf")

    # Best single ad
    b1s = np.full(N + 1, NEG_INF, dtype=np.float64)
    b1st = np.full(N + 1, -1, dtype=np.int32)
    for e in range(min_w, N + 1):
        s_lo = max(first_start_idx, e - max_w)
        s_hi = e - min_w
        if s_lo > s_hi:
            continue
        t_e = windows[e - 1].t1
        for s in range(s_hi, s_lo - 1, -1):
            if windows[s].t0 < FIRST_AD_MIN_START_SEC:
                break
            dur = t_e - windows[s].t0
            if dur < AD_MIN_SEC:
                continue
            if dur > AD_MAX_SEC:
                break
            sc = interval_score(s, e)
            if sc > b1s[e]:
                b1s[e] = sc
                b1st[e] = s

    p1s = np.full(N + 1, NEG_INF, dtype=np.float64)
    p1e = np.full(N + 1, -1, dtype=np.int32)
    p1st = np.full(N + 1, -1, dtype=np.int32)
    for i in range(N + 1):
        if i > 0:
            p1s[i] = p1s[i - 1]
            p1e[i] = p1e[i - 1]
            p1st[i] = p1st[i - 1]
        if b1s[i] > p1s[i]:
            p1s[i] = b1s[i]
            p1e[i] = i
            p1st[i] = b1st[i]

    # Best pair
    b2s = np.full(N + 1, NEG_INF, dtype=np.float64)
    b2s2 = np.full(N + 1, -1, dtype=np.int32)
    b2e1 = np.full(N + 1, -1, dtype=np.int32)
    b2s1 = np.full(N + 1, -1, dtype=np.int32)
    for e2 in range(min_w, N + 1):
        s2_lo = max(first_start_idx, e2 - max_w)
        s2_hi = e2 - min_w
        if s2_lo > s2_hi:
            continue
        t_e2 = windows[e2 - 1].t1
        for s2 in range(s2_hi, s2_lo - 1, -1):
            if windows[s2].t0 < FIRST_AD_MIN_START_SEC:
                break
            dur = t_e2 - windows[s2].t0
            if dur < AD_MIN_SEC:
                continue
            if dur > AD_MAX_SEC:
                break
            sc2 = interval_score(s2, e2)
            me1 = s2 - gap_w
            if me1 < 0 or p1s[me1] <= NEG_INF:
                continue
            total = p1s[me1] + sc2
            if total > b2s[e2]:
                b2s[e2] = total
                b2s2[e2] = s2
                b2e1[e2] = int(p1e[me1])
                b2s1[e2] = int(p1st[me1])

    p2s = np.full(N + 1, NEG_INF, dtype=np.float64)
    p2e2 = np.full(N + 1, -1, dtype=np.int32)
    p2s2 = np.full(N + 1, -1, dtype=np.int32)
    p2e1 = np.full(N + 1, -1, dtype=np.int32)
    p2s1 = np.full(N + 1, -1, dtype=np.int32)
    for i in range(N + 1):
        if i > 0:
            p2s[i] = p2s[i - 1]
            p2e2[i] = p2e2[i - 1]
            p2s2[i] = p2s2[i - 1]
            p2e1[i] = p2e1[i - 1]
            p2s1[i] = p2s1[i - 1]
        if b2s[i] > p2s[i]:
            p2s[i] = b2s[i]
            p2e2[i] = i
            p2s2[i] = b2s2[i]
            p2e1[i] = b2e1[i]
            p2s1[i] = b2s1[i]

    # Best triple
    b3s = np.full(N + 1, NEG_INF, dtype=np.float64)
    b3e3 = np.full(N + 1, -1, dtype=np.int32)
    b3s3 = np.full(N + 1, -1, dtype=np.int32)
    b3e2 = np.full(N + 1, -1, dtype=np.int32)
    b3s2 = np.full(N + 1, -1, dtype=np.int32)
    b3e1 = np.full(N + 1, -1, dtype=np.int32)
    b3s1 = np.full(N + 1, -1, dtype=np.int32)
    for e3 in range(min_w, N + 1):
        s3_lo = max(first_start_idx, e3 - max_w)
        s3_hi = e3 - min_w
        if s3_lo > s3_hi:
            continue
        t_e3 = windows[e3 - 1].t1
        for s3 in range(s3_hi, s3_lo - 1, -1):
            if windows[s3].t0 < FIRST_AD_MIN_START_SEC:
                break
            dur = t_e3 - windows[s3].t0
            if dur < AD_MIN_SEC:
                continue
            if dur > AD_MAX_SEC:
                break
            sc3 = interval_score(s3, e3)
            me2 = s3 - gap_w
            if me2 < 0 or p2s[me2] <= NEG_INF:
                continue
            total3 = p2s[me2] + sc3
            if total3 > b3s[e3]:
                b3s[e3] = total3
                b3e3[e3] = e3
                b3s3[e3] = s3
                b3e2[e3] = int(p2e2[me2])
                b3s2[e3] = int(p2s2[me2])
                b3e1[e3] = int(p2e1[me2])
                b3s1[e3] = int(p2s1[me2])

    p3s = np.full(N + 1, NEG_INF, dtype=np.float64)
    p3e3 = np.full(N + 1, -1, dtype=np.int32)
    p3s3 = np.full(N + 1, -1, dtype=np.int32)
    p3e2 = np.full(N + 1, -1, dtype=np.int32)
    p3s2 = np.full(N + 1, -1, dtype=np.int32)
    p3e1 = np.full(N + 1, -1, dtype=np.int32)
    p3s1 = np.full(N + 1, -1, dtype=np.int32)
    for i in range(N + 1):
        if i > 0:
            p3s[i] = p3s[i - 1]
            p3e3[i] = p3e3[i - 1]
            p3s3[i] = p3s3[i - 1]
            p3e2[i] = p3e2[i - 1]
            p3s2[i] = p3s2[i - 1]
            p3e1[i] = p3e1[i - 1]
            p3s1[i] = p3s1[i - 1]
        if b3s[i] > p3s[i]:
            p3s[i] = b3s[i]
            p3e3[i] = b3e3[i]
            p3s3[i] = b3s3[i]
            p3e2[i] = b3e2[i]
            p3s2[i] = b3s2[i]
            p3e1[i] = b3e1[i]
            p3s1[i] = b3s1[i]

    best3_total = NEG_INF
    best3: list[tuple[int, int]] = []
    for i in range(N + 1):
        if p3s[i] > best3_total:
            best3_total = p3s[i]
            best3 = [
                (int(p3s1[i]), int(p3e1[i])),
                (int(p3s2[i]), int(p3e2[i])),
                (int(p3s3[i]), int(p3e3[i])),
            ]

    # Best quadruple
    best4_total = NEG_INF
    best4: list[tuple[int, int]] = []
    for e4 in range(min_w, N + 1):
        s4_lo = max(first_start_idx, e4 - max_w)
        s4_hi = e4 - min_w
        if s4_lo > s4_hi:
            continue
        t_e4 = windows[e4 - 1].t1
        for s4 in range(s4_hi, s4_lo - 1, -1):
            if windows[s4].t0 < FIRST_AD_MIN_START_SEC:
                break
            dur = t_e4 - windows[s4].t0
            if dur < AD_MIN_SEC:
                continue
            if dur > AD_MAX_SEC:
                break
            sc4 = interval_score(s4, e4)
            me3 = s4 - gap_w
            if me3 < 0 or p3s[me3] <= NEG_INF:
                continue
            total4 = p3s[me3] + sc4
            if total4 > best4_total:
                best4_total = total4
                s1 = int(p3s1[me3])
                e1 = int(p3e1[me3])
                s2 = int(p3s2[me3])
                e2 = int(p3e2[me3])
                s3 = int(p3s3[me3])
                e3 = int(p3e3[me3])
                best4 = [(s1, e1), (s2, e2), (s3, e3), (s4, e4)]

    if best4_total > best3_total:
        return best4
    return best3

# Step 4 – Refine boundaries
def _refine_boundary(
    idx: int,
    edge_scores: np.ndarray,
    direction: str,
    windows: list[VisualWindow],
    search_sec: float = 15.0,
) -> int:
    N = len(windows)
    window_sec = windows[0].t1 - windows[0].t0 if windows else 2.0
    search_w = max(1, int(search_sec / window_sec))
    if direction == "start":
        lo = max(0, idx - search_w // 2)
        hi = min(N, idx + search_w)
    else:
        lo = max(0, idx - search_w)
        hi = min(N, idx + search_w // 2 + 1)
    if lo >= hi:
        return idx
    best_i = lo + int(np.argmax(edge_scores[lo:hi]))
    return best_i

# Segment building
def _make_segment_dict(label: str, start: float, end: float) -> dict[str, Any]:
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "label": label,
        "kind": _KIND_FOR_LABEL.get(label, KIND_NON_CONTENT),
    }


def _label_content_run(
    windows: list[VisualWindow],
    run_indices: list[int],
    is_before_first_ad: bool,
    is_after_last_ad: bool,
    intro_used: bool,
    outro_used: bool,
) -> tuple[str, bool, bool]:
    if not run_indices:
        return LABEL_CORE_CONTENT, intro_used, outro_used
    if is_before_first_ad and not intro_used:
        return LABEL_INTRO, True, outro_used
    if is_after_last_ad and not outro_used:
        return LABEL_OUTRO, intro_used, True
    return LABEL_CORE_CONTENT, intro_used, outro_used


def _build_segments_from_ad_intervals(
    ad_intervals: list[tuple[int, int]],
    windows: list[VisualWindow],
    duration: float,
    intro_end_sec: float | None = None,
    outro_start_sec: float | None = None,
) -> list[dict[str, Any]]:
    N = len(windows)
    is_ad = [False] * N
    for s, e in ad_intervals:
        for i in range(s, min(e, N)):
            is_ad[i] = True
    first_ad_start = ad_intervals[0][0]
    last_ad_end = ad_intervals[-1][1]
    segments: list[dict[str, Any]] = []
    i = 0
    while i < N:
        if is_ad[i]:
            j = i
            while j < N and is_ad[j]:
                j += 1
            segments.append(_make_segment_dict(
                LABEL_ADVERTISEMENT,
                windows[i].t0,
                windows[j - 1].t1,
            ))
            i = j
        else:
            j = i
            while j < N and not is_ad[j]:
                j += 1
            run_start = windows[i].t0
            run_end = windows[j - 1].t1
            is_before = (j - 1) < first_ad_start
            is_after = i >= last_ad_end
            if is_before and intro_end_sec is not None:
                cut = min(intro_end_sec, run_end)
                if cut > run_start:
                    segments.append(_make_segment_dict(LABEL_INTRO, run_start, cut))
                if cut < run_end:
                    segments.append(_make_segment_dict(LABEL_CORE_CONTENT, cut, run_end))
            elif is_after and outro_start_sec is not None:
                cut = max(outro_start_sec, run_start)
                if cut > run_start:
                    segments.append(_make_segment_dict(LABEL_CORE_CONTENT, run_start, cut))
                if cut < run_end:
                    segments.append(_make_segment_dict(LABEL_OUTRO, cut, run_end))
            else:
                segments.append(_make_segment_dict(LABEL_CORE_CONTENT, run_start, run_end))
            i = j
    segments.sort(key=lambda s: s["start"])
    return segments


def _smooth_labels(
    labels: list[str],
    windows: list[VisualWindow],
    min_segment_seconds: float = 12.0,
) -> list[str]:
    if not labels:
        return labels
    result = list(labels)
    def _dur(start: int, lbl: str) -> float:
        total = 0.0
        k = start
        while k < len(result) and result[k] == lbl:
            total += windows[k].t1 - windows[k].t0
            k += 1
        return total
    i = 0
    while i < len(result):
        if _dur(i, result[i]) < min_segment_seconds and i > 0:
            prev = result[i - 1]
            j = i
            while j < len(result) and result[j] == result[i]:
                result[j] = prev
                j += 1
        i += 1
    i = len(result) - 1
    while i >= 0:
        rs = i
        while rs > 0 and result[rs - 1] == result[i]:
            rs -= 1
        run_dur = sum(windows[k].t1 - windows[k].t0 for k in range(rs, i + 1))
        if run_dur < min_segment_seconds and i < len(result) - 1:
            nxt = result[i + 1]
            for k in range(rs, i + 1):
                result[k] = nxt
        i = rs - 1
    return result

# Public API
def fuse_bundle_to_segments(
    bundle: AnalysisBundle,
    *,
    min_segment_seconds: float = 12.0,
    enforce_three_ads: bool = True,
    intro_end_sec: float | None = None,
    outro_start_sec: float | None = None,
) -> list[dict[str, Any]]:
    if bundle.visual is None or not bundle.visual.windows:
        return []
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

    ad_intervals = _find_best_three_ads(smooth_edge, smooth_foreign, windows)

    if ad_intervals:
        refined: list[tuple[int, int]] = []
        for s, e in ad_intervals:
            rs = _refine_boundary(s, smooth_edge, "start", windows, search_sec=12.0)
            re = _refine_boundary(e, smooth_edge, "end", windows, search_sec=12.0)
            window_sec = windows[0].t1 - windows[0].t0 if windows else 2.0
            min_w = max(1, int(AD_MIN_SEC / window_sec))
            if re - rs < min_w:
                re = min(len(windows), rs + min_w)
            refined.append((rs, re))
        return _build_segments_from_ad_intervals(
            refined, windows, duration,
            intro_end_sec=intro_end_sec,
            outro_start_sec=outro_start_sec,
        )

    return [_make_segment_dict(LABEL_CORE_CONTENT, windows[0].t0, windows[-1].t1)]


def load_bundle(path: Path) -> AnalysisBundle:
    return AnalysisBundle.model_validate_json(path.read_text(encoding="utf-8"))


def write_segments_json(segments: list[dict[str, Any]], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "source": "fusion",
        "segments": segments,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")