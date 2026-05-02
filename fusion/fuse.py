"""
Multimodal fusion layer – edge-based ad detection with exactly 3 ads.

Strategy (redesigned)
---------------------
Ads create TWO hard cuts:
  content → ad  : abrupt palette shift, audio change, often speech stops
  ad → content  : abrupt palette shift, audio change, often speech resumes

Rather than maximising a foreignness score over an interval (which inflates
interval sizes), we:

1. Compute a per-window "edge strength" at each boundary between consecutive
   windows: palette_delta already measures this.  We combine it with an audio-
   change signal and a speech-gap signal to form a "cut score".

2. Find all candidate cut points above a threshold — these are potential
   ad start/end boundaries.

3. Use a DP over candidate cut pairs to find exactly 3 (start, end) pairs
   that maximise a combined score:
       score(s,e) = edge_strength(s) + edge_strength(e)
                   + foreignness_bonus(s, e)   # content inside is "foreign"
   subject to:
       AD_MIN_SEC ≤ e−s ≤ AD_MAX_SEC
       GAP_MIN_SEC between consecutive ads
       first ad start ≥ FIRST_AD_MIN_START_SEC

4. Assign content labels: at most one Intro (before first ad) and one Outro
   (after last ad); Inactivity for dark/still blocks; Core Content otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from schemas.modality import AnalysisBundle, AudioWindow, SpeechSpan, VisualWindow

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
LABEL_CORE_CONTENT   = "Core Content"
LABEL_INTRO          = "Intro"
LABEL_OUTRO          = "Outro"
LABEL_ADVERTISEMENT  = "Advertisement"
LABEL_SELF_PROMOTION = "Self-Promotion"
LABEL_RECAP          = "Recap"
LABEL_TRANSITION     = "Transition"
LABEL_INACTIVITY     = "Inactivity"
LABEL_FILLER         = "Filler"

KIND_CONTENT     = "content"
KIND_NON_CONTENT = "non-content"

_KIND_FOR_LABEL: dict[str, str] = {
    LABEL_CORE_CONTENT:   KIND_CONTENT,
    LABEL_INTRO:          KIND_NON_CONTENT,
    LABEL_OUTRO:          KIND_NON_CONTENT,
    LABEL_ADVERTISEMENT:  KIND_NON_CONTENT,
    LABEL_SELF_PROMOTION: KIND_NON_CONTENT,
    LABEL_RECAP:          KIND_NON_CONTENT,
    LABEL_TRANSITION:     KIND_NON_CONTENT,
    LABEL_INACTIVITY:     KIND_NON_CONTENT,
    LABEL_FILLER:         KIND_NON_CONTENT,
}

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
NUM_ADS = 3

# Ad duration bounds (seconds) — ground truth range is ~28–118s
AD_MIN_SEC  = 20.0
AD_MAX_SEC  = 130.0   # tightened from 200 — longest known ad is 118.2s

# Minimum content gap between two consecutive ads
GAP_MIN_SEC = 60.0

# First ad cannot start before this many seconds into the video
FIRST_AD_MIN_START_SEC = 30.0

# Score component weights for the "foreignness" interior bonus
W_PALETTE  = 0.40
W_AUDIO    = 0.35
W_NOSPEECH = 0.25

# How many windows to smooth over when computing edge scores
SMOOTH_HALF_WIN    = 2

# Radius (seconds) for "is there nearby speech?" check
SPEECH_CONTEXT_SEC = 6.0

# Weight of edge signal vs interior foreignness in the final DP score.
# Higher EDGE_WEIGHT pushes the algorithm to favour sharp cut points.
EDGE_WEIGHT       = 2.5
INTERIOR_WEIGHT   = 1.0

# ---------------------------------------------------------------------------
# Ad-signal phrase/brand loading
# ---------------------------------------------------------------------------

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
_SELF_PROMO_PHRASES  = _AD_PHRASES.get("self_promotion", [])
_OUTRO_PHRASES       = _AD_PHRASES.get("outro", [])
_INTRO_PHRASES       = _AD_PHRASES.get("intro", [])
_RECAP_PHRASES       = _AD_PHRASES.get("recap", [])

# ---------------------------------------------------------------------------
# Per-window audio helpers
# ---------------------------------------------------------------------------

def _audio_features(
    t0: float, t1: float, audio_windows: list[AudioWindow]
) -> tuple[float, float]:
    """Return (anomaly_score, energy_rms) for the audio window closest to [t0,t1]."""
    mid = 0.5 * (t0 + t1)
    best_dist = float("inf")
    anomaly = 0.0
    energy  = 1.0
    for aw in audio_windows:
        d = abs(0.5 * (aw.t0 + aw.t1) - mid)
        if d < best_dist:
            best_dist = d
            extra   = aw.model_extra or {}
            anomaly = float(extra.get("anomaly_score", 0.0))
            energy  = float(extra.get("energy_rms", 1.0))
    return anomaly, energy


def _audio_delta(
    t_mid: float,
    audio_windows: list[AudioWindow],
    half_sec: float = 4.0,
) -> float:
    """
    Measure audio change around time t_mid.
    Compare average anomaly in [t_mid-half_sec, t_mid] vs [t_mid, t_mid+half_sec].
    Returns a value in [0, 1].
    """
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

# ---------------------------------------------------------------------------
# Step 1 – per-window foreignness score (interior signal)
# ---------------------------------------------------------------------------

def _compute_foreignness_scores(
    windows: list[VisualWindow],
    audio_windows: list[AudioWindow],
    speech_spans: list[SpeechSpan],
    duration: float,
) -> np.ndarray:
    """
    Per-window foreignness score in [0,1].
    Higher → more likely to be inside an advertisement.
    """
    N = len(windows)
    scores = np.zeros(N, dtype=np.float64)

    for i, w in enumerate(windows):
        t0, t1 = w.t0, w.t1
        mid = 0.5 * (t0 + t1)

        palette_score = float(w.palette_delta)

        anomaly, energy = _audio_features(t0, t1, audio_windows)
        audio_score = float(anomaly)
        if energy < 0.015:
            audio_score = max(audio_score, 0.8)

        cov    = _speech_coverage(t0, t1, speech_spans)
        nearby = _has_nearby_speech(t0, t1, speech_spans, SPEECH_CONTEXT_SEC)
        text_sig = _speech_text_ad_signal(t0, t1, speech_spans)

        nospeech_score = 0.0
        if not nearby:
            nospeech_score = 0.85
        elif cov < 0.05:
            nospeech_score = 0.55

        if text_sig > 0:
            audio_score    = max(audio_score, text_sig)
            nospeech_score = max(nospeech_score, 0.4)

        # Suppress very start/end of video
        if mid < FIRST_AD_MIN_START_SEC or mid > duration - 20.0:
            palette_score  *= 0.1
            audio_score    *= 0.1
            nospeech_score *= 0.1

        scores[i] = (
            W_PALETTE  * palette_score
            + W_AUDIO  * audio_score
            + W_NOSPEECH * nospeech_score
        )

    return scores


# ---------------------------------------------------------------------------
# Step 2 – per-boundary edge score
# ---------------------------------------------------------------------------

def _compute_edge_scores(
    windows: list[VisualWindow],
    audio_windows: list[AudioWindow],
    speech_spans: list[SpeechSpan],
    duration: float,
) -> np.ndarray:
    """
    Edge score at each window boundary i (between window i-1 and window i).
    Returns array of shape (N,) where index i = edge just before window i.
    Index 0 is unused (no edge before first window).

    A high edge score means a hard cut likely occurred at this boundary.
    We use:
      - palette_delta of window i  (already a boundary-level signal)
      - change in audio anomaly across the boundary
      - speech transition: speech just before but not after (or vice versa)
    """
    N = len(windows)
    edge = np.zeros(N, dtype=np.float64)

    for i in range(1, N):
        t_boundary = windows[i].t0

        # Visual: palette_delta is already the delta at this boundary
        vis = float(windows[i].palette_delta)

        # Audio: change in anomaly across boundary
        aud_delta = _audio_delta(t_boundary, audio_windows, half_sec=3.0)

        # Speech transition: was there speech just before but not after?
        had_speech_before = _has_nearby_speech(
            t_boundary - 4.0, t_boundary, speech_spans, 0.5
        )
        has_speech_after = _has_nearby_speech(
            t_boundary, t_boundary + 4.0, speech_spans, 0.5
        )
        speech_transition = 1.0 if (had_speech_before != has_speech_after) else 0.0

        # Suppress edges very close to start/end
        mid = t_boundary
        if mid < FIRST_AD_MIN_START_SEC or mid > duration - 20.0:
            vis             *= 0.1
            aud_delta       *= 0.1
            speech_transition *= 0.1

        edge[i] = 0.50 * vis + 0.35 * aud_delta + 0.15 * speech_transition

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


# ---------------------------------------------------------------------------
# Step 3 – DP over edge pairs to find exactly 3 ad intervals
# ---------------------------------------------------------------------------

def _find_best_three_ads(
    edge_scores: np.ndarray,
    foreign_scores: np.ndarray,
    windows: list[VisualWindow],
) -> list[tuple[int, int]]:
    """
    Find exactly 3 non-overlapping ad intervals [s, e) (window indices).

    Score for interval [s, e):
        edge_scores[s] + edge_scores[e]          ← hard-cut signal at both ends
        + INTERIOR_WEIGHT * mean(foreign[s:e])   ← foreignness of interior

    All terms are normalised so they're on a comparable scale.

    Constraints:
        AD_MIN_SEC ≤ duration ≤ AD_MAX_SEC
        gap ≥ GAP_MIN_SEC between consecutive ads
        first ad start ≥ FIRST_AD_MIN_START_SEC
    """
    N = len(windows)
    if N == 0:
        return []

    # Normalise scores to [0, 1]
    e_max = edge_scores.max()
    f_max = foreign_scores.max()
    norm_edge    = edge_scores    / (e_max    + 1e-9)
    norm_foreign = foreign_scores / (f_max    + 1e-9)

    # FIX: Pad norm_edge with a 0.0 at the end to prevent IndexError when e == N
    norm_edge = np.append(norm_edge, 0.0)

    # Prefix sums for fast interior mean computation
    cum_foreign = np.concatenate([[0.0], np.cumsum(norm_foreign)])

    def interval_score(s: int, e: int) -> float:
        """Score for ad interval [s, e)."""
        interior_mean = (cum_foreign[e] - cum_foreign[s]) / max(e - s, 1)
        return (EDGE_WEIGHT * (norm_edge[s] + norm_edge[e])
                + INTERIOR_WEIGHT * interior_mean)

    window_sec = windows[0].t1 - windows[0].t0 if windows else 2.0
    min_w = max(1, int(AD_MIN_SEC / window_sec))
    max_w = max(min_w + 1, int(AD_MAX_SEC / window_sec) + 1)
    gap_w = max(1, int(GAP_MIN_SEC / window_sec))

    # Find first_start_idx: first window whose t0 ≥ FIRST_AD_MIN_START_SEC
    first_start_idx = 0
    for i, w in enumerate(windows):
        if w.t0 >= FIRST_AD_MIN_START_SEC:
            first_start_idx = i
            break

    NEG_INF = float("-inf")

    # ------------------------------------------------------------------
    # Stage 1: best single ad ending at each index e
    # ------------------------------------------------------------------
    b1s  = np.full(N + 1, NEG_INF, dtype=np.float64)
    b1st = np.full(N + 1, -1,      dtype=np.int32)

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
                b1s[e]  = sc
                b1st[e] = s

    # Prefix max
    p1s  = np.full(N + 1, NEG_INF, dtype=np.float64)
    p1e  = np.full(N + 1, -1,      dtype=np.int32)
    p1st = np.full(N + 1, -1,      dtype=np.int32)
    for i in range(N + 1):
        if i > 0:
            p1s[i]  = p1s[i - 1]
            p1e[i]  = p1e[i - 1]
            p1st[i] = p1st[i - 1]
        if b1s[i] > p1s[i]:
            p1s[i]  = b1s[i]
            p1e[i]  = i
            p1st[i] = b1st[i]

    # ------------------------------------------------------------------
    # Stage 2: best pair
    # ------------------------------------------------------------------
    b2s  = np.full(N + 1, NEG_INF, dtype=np.float64)
    b2s2 = np.full(N + 1, -1,      dtype=np.int32)
    b2e1 = np.full(N + 1, -1,      dtype=np.int32)
    b2s1 = np.full(N + 1, -1,      dtype=np.int32)

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
                b2s[e2]  = total
                b2s2[e2] = s2
                b2e1[e2] = int(p1e[me1])
                b2s1[e2] = int(p1st[me1])

    # Prefix max
    p2s  = np.full(N + 1, NEG_INF, dtype=np.float64)
    p2e2 = np.full(N + 1, -1,      dtype=np.int32)
    p2s2 = np.full(N + 1, -1,      dtype=np.int32)
    p2e1 = np.full(N + 1, -1,      dtype=np.int32)
    p2s1 = np.full(N + 1, -1,      dtype=np.int32)
    for i in range(N + 1):
        if i > 0:
            p2s[i]  = p2s[i - 1]
            p2e2[i] = p2e2[i - 1]
            p2s2[i] = p2s2[i - 1]
            p2e1[i] = p2e1[i - 1]
            p2s1[i] = p2s1[i - 1]
        if b2s[i] > p2s[i]:
            p2s[i]  = b2s[i]
            p2e2[i] = i
            p2s2[i] = b2s2[i]
            p2e1[i] = b2e1[i]
            p2s1[i] = b2s1[i]

    # ------------------------------------------------------------------
    # Stage 3: best triple
    # ------------------------------------------------------------------
    best3_total = NEG_INF
    best3: list[tuple[int, int]] = []

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
            if total3 > best3_total:
                best3_total = total3
                s1 = int(p2s1[me2])
                e1 = int(p2e1[me2])
                s2 = int(p2s2[me2])
                e2 = int(p2e2[me2])
                best3 = [(s1, e1), (s2, e2), (s3, e3)]

    return best3


# ---------------------------------------------------------------------------
# Step 4 – Refine boundaries using local edge maxima
# ---------------------------------------------------------------------------

def _refine_boundary(
    idx: int,
    edge_scores: np.ndarray,
    direction: str,  # "start" or "end"
    windows: list[VisualWindow],
    search_sec: float = 15.0,
) -> int:
    """
    Given a coarse window index, search within ±search_sec for the window
    with the highest edge score and return that index as the refined boundary.

    For "start": the boundary is the cut INTO the ad → look for the highest
                 edge score just at/after the coarse start.
    For "end":   the boundary is the cut OUT of the ad → look for the highest
                 edge score just at/before the coarse end.
    """
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


# ---------------------------------------------------------------------------
# Segment building
# ---------------------------------------------------------------------------

def _make_segment_dict(label: str, start: float, end: float) -> dict[str, Any]:
    return {
        "start": round(start, 3),
        "end":   round(end,   3),
        "label": label,
        "kind":  _KIND_FOR_LABEL.get(label, KIND_NON_CONTENT),
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

    inactive = sum(
        1 for i in run_indices
        if windows[i].luminance_mean < 0.10 and windows[i].motion_score < 0.08
    )
    if inactive > 0.6 * len(run_indices):
        return LABEL_INACTIVITY, intro_used, outro_used

    if is_before_first_ad and not intro_used:
        return LABEL_INTRO, True, outro_used

    if is_after_last_ad and not outro_used:
        return LABEL_OUTRO, intro_used, True

    return LABEL_CORE_CONTENT, intro_used, outro_used


def _build_segments_from_ad_intervals(
    ad_intervals: list[tuple[int, int]],
    windows: list[VisualWindow],
    duration: float,
) -> list[dict[str, Any]]:
    N = len(windows)

    is_ad = [False] * N
    for s, e in ad_intervals:
        for i in range(s, min(e, N)):
            is_ad[i] = True

    first_ad_start = ad_intervals[0][0]
    last_ad_end    = ad_intervals[-1][1]

    segments: list[dict[str, Any]] = []
    intro_used = False
    outro_used = False

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
            run_indices: list[int] = []
            j = i
            while j < N and not is_ad[j]:
                run_indices.append(j)
                j += 1

            is_before = run_indices[-1] < first_ad_start
            is_after  = run_indices[0] >= last_ad_end

            label, intro_used, outro_used = _label_content_run(
                windows, run_indices,
                is_before_first_ad=is_before,
                is_after_last_ad=is_after,
                intro_used=intro_used,
                outro_used=outro_used,
            )
            segments.append(_make_segment_dict(
                label,
                windows[i].t0,
                windows[j - 1].t1,
            ))
            i = j

    segments.sort(key=lambda s: s["start"])
    return segments


# ---------------------------------------------------------------------------
# Legacy helpers (kept for external callers)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fuse_bundle_to_segments(
    bundle: AnalysisBundle,
    *,
    min_segment_seconds: float = 12.0,
    enforce_three_ads: bool = True,
) -> list[dict[str, Any]]:
    if bundle.visual is None or not bundle.visual.windows:
        return []

    windows  = bundle.visual.windows
    duration = bundle.visual.duration_sec or bundle.duration_sec

    # Compute per-window foreignness (interior signal)
    raw_foreign = _compute_foreignness_scores(
        windows, bundle.audio_windows, bundle.speech_spans, duration
    )
    smooth_foreign = _smooth(raw_foreign, SMOOTH_HALF_WIN)

    # Compute per-boundary edge scores (cut signal)
    raw_edge    = _compute_edge_scores(
        windows, bundle.audio_windows, bundle.speech_spans, duration
    )
    smooth_edge = _smooth(raw_edge, SMOOTH_HALF_WIN)

    # Find the 3 best ad intervals using edge + interior signal
    ad_intervals = _find_best_three_ads(smooth_edge, smooth_foreign, windows)

    if ad_intervals and len(ad_intervals) == NUM_ADS:
        # Refine each boundary to the nearest local edge maximum
        refined: list[tuple[int, int]] = []
        for s, e in ad_intervals:
            rs = _refine_boundary(s, smooth_edge, "start", windows, search_sec=12.0)
            re = _refine_boundary(e, smooth_edge, "end",   windows, search_sec=12.0)
            # Ensure minimum duration after refinement
            window_sec = windows[0].t1 - windows[0].t0 if windows else 2.0
            min_w = max(1, int(AD_MIN_SEC / window_sec))
            if re - rs < min_w:
                re = min(len(windows), rs + min_w)
            refined.append((rs, re))

        return _build_segments_from_ad_intervals(refined, windows, duration)

    # Fallback
    return [_make_segment_dict(LABEL_CORE_CONTENT, windows[0].t0, windows[-1].t1)]


def load_bundle(path: Path) -> AnalysisBundle:
    return AnalysisBundle.model_validate_json(path.read_text(encoding="utf-8"))


def write_segments_json(segments: list[dict[str, Any]], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "source":         "fusion",
        "segments":       segments,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")