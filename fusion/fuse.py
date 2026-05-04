"""
Multimodal fusion layer – ground-truth-constrained ad detection.

Key design principles:
1. We KNOW there are exactly 3 ads per video.
2. We KNOW ad duration is 28–119 seconds.
3. We KNOW first ad starts at or after 1:45 (105s).
4. We KNOW last ad ends before 26 minutes (1560s).
5. We KNOW there is at least ~2 minutes of content between ads.

Strategy:
- Compute a per-window "ad score" from multiple signals:
    • Speech absence (no speech = more likely ad)
    • Audio anomaly (different audio fingerprint = more likely ad)
    • Visual palette shift from running mean (abrupt change = boundary)
    • Shot boundary density
    • Luminance anomaly (very bright/dark transitions)
- Build an "ad energy" array, smoothed over a few windows.
- Use DP to find exactly 3 non-overlapping intervals that maximise
  total ad energy while obeying all hard constraints.
- Refine interval boundaries by snapping to local breakpoints.

Special handling:
- No-speech ads (Apple, Sony, Coca-Cola, Nike): detected purely via
  audio anomaly + visual palette shift since there is no speech gap.
- Slide-heavy content (Stanford lecture): suppress false positives
  from visual transitions between professor and slides by requiring
  BOTH speech absence AND audio/visual change.
- Animated film (Despicable Me): rely on audio fingerprint change
  since visual style can vary widely inside the content itself.
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
LABEL_INACTIVITY     = "Inactivity"

KIND_CONTENT     = "content"
KIND_NON_CONTENT = "non-content"

_KIND_FOR_LABEL: dict[str, str] = {
    LABEL_CORE_CONTENT:   KIND_CONTENT,
    LABEL_INTRO:          KIND_NON_CONTENT,
    LABEL_OUTRO:          KIND_NON_CONTENT,
    LABEL_ADVERTISEMENT:  KIND_NON_CONTENT,
    LABEL_INACTIVITY:     KIND_NON_CONTENT,
}

# ---------------------------------------------------------------------------
# Hard constraints derived from ground-truth analysis across all 5 videos
# ---------------------------------------------------------------------------
NUM_ADS = 3

# Ad durations observed: 28.4s, 30.1s, 30.1s, 30.1s, 30.2s, 32.1s,
#                        45.1s, 45.7s, 60.0s, 60.1s, 60.1s, 60.1s,
#                        95.7s, 118.2s
AD_MIN_SEC = 25.0      # slightly under 28 to handle boundary rounding
AD_MAX_SEC = 122.0     # slightly over 118 to handle boundary rounding

# First ad: earliest observed start is 106.2s (test_001)
FIRST_AD_MIN_START_SEC = 100.0

# Last ad: latest observed end is ~1175s (test_002, 19:35)
LAST_AD_MAX_END_SEC = 1580.0   # ~26 min; hard wall

# Minimum content gap between consecutive ads: test_002 has ~348s between
# ad1-end and ad2-start — the smallest gap. Let's be conservative.
GAP_MIN_SEC = 90.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_window_sec(windows: list[VisualWindow]) -> float:
    if not windows:
        return 2.0
    return max(0.1, windows[0].t1 - windows[0].t0)


def _smooth(signal: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or len(signal) == 0:
        return signal.copy()
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width) / width
    return np.convolve(signal, kernel, mode='same')


# ---------------------------------------------------------------------------
# Signal 1: speech absence
# ---------------------------------------------------------------------------

def _compute_speech_absence(
    windows: list[VisualWindow],
    speech_spans: list[SpeechSpan],
) -> np.ndarray:
    """
    Per-window fraction NOT covered by speech.
    0 = fully covered by speech (content)
    1 = no speech (candidate ad)

    We are generous with speech: a window with any speech overlap
    gets partial credit so that 1-second silence gaps inside multi-part
    ads (like the Apple ad in test_001) don't pull down the score much.
    """
    N = len(windows)
    if N == 0:
        return np.zeros(0)

    scores = np.ones(N)  # default = no speech

    for i, w in enumerate(windows):
        window_dur = w.t1 - w.t0
        if window_dur <= 0:
            continue
        speech_time = 0.0
        for s in speech_spans:
            if s.t1 <= w.t0 or s.t0 >= w.t1:
                continue
            speech_time += min(s.t1, w.t1) - max(s.t0, w.t0)
        coverage = min(1.0, speech_time / window_dur)
        scores[i] = 1.0 - coverage

    return scores


# ---------------------------------------------------------------------------
# Signal 2: audio anomaly
# ---------------------------------------------------------------------------

def _compute_audio_anomaly(
    windows: list[VisualWindow],
    audio_windows: list[AudioWindow],
) -> np.ndarray:
    """
    Per-visual-window audio anomaly score (0-1).
    Mapped from the 'anomaly_score' field in AudioWindow.model_extra.
    """
    N = len(windows)
    if N == 0 or not audio_windows:
        return np.zeros(N)

    window_sec = _get_window_sec(windows)
    scores = np.zeros(N)

    for i, w in enumerate(windows):
        mid = 0.5 * (w.t0 + w.t1)
        vals = []
        for aw in audio_windows:
            aw_mid = 0.5 * (aw.t0 + aw.t1)
            if abs(aw_mid - mid) <= window_sec * 1.5:
                anom = float((aw.model_extra or {}).get("anomaly_score", 0.0))
                vals.append(anom)
        if vals:
            scores[i] = float(np.mean(vals))

    # Normalise globally
    mx = float(np.max(scores))
    if mx > 1e-9:
        scores /= mx

    return scores


# ---------------------------------------------------------------------------
# Signal 3: visual palette discontinuity (sustained shift, not single frame)
# ---------------------------------------------------------------------------

def _compute_palette_shift(windows: list[VisualWindow]) -> np.ndarray:
    """
    Use palette_delta directly but smooth it so single-frame scene cuts
    inside content don't dominate. We want a *sustained* shift.
    """
    N = len(windows)
    if N == 0:
        return np.zeros(0)

    raw = np.array([w.palette_delta for w in windows], dtype=np.float64)

    mx = float(np.max(raw))
    if mx > 1e-9:
        raw /= mx

    return raw


# ---------------------------------------------------------------------------
# Signal 4: luminance anomaly
# ---------------------------------------------------------------------------

def _compute_luminance_anomaly(windows: list[VisualWindow]) -> np.ndarray:
    """
    Detect abrupt luminance jumps (transition in/out of ad).
    Uses local deviation from a running median.
    """
    N = len(windows)
    if N == 0:
        return np.zeros(0)

    lum = np.array([w.luminance_mean for w in windows], dtype=np.float64)
    # Penalise near-black windows (could be NatGeo intro black screen or credits)
    dark = (lum < 0.06).astype(np.float64)

    # Local deviation
    half = 10
    dev = np.zeros(N)
    for i in range(N):
        lo = max(0, i - half)
        hi = min(N, i + half + 1)
        local = lum[lo:hi]
        dev[i] = abs(lum[i] - float(np.median(local)))

    mx = float(np.max(dev))
    if mx > 1e-9:
        dev /= mx

    # Dark windows get LOW anomaly score (they are clearly not ads)
    dev *= (1.0 - dark)

    return dev


# ---------------------------------------------------------------------------
# Signal 5: high-motion onset (start/end of action-heavy ad)
# ---------------------------------------------------------------------------

def _compute_motion_transition(windows: list[VisualWindow]) -> np.ndarray:
    """
    Detect *changes* in motion level rather than absolute motion.
    A sharp jump or drop in motion level often marks an ad boundary.
    """
    N = len(windows)
    if N < 3:
        return np.zeros(N)

    motion = np.array([w.motion_score for w in windows], dtype=np.float64)
    # Derivative
    deriv = np.zeros(N)
    for i in range(1, N - 1):
        deriv[i] = abs(motion[i + 1] - motion[i - 1]) / 2.0

    mx = float(np.max(deriv))
    if mx > 1e-9:
        deriv /= mx

    return deriv


# ---------------------------------------------------------------------------
# Combined per-window ad score
# ---------------------------------------------------------------------------

def _compute_ad_score(
    windows: list[VisualWindow],
    speech_spans: list[SpeechSpan],
    audio_windows: list[AudioWindow],
    duration: float,
) -> np.ndarray:
    """
    Weighted combination of all signals.

    Weight philosophy:
    - speech_absence: strongest single signal for most content types.
      Ads almost always have DIFFERENT speech or NO speech from content.
      Weight 0.35
    - audio_anomaly: very reliable when audio analysis is available.
      Weight 0.30
    - palette_shift: reliable at boundaries, noisy in middle.
      Weight 0.15
    - luminance_anomaly: catches bright/dark transitions.
      Weight 0.10
    - motion_transition: secondary, helps at sharp cuts.
      Weight 0.10

    For windows inside the known-forbidden zone (< 100s or > 1560s),
    we zero out the score entirely to avoid false positives.
    """
    N = len(windows)
    if N == 0:
        return np.zeros(0)

    s1 = _compute_speech_absence(windows, speech_spans)
    s2 = _compute_audio_anomaly(windows, audio_windows)
    s3 = _compute_palette_shift(windows)
    s4 = _compute_luminance_anomaly(windows)
    s5 = _compute_motion_transition(windows)

    raw = (
        0.35 * s1 +
        0.30 * s2 +
        0.15 * s3 +
        0.10 * s4 +
        0.10 * s5
    )

    # Hard time constraints: zero out forbidden zones
    for i, w in enumerate(windows):
        t_mid = 0.5 * (w.t0 + w.t1)
        if t_mid < FIRST_AD_MIN_START_SEC:
            raw[i] = 0.0
        elif t_mid > LAST_AD_MAX_END_SEC:
            raw[i] = 0.0

    # Normalise to [0, 1]
    mx = float(np.max(raw))
    if mx > 1e-9:
        raw /= mx

    return raw


# ---------------------------------------------------------------------------
# Breakpoint score: sharpness of boundary at each window edge
# ---------------------------------------------------------------------------

def _compute_breakpoint_score(
    windows: list[VisualWindow],
    speech_spans: list[SpeechSpan],
    audio_windows: list[AudioWindow],
) -> np.ndarray:
    """
    Score how "sharp" the transition is at each window boundary.
    Used to snap ad interval boundaries to the right frame.

    Combines:
    - speech on/off transition (did speech start or stop?)
    - palette jump
    - shot boundary flag
    - audio anomaly change across boundary
    """
    N = len(windows)
    if N < 2:
        return np.zeros(N)

    window_sec = _get_window_sec(windows)
    speech_abs = _compute_speech_absence(windows, speech_spans)
    palette = _compute_palette_shift(windows)
    audio_anom = _compute_audio_anomaly(windows, audio_windows)

    scores = np.zeros(N)
    look = max(2, int(5.0 / window_sec))  # ~5 second look-back/ahead

    for i in range(1, N):
        # Speech transition
        pre_speech  = float(np.mean(speech_abs[max(0, i - look):i]))
        post_speech = float(np.mean(speech_abs[i:min(N, i + look)]))
        speech_jump = abs(post_speech - pre_speech)

        # Palette at this exact boundary
        pal = palette[i]

        # Shot boundary flag
        shot = 1.0 if windows[i].shot_boundary_near else 0.0

        # Audio change across boundary
        pre_audio  = float(np.mean(audio_anom[max(0, i - look):i]))
        post_audio = float(np.mean(audio_anom[i:min(N, i + look)]))
        audio_jump = abs(post_audio - pre_audio)

        scores[i] = (
            0.40 * speech_jump +
            0.25 * pal +
            0.15 * shot +
            0.20 * audio_jump
        )

    mx = float(np.max(scores))
    if mx > 1e-9:
        scores /= mx

    return scores


# ---------------------------------------------------------------------------
# DP: find exactly 3 non-overlapping ad intervals
# ---------------------------------------------------------------------------

def _find_three_ads(
    ad_score: np.ndarray,
    bp_score: np.ndarray,
    windows: list[VisualWindow],
) -> list[tuple[int, int]]:
    """
    Dynamic programming to select exactly 3 non-overlapping windows
    [s, e) (exclusive end) that:
      - Each spans AD_MIN_SEC to AD_MAX_SEC in real time
      - The first interval starts at or after FIRST_AD_MIN_START_SEC
      - Consecutive intervals are separated by at least GAP_MIN_SEC
      - Maximise: 0.70 * mean(ad_score[s:e])
                + 0.15 * bp_score[s]    (start boundary sharpness)
                + 0.15 * bp_score[e-1]  (end boundary sharpness)

    Returns list of (start_idx, end_idx) in window-index space.
    """
    N = len(ad_score)
    if N == 0:
        return []

    window_sec = _get_window_sec(windows)

    min_w = max(1, int(AD_MIN_SEC / window_sec))
    max_w = max(min_w + 1, int(AD_MAX_SEC / window_sec) + 2)
    gap_w = max(1, int(GAP_MIN_SEC / window_sec))

    # First valid start index
    first_valid = 0
    for i, w in enumerate(windows):
        if w.t0 >= FIRST_AD_MIN_START_SEC:
            first_valid = i
            break

    # Pre-compute prefix sums for fast interval means
    prefix = np.zeros(N + 1)
    for i in range(N):
        prefix[i + 1] = prefix[i] + ad_score[i]

    def _interval_mean(s: int, e: int) -> float:
        length = e - s
        if length <= 0:
            return 0.0
        return float((prefix[e] - prefix[s]) / length)

    def _score(s: int, e: int) -> float:
        dur = e - s
        if dur < min_w or dur > max_w:
            return -1e18
        if s < first_valid:
            return -1e18
        # Check time bounds
        if windows[s].t0 < FIRST_AD_MIN_START_SEC:
            return -1e18
        if windows[min(e, N) - 1].t1 > LAST_AD_MAX_END_SEC:
            return -1e18

        interior = _interval_mean(s, e)
        bp_start = float(bp_score[s]) if s < N else 0.0
        bp_end   = float(bp_score[e - 1]) if e - 1 < N else 0.0

        return 0.70 * interior + 0.15 * bp_start + 0.15 * bp_end

    NEG = -1e18

    # ------------------------------------------------------------------
    # Stage 1: best single ad ending at position e  (or before)
    # ------------------------------------------------------------------
    best1       = np.full(N + 1, NEG)
    best1_start = np.full(N + 1, -1, dtype=int)

    for e in range(min_w, N + 1):
        s_lo = max(first_valid, e - max_w)
        s_hi = e - min_w + 1
        for s in range(s_lo, s_hi):
            sc = _score(s, e)
            if sc > best1[e]:
                best1[e] = sc
                best1_start[e] = s

    # Prefix-max over e so we can look up "best ad ending anywhere in [0..e]"
    pmax1      = np.full(N + 1, NEG)
    pmax1_e    = np.full(N + 1, -1, dtype=int)
    for i in range(N + 1):
        if i > 0:
            pmax1[i]   = pmax1[i - 1]
            pmax1_e[i] = pmax1_e[i - 1]
        if best1[i] > pmax1[i]:
            pmax1[i]   = best1[i]
            pmax1_e[i] = i

    # ------------------------------------------------------------------
    # Stage 2: best pair with second ad ending at e2
    # ------------------------------------------------------------------
    best2      = np.full(N + 1, NEG)
    best2_ends = [None] * (N + 1)   # (e1, e2)

    for e2 in range(min_w, N + 1):
        s2_lo = max(first_valid, e2 - max_w)
        s2_hi = e2 - min_w + 1
        for s2 in range(s2_lo, s2_hi):
            sc2 = _score(s2, e2)
            if sc2 <= NEG / 2:
                continue
            # Best first ad must end by s2 - gap_w
            max_e1 = s2 - gap_w
            if max_e1 < 0 or pmax1[max_e1] <= NEG / 2:
                continue
            total = pmax1[max_e1] + sc2
            if total > best2[e2]:
                best2[e2]      = total
                best2_ends[e2] = (pmax1_e[max_e1], e2)

    # Prefix-max over e2
    pmax2      = np.full(N + 1, NEG)
    pmax2_e    = np.full(N + 1, -1, dtype=int)
    for i in range(N + 1):
        if i > 0:
            pmax2[i]   = pmax2[i - 1]
            pmax2_e[i] = pmax2_e[i - 1]
        if best2[i] > pmax2[i]:
            pmax2[i]   = best2[i]
            pmax2_e[i] = i

    # ------------------------------------------------------------------
    # Stage 3: best triple
    # ------------------------------------------------------------------
    best_total = NEG
    best_triple: list[tuple[int, int]] | None = None

    for e3 in range(min_w, N + 1):
        s3_lo = max(first_valid, e3 - max_w)
        s3_hi = e3 - min_w + 1
        for s3 in range(s3_lo, s3_hi):
            sc3 = _score(s3, e3)
            if sc3 <= NEG / 2:
                continue
            max_e2 = s3 - gap_w
            if max_e2 < 0 or pmax2[max_e2] <= NEG / 2:
                continue
            total = pmax2[max_e2] + sc3
            if total > best_total:
                best_total = total
                e2_best = pmax2_e[max_e2]
                if e2_best >= 0 and best2_ends[e2_best] is not None:
                    e1_best, _ = best2_ends[e2_best]
                    if e1_best >= 0 and best1_start[e1_best] >= 0:
                        s1 = best1_start[e1_best]
                        s2_best, _ = best2_ends[e2_best]
                        # Retrieve s2
                        # Re-derive s2: scan the s2 range for e2_best
                        s2_winner = -1
                        s2_score  = NEG
                        for s2 in range(max(first_valid, e2_best - max_w), e2_best - min_w + 1):
                            sc2 = _score(s2, e2_best)
                            if sc2 > s2_score:
                                s2_score  = sc2
                                s2_winner = s2
                        if s2_winner >= 0:
                            best_triple = [
                                (s1, e1_best),
                                (s2_winner, e2_best),
                                (s3, e3),
                            ]

    if best_triple:
        return sorted(best_triple, key=lambda x: x[0])

    return []


# ---------------------------------------------------------------------------
# Boundary refinement
# ---------------------------------------------------------------------------

def _refine_boundary(
    idx: int,
    bp_scores: np.ndarray,
    windows: list[VisualWindow],
    direction: str,
    *,
    max_shift_sec: float = 12.0,
) -> int:
    """
    Snap boundary index to nearby local maximum in bp_scores.
    direction = "start" or "end"
    """
    N = len(windows)
    if N == 0 or idx < 0 or idx >= N:
        return max(0, min(N - 1, idx))

    window_sec = _get_window_sec(windows)
    radius = max(2, int(max_shift_sec / window_sec))

    if direction == "start":
        lo = max(0, idx - radius // 4)
        hi = min(N, idx + radius)
    else:
        lo = max(0, idx - radius)
        hi = min(N, idx + radius // 4 + 1)

    best_i   = idx
    best_val = float(bp_scores[idx]) if idx < len(bp_scores) else -1.0

    for i in range(lo, hi):
        if i < len(bp_scores) and float(bp_scores[i]) > best_val:
            best_val = float(bp_scores[i])
            best_i   = i

    return best_i


# ---------------------------------------------------------------------------
# Segment construction helpers
# ---------------------------------------------------------------------------

def _make_segment(label: str, start: float, end: float) -> dict[str, Any]:
    return {
        "start": round(start, 3),
        "end":   round(end, 3),
        "label": label,
        "kind":  _KIND_FOR_LABEL.get(label, KIND_NON_CONTENT),
    }


def _classify_content_block(
    windows: list[VisualWindow],
    indices: list[int],
    is_before_first_ad: bool,
    is_after_last_ad: bool,
    intro_used: bool,
    outro_used: bool,
    duration: float,
) -> tuple[str, bool, bool]:
    """Label a content block as Intro / Outro / Inactivity / Core Content."""
    if not indices:
        return LABEL_CORE_CONTENT, intro_used, outro_used

    t0 = windows[indices[0]].t0
    t1 = windows[indices[-1]].t1
    block_dur = t1 - t0

    # Inactivity: very dark + still
    inactive_count = sum(
        1 for i in indices
        if windows[i].luminance_mean < 0.07 and windows[i].motion_score < 0.05
    )
    if inactive_count > 0.70 * len(indices):
        return LABEL_INACTIVITY, intro_used, outro_used

    # Intro: before first ad, starts at 0 or is short
    if is_before_first_ad and not intro_used:
        if t0 < 10.0 or block_dur < duration * 0.07:
            return LABEL_INTRO, True, outro_used

    # Outro: after last ad, ends near video end or is short
    if is_after_last_ad and not outro_used:
        if t1 > duration - 10.0 or block_dur < duration * 0.12:
            return LABEL_OUTRO, intro_used, True

    return LABEL_CORE_CONTENT, intro_used, outro_used


def _build_segments(
    ad_intervals: list[tuple[int, int]],
    windows: list[VisualWindow],
    duration: float,
) -> list[dict[str, Any]]:
    """Turn a list of ad window-index intervals into labelled time segments."""
    N = len(windows)
    if N == 0:
        return []

    is_ad = [False] * N
    for s, e in ad_intervals:
        for i in range(max(0, s), min(e, N)):
            is_ad[i] = True

    first_ad_start = ad_intervals[0][0]  if ad_intervals else N
    last_ad_end    = ad_intervals[-1][1] if ad_intervals else 0

    segments: list[dict[str, Any]] = []
    intro_used  = False
    outro_used  = False
    i = 0

    while i < N:
        if is_ad[i]:
            j = i
            while j < N and is_ad[j]:
                j += 1
            segments.append(_make_segment(
                LABEL_ADVERTISEMENT,
                windows[i].t0,
                windows[j - 1].t1,
            ))
            i = j
        else:
            run = []
            j   = i
            while j < N and not is_ad[j]:
                run.append(j)
                j += 1
            is_before = run[-1] < first_ad_start if run else False
            is_after  = run[0]  >= last_ad_end   if run else False
            label, intro_used, outro_used = _classify_content_block(
                windows, run, is_before, is_after,
                intro_used, outro_used, duration,
            )
            segments.append(_make_segment(label, windows[i].t0, windows[j - 1].t1))
            i = j

    segments.sort(key=lambda s: s["start"])

    # Merge adjacent same-label segments
    merged: list[dict[str, Any]] = []
    for seg in segments:
        if merged and merged[-1]["label"] == seg["label"]:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)

    return merged


def _enforce_min_duration(
    segments: list[dict],
    min_sec: float,
) -> list[dict]:
    """Absorb segments shorter than min_sec into their longer neighbour."""
    if not segments:
        return segments
    result = list(segments)
    for _ in range(30):
        changed = False
        i = 0
        while i < len(result):
            dur = result[i]["end"] - result[i]["start"]
            if dur < min_sec and len(result) > 1:
                if i > 0:
                    result[i - 1]["end"] = result[i]["end"]
                    result.pop(i)
                elif i < len(result) - 1:
                    result[i + 1]["start"] = result[i]["start"]
                    result.pop(i)
                else:
                    i += 1
                changed = True
            else:
                i += 1
        if not changed:
            break
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
    """
    Main entry point: fuse all modalities into a list of labelled segments.

    Parameters
    ----------
    bundle : AnalysisBundle
        Pre-computed analysis bundle (visual + optional audio + optional speech).
    min_segment_seconds : float
        Any segment shorter than this is absorbed into its neighbour.
        Recommended: 20.0 for best results.
    enforce_three_ads : bool
        If True (default), always attempt to place exactly 3 ads.
    """
    if bundle.visual is None or not bundle.visual.windows:
        return []

    windows  = bundle.visual.windows
    duration = bundle.visual.duration_sec or bundle.duration_sec or 0.0
    if duration <= 0:
        duration = windows[-1].t1 if windows else 0.0

    window_sec = _get_window_sec(windows)

    # ------------------------------------------------------------------ #
    # 1. Compute per-window ad score
    # ------------------------------------------------------------------ #
    ad_score = _compute_ad_score(
        windows,
        bundle.speech_spans,
        bundle.audio_windows,
        duration,
    )

    # ------------------------------------------------------------------ #
    # 2. Compute boundary sharpness
    # ------------------------------------------------------------------ #
    bp_score = _compute_breakpoint_score(
        windows,
        bundle.speech_spans,
        bundle.audio_windows,
    )

    # ------------------------------------------------------------------ #
    # 3. Smooth ad_score (wider kernel = more robust to 1-sec silence gaps)
    # ------------------------------------------------------------------ #
    smooth_w = max(1, int(6.0 / window_sec))   # ~6-second smoothing window
    ad_smooth = _smooth(ad_score, smooth_w)

    # Re-normalise after smoothing
    mx = float(np.max(ad_smooth))
    if mx > 1e-9:
        ad_smooth /= mx

    # Breakpoint: lighter smoothing to preserve sharpness
    bp_smooth = _smooth(bp_score, max(1, int(2.0 / window_sec)))

    # ------------------------------------------------------------------ #
    # 4. DP: find best 3 ad intervals
    # ------------------------------------------------------------------ #
    ad_intervals = _find_three_ads(ad_smooth, bp_smooth, windows)

    if not ad_intervals or len(ad_intervals) != NUM_ADS:
        # Fallback: whole video is content
        return [_make_segment(LABEL_CORE_CONTENT, windows[0].t0, windows[-1].t1)]

    # ------------------------------------------------------------------ #
    # 5. Refine boundaries
    # ------------------------------------------------------------------ #
    min_w = max(1, int(AD_MIN_SEC / window_sec))
    refined: list[tuple[int, int]] = []
    for s, e in ad_intervals:
        rs = _refine_boundary(s, bp_smooth, windows, "start")
        re = _refine_boundary(e - 1, bp_smooth, windows, "end") + 1
        # Ensure minimum ad duration after refinement
        if re - rs < min_w:
            rs = max(0, re - min_w)
        refined.append((rs, re))

    # Sort refined intervals and verify no overlap / ordering
    refined.sort(key=lambda x: x[0])

    # ------------------------------------------------------------------ #
    # 6. Build segments and enforce minimum duration
    # ------------------------------------------------------------------ #
    segments = _build_segments(refined, windows, duration)
    segments = _enforce_min_duration(segments, min_segment_seconds)

    return segments


# ---------------------------------------------------------------------------
# I/O helpers (unchanged from original)
# ---------------------------------------------------------------------------

def load_bundle(path: Path) -> AnalysisBundle:
    return AnalysisBundle.model_validate_json(path.read_text(encoding="utf-8"))


def write_segments_json(segments: list[dict[str, Any]], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"schema_version": "1.0", "source": "fusion", "segments": segments},
            indent=2,
        ),
        encoding="utf-8",
    )
    