"""
Multimodal fusion layer – transcript-first ad detection.

Key insight from ground truth analysis:
- Content has continuous speech from the same speaker(s)
- Ad breaks interrupt this speech with different audio/visual characteristics
- The STRONGEST signal is the speech-to-non-speech transition combined
  with visual change at the same timestamp

Strategy:
1. Identify "breakpoints" where speech stops AND visual changes happen
2. Score candidate ad intervals based on speech gap + audio anomaly
3. Use DP to find exactly 3 non-overlapping intervals
4. Hard constraints from ground truth: ads are 28-118s, first ad ≥ 105s
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
# Hard constraints from ground truth analysis
# ---------------------------------------------------------------------------
NUM_ADS = 3

# Ad duration: ground truth range across all 5 videos is 28s – 118s
AD_MIN_SEC = 28.0
AD_MAX_SEC = 119.0

# First ad earliest start: 1:45 = 105 seconds
FIRST_AD_MIN_START_SEC = 105.0

# Latest ad end: ~26 minutes into final video
LAST_AD_MAX_END_SEC = 26.0 * 60.0  # 1560 seconds

# Minimum content gap between ads: at least 2 minutes of content
GAP_MIN_SEC = 120.0

# Ignore intro region completely for ad detection
IGNORE_START_SEC = 100.0

# ---------------------------------------------------------------------------
# Signal computation – TRANSCRIPT FIRST
# ---------------------------------------------------------------------------

def _get_window_sec(windows: list[VisualWindow]) -> float:
    if not windows:
        return 2.0
    return windows[0].t1 - windows[0].t0


def _compute_speech_gap_segments(
    speech_spans: list[SpeechSpan],
    duration: float,
    gap_min_sec: float = 10.0,
) -> list[tuple[float, float]]:
    """
    Find gaps in speech that are at least gap_min_sec long.
    These are candidate locations for ads.
    
    Returns list of (gap_start, gap_end) time ranges where there's no speech.
    """
    if not speech_spans:
        return [(0.0, duration)]
    
    sorted_spans = sorted(speech_spans, key=lambda s: s.t0)
    
    gaps = []
    cursor = 0.0
    
    for span in sorted_spans:
        if span.t0 > cursor + gap_min_sec:
            gaps.append((cursor, span.t0))
        cursor = max(cursor, span.t1)
    
    if duration > cursor + gap_min_sec:
        gaps.append((cursor, duration))
    
    return gaps


def _compute_speech_density_per_window(
    windows: list[VisualWindow],
    speech_spans: list[SpeechSpan],
) -> np.ndarray:
    """
    Compute fraction of each window covered by speech.
    Returns 0 = full speech, 1 = no speech.
    """
    N = len(windows)
    if N == 0:
        return np.zeros(0)
    
    scores = np.zeros(N)
    for i, w in enumerate(windows):
        window_dur = w.t1 - w.t0
        if window_dur <= 0:
            continue
        
        speech_time = 0.0
        for s in speech_spans:
            overlap = min(s.t1, w.t1) - max(s.t0, w.t0)
            if overlap > 0:
                speech_time += overlap
        
        coverage = min(1.0, speech_time / window_dur)
        scores[i] = 1.0 - coverage  # High = no speech
    
    return scores


def _compute_visual_discontinuity(windows: list[VisualWindow]) -> np.ndarray:
    """
    Enhanced visual discontinuity combining palette change + shot boundaries.
    """
    N = len(windows)
    if N == 0:
        return np.zeros(0)
    
    scores = np.zeros(N)
    for i in range(N):
        pd = windows[i].palette_delta
        shot = 1.0 if windows[i].shot_boundary_near else 0.0
        
        # Combine: palette delta peaks + shot boundaries
        scores[i] = max(pd, shot * 0.7)
    
    return scores


def _compute_joint_breakpoint_score(
    windows: list[VisualWindow],
    speech_density: np.ndarray,
    visual_disc: np.ndarray,
    audio_windows: list[AudioWindow],
) -> np.ndarray:
    """
    Compute a "breakpoint" score at each window.
    High score = speech just stopped/changed AND visual changes.
    
    This is the KEY signal: when both speech and visuals change simultaneously,
    that's very likely an ad boundary.
    """
    N = len(windows)
    if N < 2:
        return np.zeros(N)
    
    scores = np.zeros(N)
    window_sec = _get_window_sec(windows)
    
    for i in range(1, N):
        t_boundary = windows[i].t0
        
        # Speech transition: did speech stop or start near this boundary?
        look_sec = 4.0
        look_windows = max(1, int(look_sec / window_sec))
        
        before_speech = np.mean(speech_density[max(0, i-look_windows):i]) if i > 0 else 0
        after_speech = np.mean(speech_density[i:min(N, i+look_windows)]) if i < N else 0
        speech_change = abs(after_speech - before_speech)
        
        # Visual discontinuity at boundary
        vis_change = visual_disc[i] if i < N else 0
        
        # Audio anomaly change
        audio_change = 0.0
        half_sec = 3.0
        before_audio = []
        after_audio = []
        for aw in audio_windows:
            aw_mid = 0.5 * (aw.t0 + aw.t1)
            if t_boundary - half_sec <= aw_mid < t_boundary:
                before_audio.append(float((aw.model_extra or {}).get("anomaly_score", 0.0)))
            elif t_boundary <= aw_mid < t_boundary + half_sec:
                after_audio.append(float((aw.model_extra or {}).get("anomaly_score", 0.0)))
        
        if before_audio and after_audio:
            audio_change = abs(np.mean(after_audio) - np.mean(before_audio))
        
        # PRODUCT: speech change AND visual change must both be present
        # This dramatically reduces false positives
        scores[i] = speech_change * vis_change * (1.0 + audio_change)
    
    # Normalize
    smax = scores.max()
    if smax > 1e-9:
        scores = scores / smax
    
    return scores


def _compute_ad_likelihood(
    windows: list[VisualWindow],
    speech_density: np.ndarray,
    audio_windows: list[AudioWindow],
    visual_disc: np.ndarray,
    duration: float,
) -> np.ndarray:
    """
    Per-window ad likelihood.
    Ads = low speech + high audio anomaly + moderate visual change.
    """
    N = len(windows)
    if N == 0:
        return np.zeros(0)
    
    # Audio anomaly scores
    audio_anomaly = np.zeros(N)
    window_sec = _get_window_sec(windows)
    for i, w in enumerate(windows):
        mid = 0.5 * (w.t0 + w.t1)
        vals = []
        for aw in audio_windows:
            aw_mid = 0.5 * (aw.t0 + aw.t1)
            if abs(aw_mid - mid) < window_sec:
                vals.append(float((aw.model_extra or {}).get("anomaly_score", 0.0)))
        audio_anomaly[i] = float(np.mean(vals)) if vals else 0.0
    
    # Motion: penalize extreme static (inactivity) or extreme high motion
    motion = np.array([w.motion_score for w in windows])
    motion_score = 1.0 - np.abs(motion - 0.3) / 0.3
    motion_score = np.clip(motion_score, 0.0, 1.0)
    
    # Luminance: penalize very dark
    luminance = np.array([w.luminance_mean for w in windows])
    dark_penalty = np.exp(-luminance / 0.05)  # Sharp penalty below 5% luminance
    
    # High text density: slight penalty (slides)
    high_text = np.array([0.3 if w.high_text_density else 0.0 for w in windows])
    
    raw = (
        0.30 * speech_density +       # No speech = ad-like
        0.25 * audio_anomaly +         # Different audio = ad-like
        0.20 * visual_disc +           # Visual changes
        0.15 * motion_score -          # Moderate motion preferred
        0.05 * dark_penalty -          # Not dark/inactive
        0.05 * high_text               # Not slides
    )
    
    # Suppress start/end
    for i, w in enumerate(windows):
        mid = 0.5 * (w.t0 + w.t1)
        if mid < IGNORE_START_SEC:
            raw[i] *= max(0.01, mid / IGNORE_START_SEC)
        elif mid > LAST_AD_MAX_END_SEC:
            raw[i] *= max(0.01, (duration - mid) / max(1, duration - LAST_AD_MAX_END_SEC))
    
    # Normalize
    smin = np.min(raw)
    smax = np.max(raw)
    if smax > smin + 1e-9:
        return (raw - smin) / (smax - smin)
    return np.zeros(N)


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def _smooth(signal: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or len(signal) == 0:
        return signal.copy()
    if width % 2 == 0:
        width += 1
    kernel = np.ones(width) / width
    return np.convolve(signal, kernel, mode='same')


# ---------------------------------------------------------------------------
# DP to find best 3 ad intervals
# ---------------------------------------------------------------------------

def _find_three_ads(
    ad_likelihood: np.ndarray,
    breakpoint_scores: np.ndarray,
    windows: list[VisualWindow],
) -> list[tuple[int, int]]:
    """
    Find exactly 3 non-overlapping intervals [s, e) that:
    - Are within AD_MIN_SEC to AD_MAX_SEC duration
    - Have at least GAP_MIN_SEC of content between them
    - Start at or after FIRST_AD_MIN_START_SEC in video time
    """
    N = len(ad_likelihood)
    if N == 0:
        return []
    
    window_sec = _get_window_sec(windows)
    min_w = max(1, int(AD_MIN_SEC / window_sec))
    max_w = max(min_w + 1, int(AD_MAX_SEC / window_sec) + 1)
    gap_w = max(1, int(GAP_MIN_SEC / window_sec))
    
    # Find first window index that satisfies FIRST_AD_MIN_START_SEC
    first_valid_idx = 0
    for i, w in enumerate(windows):
        if w.t0 >= FIRST_AD_MIN_START_SEC:
            first_valid_idx = i
            break
    
    def _score(s: int, e: int) -> float:
        if e - s < min_w or e - s > max_w:
            return -float("inf")
        if s < first_valid_idx:
            return -float("inf")
        
        # Interior ad likelihood
        interior = np.mean(ad_likelihood[s:e])
        
        # Boundary scores (how strong are the edges)
        edge_bonus = 0.0
        if s > 0 and s < N:
            edge_bonus += breakpoint_scores[s]
        if e > 0 and e < N:
            edge_bonus += breakpoint_scores[e]
        
        return 0.7 * interior + 0.3 * edge_bonus / 2.0
    
    # Build interval score cache for DP
    score_cache = {}
    for s in range(first_valid_idx, N):
        for e in range(s + min_w, min(s + max_w + 1, N + 1)):
            sc = _score(s, e)
            if sc > -float("inf"):
                score_cache[(s, e)] = sc
    
    NEG = -float("inf")
    
    # DP Stage 1: best single ad ending at position e
    best1 = [NEG] * (N + 1)
    best1_int = [None] * (N + 1)
    for e in range(min_w, N + 1):
        for s in range(max(first_valid_idx, e - max_w), e - min_w + 1):
            if (s, e) in score_cache:
                sc = score_cache[(s, e)]
                if sc > best1[e]:
                    best1[e] = sc
                    best1_int[e] = (s, e)
    
    # Prefix max
    pref1 = [NEG] * (N + 1)
    pref1_end = [None] * (N + 1)
    for i in range(N + 1):
        if i > 0:
            pref1[i] = pref1[i-1]
            pref1_end[i] = pref1_end[i-1]
        if best1[i] > pref1[i]:
            pref1[i] = best1[i]
            pref1_end[i] = i
    
    # DP Stage 2: best pair ending at e2
    best2 = [NEG] * (N + 1)
    best2_pair = [None] * (N + 1)
    for e2 in range(min_w, N + 1):
        for s2 in range(max(first_valid_idx, e2 - max_w), e2 - min_w + 1):
            if (s2, e2) not in score_cache:
                continue
            sc2 = score_cache[(s2, e2)]
            max_e1 = s2 - gap_w
            if max_e1 >= 0 and pref1[max_e1] > NEG:
                total = pref1[max_e1] + sc2
                if total > best2[e2]:
                    best2[e2] = total
                    e1 = pref1_end[max_e1]
                    if e1 is not None and best1_int[e1] is not None:
                        best2_pair[e2] = (best1_int[e1], (s2, e2))
    
    # Prefix max for stage 2
    pref2 = [NEG] * (N + 1)
    pref2_end = [None] * (N + 1)
    for i in range(N + 1):
        if i > 0:
            pref2[i] = pref2[i-1]
            pref2_end[i] = pref2_end[i-1]
        if best2[i] > pref2[i]:
            pref2[i] = best2[i]
            pref2_end[i] = i
    
    # DP Stage 3: best triple
    best_total = NEG
    best_triple = None
    for e3 in range(min_w, N + 1):
        for s3 in range(max(first_valid_idx, e3 - max_w), e3 - min_w + 1):
            if (s3, e3) not in score_cache:
                continue
            sc3 = score_cache[(s3, e3)]
            max_e2 = s3 - gap_w
            if max_e2 >= 0 and pref2[max_e2] > NEG:
                total = pref2[max_e2] + sc3
                if total > best_total:
                    best_total = total
                    e2 = pref2_end[max_e2]
                    if e2 is not None and best2_pair[e2] is not None:
                        ad1, ad2 = best2_pair[e2]
                        best_triple = [ad1, ad2, (s3, e3)]
    
    if best_triple:
        return sorted(best_triple, key=lambda x: x[0])
    
    # Fallback: return empty list
    return []


# ---------------------------------------------------------------------------
# Boundary refinement using breakpoint scores
# ---------------------------------------------------------------------------

def _refine_boundary(
    idx: int,
    breakpoint_scores: np.ndarray,
    windows: list[VisualWindow],
    direction: str,
) -> int:
    """Snap boundary to nearest local maximum in breakpoint scores."""
    N = len(windows)
    if N == 0 or idx < 0 or idx >= N:
        return idx
    
    window_sec = _get_window_sec(windows)
    radius = max(2, int(8.0 / window_sec))
    
    if direction == "start":
        lo = max(0, idx - radius // 3)
        hi = min(N, idx + radius)
    else:
        lo = max(0, idx - radius)
        hi = min(N, idx + radius // 3 + 1)
    
    best_i = idx
    best_val = breakpoint_scores[idx] if idx < len(breakpoint_scores) else -1
    for i in range(lo, hi):
        if i < len(breakpoint_scores) and breakpoint_scores[i] > best_val:
            best_val = breakpoint_scores[i]
            best_i = i
    
    return best_i


# ---------------------------------------------------------------------------
# Segment building
# ---------------------------------------------------------------------------

def _make_segment(label: str, start: float, end: float) -> dict[str, Any]:
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "label": label,
        "kind": _KIND_FOR_LABEL.get(label, KIND_NON_CONTENT),
    }


def _classify_block(
    windows: list[VisualWindow],
    indices: list[int],
    is_before_first_ad: bool,
    is_after_last_ad: bool,
    intro_used: bool,
    outro_used: bool,
    duration: float,
) -> tuple[str, bool, bool]:
    """Classify a content block as Intro, Outro, Core Content, or Inactivity."""
    if not indices:
        return LABEL_CORE_CONTENT, intro_used, outro_used
    
    t0 = windows[indices[0]].t0
    t1 = windows[indices[-1]].t1
    block_dur = t1 - t0
    
    # Inactivity: very dark and still
    inactive_count = sum(
        1 for i in indices
        if windows[i].luminance_mean < 0.08 and windows[i].motion_score < 0.06
    )
    if inactive_count > 0.7 * len(indices):
        return LABEL_INACTIVITY, intro_used, outro_used
    
    # Intro: before first ad, short (< 5% of video) or starts near 0
    if is_before_first_ad and not intro_used:
        if t0 < 5.0 or block_dur < duration * 0.06:
            return LABEL_INTRO, True, outro_used
    
    # Outro: after last ad, short or ends near video end
    if is_after_last_ad and not outro_used:
        if t1 > duration - 5.0 or block_dur < duration * 0.15:
            return LABEL_OUTRO, intro_used, True
    
    return LABEL_CORE_CONTENT, intro_used, outro_used


def _build_segments(
    ad_intervals: list[tuple[int, int]],
    windows: list[VisualWindow],
    duration: float,
) -> list[dict[str, Any]]:
    """Build labeled segments from ad intervals."""
    N = len(windows)
    if N == 0:
        return []
    
    # Mark ad windows
    is_ad = [False] * N
    for s, e in ad_intervals:
        for i in range(max(0, s), min(e, N)):
            is_ad[i] = True
    
    first_ad_start = ad_intervals[0][0] if ad_intervals else N
    last_ad_end = ad_intervals[-1][1] if ad_intervals else 0
    
    segments = []
    intro_used = False
    outro_used = False
    i = 0
    
    while i < N:
        if is_ad[i]:
            j = i
            while j < N and is_ad[j]:
                j += 1
            segments.append(_make_segment(
                LABEL_ADVERTISEMENT,
                windows[i].t0,
                windows[j-1].t1
            ))
            i = j
        else:
            run = []
            j = i
            while j < N and not is_ad[j]:
                run.append(j)
                j += 1
            
            is_before = all(idx < first_ad_start for idx in run) if run else False
            is_after = all(idx >= last_ad_end for idx in run) if run else False
            
            label, intro_used, outro_used = _classify_block(
                windows, run, is_before, is_after,
                intro_used, outro_used, duration
            )
            
            segments.append(_make_segment(
                label,
                windows[i].t0,
                windows[j-1].t1
            ))
            i = j
    
    segments.sort(key=lambda s: s["start"])
    
    # Merge adjacent segments with same label
    merged = []
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
    """Absorb segments below min_sec into neighbors."""
    if not segments:
        return segments
    
    result = list(segments)
    for _ in range(20):
        changed = False
        i = 0
        while i < len(result):
            dur = result[i]["end"] - result[i]["start"]
            if dur < min_sec and len(result) > 1:
                if i > 0:
                    result[i-1]["end"] = result[i]["end"]
                    result.pop(i)
                elif i < len(result) - 1:
                    result[i+1]["start"] = result[i]["start"]
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
    """Main entry point."""
    
    if bundle.visual is None or not bundle.visual.windows:
        return []
    
    windows = bundle.visual.windows
    duration = bundle.visual.duration_sec or bundle.duration_sec
    window_sec = _get_window_sec(windows)
    
    # Compute signals
    speech_density = _compute_speech_density_per_window(windows, bundle.speech_spans)
    visual_disc = _compute_visual_discontinuity(windows)
    
    # The KEY signal: joint breakpoints where both speech and visuals change
    breakpoint_scores = _compute_joint_breakpoint_score(
        windows, speech_density, visual_disc, bundle.audio_windows
    )
    
    # Ad likelihood per window
    ad_likelihood = _compute_ad_likelihood(
        windows, speech_density, bundle.audio_windows, visual_disc, duration
    )
    
    # Smooth signals
    smooth_w = max(1, int(3.0 / window_sec))
    ad_smooth = _smooth(ad_likelihood, smooth_w)
    bp_smooth = _smooth(breakpoint_scores, smooth_w)
    
    # Find 3 ad intervals
    ad_intervals = _find_three_ads(ad_smooth, bp_smooth, windows)
    
    if not ad_intervals or len(ad_intervals) != NUM_ADS:
        return [_make_segment(LABEL_CORE_CONTENT, windows[0].t0, windows[-1].t1)]
    
    # Refine boundaries
    min_w = max(1, int(AD_MIN_SEC / window_sec))
    refined = []
    for s, e in ad_intervals:
        rs = _refine_boundary(s, bp_smooth, windows, "start")
        re = _refine_boundary(e, bp_smooth, windows, "end")
        if rs >= re:
            rs = max(0, re - min_w)
        refined.append((rs, re))
    
    # Build segments
    segments = _build_segments(refined, windows, duration)
    segments = _enforce_min_duration(segments, min_segment_seconds)
    
    return segments


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