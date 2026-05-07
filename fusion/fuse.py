"""
Multimodal ad detection fusion – simplified.

Strategy
--------
Ads create TWO hard cuts (content → ad, ad → content). We detect these
cuts with three complementary edge signals, then search for paired cut
points that bracket a plausible ad interval.  A separate interior signal
confirms that the bracketed region looks/sounds different from the main
content.

Edge signals (at each window boundary)
---------------------------------------
  1. Palette delta    – abrupt colour-palette shift            [0.45 weight]
  2. Audio delta      – abrupt change in music / audio quality [0.45 weight]
  3. Speech interrupt – speech was mid-sentence, then cut off  [0.10 weight]

Interior signals (confirm the bracketed region is ad-like)
-----------------------------------------------------------
  1. Audio anomaly    – audio sounds different from baseline   [0.50 weight]
  2. Visual semantic  – graphics-heavy or text-dense frames    [0.40 weight]
  3. Brand name       – brand word appears in transcript       [0.10 weight]

Palette and audio are the primary drivers; speech is a weak tiebreaker.
No intro / outro detection – every non-ad segment is "Core Content".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from schemas.modality import AnalysisBundle, AudioWindow, SpeechSpan, VisualWindow

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
LABEL_CORE_CONTENT  = "Core Content"
LABEL_ADVERTISEMENT = "Advertisement"

KIND_CONTENT     = "content"
KIND_NON_CONTENT = "non-content"

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
# Ad duration bounds (seconds)
AD_MIN_SEC  = 28.0
AD_MAX_SEC  = 125.0

# Minimum content gap between consecutive ads
GAP_MIN_SEC = 120.0

# First ad cannot start before this many seconds in
FIRST_AD_MIN_START_SEC = 100.0

# Edge signal weights  (must sum to 1.0)
W_EDGE_PALETTE  = 0.45
W_EDGE_AUDIO    = 0.45
W_EDGE_SPEECH   = 0.10

# Interior signal weights  (must sum to 1.0)
W_INT_AUDIO     = 0.50
W_INT_VISUAL    = 0.40
W_INT_BRAND     = 0.10

# How heavily to weight cut-point sharpness vs. interior foreignness in the
# greedy selection score.  Higher → favour sharper cut points.
EDGE_WEIGHT     = 2.5
INTERIOR_WEIGHT = 1.8

# Smoothing half-window (windows, not seconds)
SMOOTH_HALF_WIN = 1

# Minimum fraction of the best interval's score required to accept an ad
AD_SCORE_THRESHOLD_FRAC = 0.60

# ---------------------------------------------------------------------------
# Brand-name loading (only brands – no phrase lists needed)
# ---------------------------------------------------------------------------

def _load_brand_names() -> list[str]:
    signals_file = Path(__file__).parent / "ad_signals.json"
    if not signals_file.is_file():
        return []
    data = json.loads(signals_file.read_text(encoding="utf-8"))
    seen: set[str] = set()
    names: list[str] = []
    for category_brands in data.get("brands", {}).values():
        for name in category_brands:
            lname = name.lower()
            if lname not in seen:
                seen.add(lname)
                names.append(lname)
    return names


_BRAND_NAMES: list[str] = _load_brand_names()

# Sentence-final punctuation pattern used for the interruption detector
_SENTENCE_END_RE = re.compile(r"[.!?]\s*$")

# ---------------------------------------------------------------------------
# Helpers – audio
# ---------------------------------------------------------------------------

def _audio_anomaly(t0: float, t1: float, audio_windows: list[AudioWindow]) -> float:
    """Return the anomaly score of the audio window closest to [t0, t1]."""
    mid = 0.5 * (t0 + t1)
    best_dist = float("inf")
    anomaly = 0.0
    for aw in audio_windows:
        d = abs(0.5 * (aw.t0 + aw.t1) - mid)
        if d < best_dist:
            best_dist = d
            extra = aw.model_extra or {}
            anomaly = float(extra.get("anomaly_score", 0.0))
            # Very low energy (silence / dead air) is also ad-like
            energy = float(extra.get("energy_rms", 1.0))
            if energy < 0.015:
                anomaly = max(anomaly, 0.8)
    return anomaly


def _audio_delta(t_mid: float, audio_windows: list[AudioWindow], half_sec: float = 4.0) -> float:
    """
    Measure abruptness of the audio change around t_mid.
    Compares average anomaly in the window just before vs. just after.
    Returns a value in [0, 1].
    """
    before_vals: list[float] = []
    after_vals:  list[float] = []
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
    return float(abs(np.mean(after_vals) - np.mean(before_vals)))

# ---------------------------------------------------------------------------
# Helpers – speech
# ---------------------------------------------------------------------------

def _speech_interrupted_at(t_boundary: float, speech_spans: list[SpeechSpan],
                            before_sec: float = 5.0, after_sec: float = 5.0) -> float:
    """
    Detect a mid-sentence cut at t_boundary.

    Returns a score in [0, 1]:
      - 1.0  speech was present just before AND just after, and the span
             ending before the boundary does NOT end with sentence-final
             punctuation (it was cut off mid-sentence).
      - 0.5  speech transitions (present on one side only) but no clean
             sentence break detected.
      - 0.0  no nearby speech on either side.
    """
    spans_before = [s for s in speech_spans if s.t1 > t_boundary - before_sec and s.t1 <= t_boundary + 0.5]
    spans_after  = [s for s in speech_spans if s.t0 < t_boundary + after_sec  and s.t0 >= t_boundary - 0.5]

    has_before = len(spans_before) > 0
    has_after  = len(spans_after)  > 0

    if not has_before and not has_after:
        return 0.0

    if has_before and has_after:
        # Both sides have speech – check whether it was cut mid-sentence
        latest_before = max(spans_before, key=lambda s: s.t1)
        text = (latest_before.text or "").strip()
        if text and not _SENTENCE_END_RE.search(text):
            return 1.0   # definite mid-sentence cut
        return 0.3       # speech both sides but clean-ish sentence boundary

    # Speech on only one side → less informative, but still a transition
    return 0.5


def _brand_score(t0: float, t1: float, speech_spans: list[SpeechSpan],
                 context_sec: float = 20.0) -> float:
    """
    Return a score based on how many brand names appear in the transcript
    near [t0, t1].
    """
    if not _BRAND_NAMES:
        return 0.0
    lo, hi = t0 - context_sec, t1 + context_sec
    chunks = [
        s.text.lower()
        for s in speech_spans
        if s.t1 >= lo and s.t0 <= hi and s.text
    ]
    if not chunks:
        return 0.0
    combined = " ".join(chunks)
    hits = sum(1 for b in _BRAND_NAMES if b in combined)
    if hits >= 2:
        return 0.6
    if hits == 1:
        return 0.3
    return 0.0

# ---------------------------------------------------------------------------
# Helpers – visual semantic
# ---------------------------------------------------------------------------

def _visual_semantic_score(w: VisualWindow) -> float:
    """
    Score how ad-like a window looks based on visual structure.
    High text density and graphics-heavy frames are typical of ads.
    """
    score = 0.0
    if w.high_text_density:
        score += 0.35
    if w.visual_hypothesis == "graphics_heavy":
        score += 0.45 * float(w.hypothesis_confidence)
    if w.edge_density > 0.45:
        score += 0.20 * min(1.0, (float(w.edge_density) - 0.45) / 0.35)
    return min(1.0, score)

# ---------------------------------------------------------------------------
# Step 1 – per-window interior foreignness score
def _compute_interior_scores(
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

        audio  = _audio_anomaly(t0, t1, audio_windows)
        visual = _visual_semantic_score(w)
        brand  = _brand_score(t0, t1, speech_spans)

        raw = (W_INT_AUDIO * audio * 1.4 +
               W_INT_VISUAL * visual * 1.2 +
               W_INT_BRAND * brand * 0.8)

        if mid < FIRST_AD_MIN_START_SEC or mid > duration - 40.0:
            raw *= 0.05

        scores[i] = min(1.0, raw)

    return scores

# ---------------------------------------------------------------------------
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

        palette = float(windows[i].palette_delta)
        audio   = _audio_delta(t_boundary, audio_windows, half_sec=5.0)
        speech  = _speech_interrupted_at(t_boundary, speech_spans)

        # Very strong bias toward sharp cuts (PySceneDetect style)
        raw = (W_EDGE_PALETTE * palette * 1.6 +
               W_EDGE_AUDIO * audio * 1.5 +
               W_EDGE_SPEECH * speech * 0.4)

        # Strong suppression near start/end
        if t_boundary < FIRST_AD_MIN_START_SEC - 20 or t_boundary > duration - 40.0:
            raw *= 0.05

        edge[i] = min(1.0, raw)

    return edge

# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

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
# Step 3 – greedy interval selection
def _find_ad_intervals(
    edge_scores: np.ndarray,
    interior_scores: np.ndarray,
    windows: list[VisualWindow],
) -> list[tuple[int, int]]:
    N = len(windows)
    if N == 0:
        return []

    e_max = edge_scores.max() + 1e-9
    f_max = interior_scores.max() + 1e-9
    norm_edge     = np.append(edge_scores / e_max, 0.0)
    norm_interior = interior_scores / f_max
    cum_interior  = np.concatenate([[0.0], np.cumsum(norm_interior)])

    def interval_score(s: int, e: int) -> float:
        interior_mean = (cum_interior[e] - cum_interior[s]) / max(e - s, 1)
        return (EDGE_WEIGHT * (norm_edge[s] + norm_edge[e]) +
                INTERIOR_WEIGHT * interior_mean)

    window_sec = windows[0].t1 - windows[0].t0 if windows else 2.0
    min_w = max(1, int(AD_MIN_SEC / window_sec))
    max_w = max(min_w + 6, int(AD_MAX_SEC / window_sec) + 6)
    gap_w = max(1, int(GAP_MIN_SEC / window_sec))

    first_start_idx = next(
        (i for i, w in enumerate(windows) if w.t0 >= FIRST_AD_MIN_START_SEC), 0
    )

    candidates: list[tuple[float, int, int]] = []
    for s in range(first_start_idx, N):
        for e in range(s + min_w, min(N + 1, s + max_w + 1)):
            dur = windows[e - 1].t1 - windows[s].t0
            if AD_MIN_SEC <= dur <= AD_MAX_SEC:
                score = interval_score(s, e)
                candidates.append((score, s, e))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score = candidates[0][0]
    min_score  = best_score * AD_SCORE_THRESHOLD_FRAC

    blocked = [False] * (N + 1)
    selected: list[tuple[int, int]] = []

    for sc, s, e in candidates:
        if sc < min_score:
            break
        gap_lo = max(0, s - gap_w)
        gap_hi = min(N + 1, e + gap_w)
        if any(blocked[gap_lo:gap_hi]):
            continue
        selected.append((s, e))
        for i in range(gap_lo, gap_hi):
            blocked[i] = True

    selected.sort(key=lambda x: x[0])
    return selected


# ---------------------------------------------------------------------------
# Step 4 – refine ad boundaries to the nearest local edge maximum
# ---------------------------------------------------------------------------

def _refine_boundary(
    idx: int,
    edge_scores: np.ndarray,
    direction: str,          # "start" or "end"
    windows: list[VisualWindow],
    search_sec: float = 12.0,
) -> int:
    N = len(windows)
    window_sec = windows[0].t1 - windows[0].t0 if windows else 2.0
    search_w   = max(1, int(search_sec / window_sec))

    if direction == "start":
        lo = max(0, idx - search_w // 2)
        hi = min(N, idx + search_w)
    else:
        lo = max(0, idx - search_w)
        hi = min(N, idx + search_w // 2 + 1)

    if lo >= hi:
        return idx
    return lo + int(np.argmax(edge_scores[lo:hi]))

# ---------------------------------------------------------------------------
# Segment building
# ---------------------------------------------------------------------------

def _make_segment(label: str, start: float, end: float) -> dict[str, Any]:
    return {
        "start": round(start, 3),
        "end":   round(end,   3),
        "label": label,
        "kind":  KIND_NON_CONTENT if label == LABEL_ADVERTISEMENT else KIND_CONTENT,
    }


def _build_segments(
    ad_intervals: list[tuple[int, int]],
    windows: list[VisualWindow],
    duration: float,
) -> list[dict[str, Any]]:
    N = len(windows)
    is_ad = [False] * N
    for s, e in ad_intervals:
        for i in range(s, min(e, N)):
            is_ad[i] = True

    segments: list[dict[str, Any]] = []
    i = 0
    while i < N:
        if is_ad[i]:
            j = i
            while j < N and is_ad[j]:
                j += 1
            segments.append(_make_segment(LABEL_ADVERTISEMENT, windows[i].t0, windows[j - 1].t1))
            i = j
        else:
            j = i
            while j < N and not is_ad[j]:
                j += 1
            segments.append(_make_segment(LABEL_CORE_CONTENT, windows[i].t0, windows[j - 1].t1))
            i = j

    segments.sort(key=lambda s: s["start"])
    return segments

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fuse_bundle_to_segments(
    bundle: AnalysisBundle,
    *,
    min_segment_seconds: float = 12.0,  # kept for API compat, not currently used
) -> list[dict[str, Any]]:
    if bundle.visual is None or not bundle.visual.windows:
        return []

    windows  = bundle.visual.windows
    duration = bundle.visual.duration_sec or bundle.duration_sec

    # Interior foreignness per window
    raw_interior    = _compute_interior_scores(windows, bundle.audio_windows, bundle.speech_spans, duration)
    smooth_interior = _smooth(raw_interior, SMOOTH_HALF_WIN)

    # Edge sharpness per boundary
    raw_edge    = _compute_edge_scores(windows, bundle.audio_windows, bundle.speech_spans, duration)
    smooth_edge = _smooth(raw_edge, SMOOTH_HALF_WIN)

    # Find ad intervals
    ad_intervals = _find_ad_intervals(smooth_edge, smooth_interior, windows)

    if ad_intervals:
        refined: list[tuple[int, int]] = []
        window_sec = windows[0].t1 - windows[0].t0 if windows else 2.0
        min_w = max(1, int(AD_MIN_SEC / window_sec))
        for s, e in ad_intervals:
            rs = _refine_boundary(s, smooth_edge, "start", windows, search_sec=12.0)
            re = _refine_boundary(e, smooth_edge, "end",   windows, search_sec=12.0)
            if re - rs < min_w:
                re = min(len(windows), rs + min_w)
            refined.append((rs, re))
        return _build_segments(refined, windows, duration)

    # Fallback – no ads detected
    return [_make_segment(LABEL_CORE_CONTENT, windows[0].t0, windows[-1].t1)]


def load_bundle(path: Path) -> AnalysisBundle:
    return AnalysisBundle.model_validate_json(path.read_text(encoding="utf-8"))


def write_segments_json(segments: list[dict[str, Any]], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"schema_version": "1.0", "source": "fusion", "segments": segments}, indent=2),
        encoding="utf-8",
    )
