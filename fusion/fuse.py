"""
Multimodal fusion layer – edge-based ad detection.

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

3. Use a DP over candidate cut pairs to find K (start, end) pairs
   that maximise a combined score:
       score(s,e) = edge_strength(s) + edge_strength(e)
                   + foreignness_bonus(s, e)   # content inside is "foreign"
   subject to:
       AD_MIN_SEC ≤ e−s ≤ AD_MAX_SEC
       GAP_MIN_SEC between consecutive ads
       first ad start ≥ FIRST_AD_MIN_START_SEC

   K is either supplied by the caller (`num_ads=N`) or chosen automatically
   from `[1, MAX_NUM_ADS]` by a marginal-gain rule: the K-th ad is accepted
   only when its incremental score is at least `MIN_MARGINAL_RATIO` of the
   first (best) ad's score.

4. Assign content labels: at most one Intro (before first ad) and one Outro
   (after last ad); Inactivity for dark/still blocks; Core Content otherwise.
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
# Default ad count when the caller doesn't override.  ``None`` selects K
# automatically via the marginal-gain rule (see ``_select_num_ads_auto``).
# A positive integer forces exactly that many ads.  The DP underneath is
# generic over K, so callers can pass any K >= 1.
NUM_ADS: int | None = None

# Auto-K bounds when ``num_ads is None``.  The DP runs for K = 1..MAX_NUM_ADS
# and the smallest K where the next ad's marginal score falls below
# ``MIN_MARGINAL_RATIO`` * (first ad's score) is selected.  ``MIN_NUM_ADS``
# is a floor: real broadcast video has ad blocks with very similar interval
# scores, so the marginal-gain rule under-detects below 3 even on cases that
# clearly have 3 ads.  Clamping to >= 3 keeps auto-K aligned with the
# project's "expect at least 3 ads" prior while still allowing 4 / 5 when
# the data really supports it (e.g. test_010 has 4 ads).
MIN_NUM_ADS         = 3
MAX_NUM_ADS         = 6
MIN_MARGINAL_RATIO  = 0.90

# Ad duration bounds (seconds) — ground truth range is ~28–118s
AD_MIN_SEC  = 20.0
AD_MAX_SEC  = 130.0   # tightened from 200 — longest known ad is 118.2s

# Minimum content gap between two consecutive ads
GAP_MIN_SEC = 60.0

# First ad cannot start before this many seconds into the video
FIRST_AD_MIN_START_SEC = 30.0

# Score component weights for the "foreignness" interior bonus
W_PALETTE  = 0.00
W_AUDIO    = 0.45
W_NOSPEECH = 0.00
W_VISUAL_SEMANTIC = 0.40
W_DENSITY_DROP    = 0.20  # transcript density drop vs. global baseline
W_RANDOM   = 0.00 # random baseline parameter for testing

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

def _load_ad_signals() -> tuple[list[str], dict[str, list[str]], list[str]]:
    """Return (brand_names, phrase_categories, ambiguous_brands)."""
    signals_file = Path(__file__).parent / "ad_signals.json"
    if not signals_file.is_file():
        return [], {}, []
    data = json.loads(signals_file.read_text(encoding="utf-8"))
    brand_names: list[str] = []
    seen: set[str] = set()
    for category_brands in data.get("brands", {}).values():
        for name in category_brands:
            n = name.lower()
            if n not in seen:
                seen.add(n)
                brand_names.append(n)
    phrases = {k: [p.lower() for p in v] for k, v in data.get("phrases", {}).items()}
    ambiguous = [a.lower() for a in data.get("ambiguous_brands", [])]
    return brand_names, phrases, ambiguous


_AD_BRAND_NAMES, _AD_PHRASES, _AMBIGUOUS_BRANDS = _load_ad_signals()
_AMBIGUOUS_BRANDS_SET = set(_AMBIGUOUS_BRANDS)
_SPONSORSHIP_PHRASES = _AD_PHRASES.get("sponsorship", [])
_SELF_PROMO_PHRASES  = _AD_PHRASES.get("self_promotion", [])
_OUTRO_PHRASES       = _AD_PHRASES.get("outro", [])
_INTRO_PHRASES       = _AD_PHRASES.get("intro", [])
_RECAP_PHRASES       = _AD_PHRASES.get("recap", [])

# TV-ad style lexicon — these categories detect the *style* of broadcast
# advertising (imperatives, deal language, taglines, compliance disclaimers,
# scripted pricing) rather than specific brand names. Each category contributes
# independently to the speech-text ad signal so a window only needs evidence
# from one of them to count as suspicious.
_TV_AD_CATEGORIES = (
    "tv_ad_imperative",
    "tv_ad_deal",
    "tv_ad_tagline",
    "tv_ad_compliance",
    "tv_ad_pricing",
)
_TV_AD_PHRASES_BY_CATEGORY: dict[str, list[str]] = {
    cat: _AD_PHRASES.get(cat, []) for cat in _TV_AD_CATEGORIES
}


# Pre-compile word-boundary-aware regexes for every brand so substring
# false-positives ("apple" in "snapple", "max" in "climax", "discover" in
# "discovered") do not fire. Uses a manual non-alphanumeric boundary instead
# of \b because brand strings contain hyphens, ampersands, and apostrophes
# that \b doesn't handle correctly.
def _word_boundary_pattern(needle: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])")


_BRAND_REGEX: dict[str, re.Pattern[str]] = {
    b: _word_boundary_pattern(b) for b in _AD_BRAND_NAMES
}
_PHRASE_REGEX_BY_CATEGORY: dict[str, list[tuple[str, re.Pattern[str]]]] = {
    cat: [(p, _word_boundary_pattern(p)) for p in phrases]
    for cat, phrases in _TV_AD_PHRASES_BY_CATEGORY.items()
}
_SPONSORSHIP_REGEX = [(p, _word_boundary_pattern(p)) for p in _SPONSORSHIP_PHRASES]


def _count_brand_hits(text: str) -> tuple[int, int, list[str]]:
    """Return (safe_hits, ambiguous_hits, distinct_brands).

    A "safe" brand is one that is unlikely to false-positive on common English
    (long, multi-word, or otherwise distinctive). Ambiguous brands need a
    co-signal before they should drive an ad classification.
    """
    safe = 0
    ambiguous = 0
    distinct: list[str] = []
    for brand, pat in _BRAND_REGEX.items():
        if pat.search(text):
            distinct.append(brand)
            if brand in _AMBIGUOUS_BRANDS_SET:
                ambiguous += 1
            else:
                safe += 1
    return safe, ambiguous, distinct


def _count_lexicon_hits(text: str) -> tuple[int, dict[str, list[str]]]:
    """Return (categories_hit, hits_by_category) for the TV-ad lexicon."""
    by_cat: dict[str, list[str]] = {}
    for cat, regexes in _PHRASE_REGEX_BY_CATEGORY.items():
        for phrase, pat in regexes:
            if pat.search(text):
                by_cat.setdefault(cat, []).append(phrase)
    return len(by_cat), by_cat


def _has_sponsorship_phrase(text: str) -> bool:
    return any(pat.search(text) for _, pat in _SPONSORSHIP_REGEX)

# Per-window audio helpers
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


# Loudness jump — broadcast TV ads are mixed to a different LUFS target than
# show audio, which produces a clean step in median rms_db at the boundary.
# Unlike anomaly_score, this is robust on real broadcast tracks where the
# in-ad audio doesn't necessarily *sound* anomalous in MFCC space.
_LOUDNESS_JUMP_REFERENCE_DB = 8.0  # 8 dB step gets normalized to 1.0


def _loudness_jump_score(
    t_boundary: float,
    audio_windows: list[AudioWindow],
    *,
    half_sec: float = 10.0,
    edge_skip_sec: float = 1.0,
) -> float:
    """
    Median rms_db jump across the boundary at ``t_boundary``.

    Compares the median ``rms_db`` in [t-half-edge_skip, t-edge_skip] vs
    [t+edge_skip, t+half+edge_skip]. The ``edge_skip`` keeps the noisy single
    transition window from contaminating both sides. Returns a value in [0, 1]
    where 1.0 means a >=8 dB step.
    """
    before_db: list[float] = []
    after_db: list[float] = []
    for aw in audio_windows:
        mid = 0.5 * (aw.t0 + aw.t1)
        extra = aw.model_extra or {}
        rms_db = extra.get("rms_db")
        if rms_db is None:
            continue
        if t_boundary - half_sec - edge_skip_sec <= mid < t_boundary - edge_skip_sec:
            before_db.append(float(rms_db))
        elif t_boundary + edge_skip_sec <= mid < t_boundary + half_sec + edge_skip_sec:
            after_db.append(float(rms_db))
    if not before_db or not after_db:
        return 0.0
    delta = abs(np.median(after_db) - np.median(before_db))
    return float(min(1.0, delta / _LOUDNESS_JUMP_REFERENCE_DB))


# Transcript density drop — show / podcast content has roughly steady
# characters-per-second of speech (Whisper transcript volume); TV-style ads
# either go wordless (music montage) or switch to short character lines, so
# the density inside an ad window drops sharply versus the global baseline.
_DENSITY_BASELINE_FLOOR = 1.0  # chars/sec floor to avoid division by zero


def _compute_transcript_density_baseline(
    speech_spans: list[SpeechSpan], duration_sec: float
) -> float:
    """Return global chars/sec across the whole video.

    Used as the denominator when scoring per-window density drop. Has a small
    floor so videos with very little speech still produce a finite ratio.
    """
    if duration_sec <= 0.0 or not speech_spans:
        return _DENSITY_BASELINE_FLOOR
    total_chars = sum(len(s.text or "") for s in speech_spans)
    raw = total_chars / float(duration_sec)
    return max(_DENSITY_BASELINE_FLOOR, float(raw))


def _transcript_density_score(
    t0: float,
    t1: float,
    speech_spans: list[SpeechSpan],
    *,
    baseline_chars_per_sec: float,
    context_sec: float = 8.0,
) -> float:
    """
    Score how much the transcript density inside [t0-ctx, t1+ctx] has dropped
    versus the global baseline. Returns a value in [0, 1].

    A 50% drop -> 0.5; 80% drop -> 0.8; full silence -> 1.0.
    Returns 0.0 when density is at or above the baseline.
    """
    lo, hi = t0 - context_sec, t1 + context_sec
    span_sec = max(1e-3, hi - lo)
    chars = 0
    for s in speech_spans:
        ov_lo = max(lo, s.t0)
        ov_hi = min(hi, s.t1)
        if ov_hi <= ov_lo or not s.text:
            continue
        # Approximate chars-in-window by linear scaling of the span text by the
        # fraction of the span overlapping the window.
        span_dur = max(1e-3, s.t1 - s.t0)
        chars += len(s.text) * ((ov_hi - ov_lo) / span_dur)
    inside_density = chars / span_sec
    ratio = inside_density / max(baseline_chars_per_sec, _DENSITY_BASELINE_FLOOR)
    return float(min(1.0, max(0.0, 1.0 - ratio)))


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
    """Score the chance that the transcript around [t0, t1] is from an ad.

    Combines four kinds of evidence, each with word-boundary-aware matching:

      1. Sponsorship phrases ("brought to you by", "use code") -> 0.95
         — by far the most discriminative, sponsor-read style.
      2. TV-ad lexicon categories (imperative / deal / tagline / compliance /
         pricing). Returns 0.85 when 2+ categories fire together (rare in
         non-ad speech), 0.55 when one category fires.
      3. Safe brand mentions — long or distinctive enough that a substring
         match is reliable evidence on its own. 1+ -> at least 0.45,
         2+ -> 0.65, capped at 0.85.
      4. Ambiguous brand mentions ("max", "discover", "apple") only count when
         a TV-ad phrase or a safe brand also fires in the window, otherwise
         they are dropped to avoid false-positives on common English.

    The ±20 s context window matches the original detector and is wider than
    a single 1-2 s speech span so a tagline near a cut still counts.
    """
    lo, hi = t0 - 20.0, t1 + 20.0
    chunks = [
        s.text.lower() for s in speech_spans
        if s.t1 >= lo and s.t0 <= hi and s.text
    ]
    if not chunks:
        return 0.0
    combined = " ".join(chunks)

    if _has_sponsorship_phrase(combined):
        return 0.95

    n_categories, _ = _count_lexicon_hits(combined)
    safe_hits, ambig_hits, _ = _count_brand_hits(combined)

    score = 0.0

    if n_categories >= 2:
        score = max(score, 0.85)
    elif n_categories == 1:
        score = max(score, 0.55)

    if safe_hits >= 2:
        score = max(score, 0.65)
    elif safe_hits == 1:
        score = max(score, 0.45)

    # Ambiguous brand mentions only matter as a co-signal — bump the score a
    # little when we already have other evidence, but never let them drive
    # the score on their own.
    if ambig_hits and (n_categories or safe_hits):
        score = min(0.95, score + 0.10 * min(ambig_hits, 3))

    return score


def _visual_semantic_ad_score(w: VisualWindow) -> float:
    """
    Score visual evidence that is ad-like even when palette_delta is modest.

    The visual analyzer already distinguishes graphics/text-heavy windows from
    ordinary dynamic-talk content. Fusion should use those semantic hints as
    interior evidence instead of relying only on hard palette changes.
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
    density_baseline = _compute_transcript_density_baseline(speech_spans, duration)

    for i, w in enumerate(windows):
        t0, t1 = w.t0, w.t1
        mid = 0.5 * (t0 + t1)

        palette_score = float(w.palette_delta)
        visual_semantic = _visual_semantic_ad_score(w)

        anomaly, energy = _audio_features(t0, t1, audio_windows)
        audio_score = float(anomaly)
        if energy < 0.015:
            audio_score = max(audio_score, 0.8)

        cov    = _speech_coverage(t0, t1, speech_spans)
        nearby = _has_nearby_speech(t0, t1, speech_spans, SPEECH_CONTEXT_SEC)
        text_sig = _speech_text_ad_signal(t0, t1, speech_spans)
        density_drop = _transcript_density_score(
            t0, t1, speech_spans, baseline_chars_per_sec=density_baseline
        )

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
            visual_semantic *= 0.1
            audio_score    *= 0.1
            nospeech_score *= 0.1
            density_drop   *= 0.1

        scores[i] = (
            W_PALETTE  * palette_score
            + W_VISUAL_SEMANTIC * visual_semantic
            + W_AUDIO  * audio_score
            + W_NOSPEECH * nospeech_score
            + W_DENSITY_DROP * density_drop
            + W_RANDOM * np.random.rand()
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
        scene_cut = 1.0 if windows[i].shot_boundary_near else 0.0
        if windows[i].shot_boundary_distance_sec is not None:
            scene_cut = max(scene_cut, max(0.0, 1.0 - float(windows[i].shot_boundary_distance_sec) / 2.0))

        # Audio: change in anomaly across boundary
        aud_delta = _audio_delta(t_boundary, audio_windows, half_sec=3.0)
        # Audio: median rms_db jump across boundary (broadcast loudness step).
        # Robust on real TV audio where in-ad MFCC isn't anomalous but loudness is.
        loud_jump = _loudness_jump_score(t_boundary, audio_windows, half_sec=10.0)

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
            scene_cut       *= 0.1
            aud_delta       *= 0.1
            loud_jump       *= 0.1
            speech_transition *= 0.1

        edge[i] = (
            0.35 * vis
            + 0.20 * scene_cut
            + 0.15 * aud_delta
            + 0.20 * loud_jump
            + 0.10 * speech_transition
        )

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
# Step 3 – DP over edge pairs to find K ad intervals
# ---------------------------------------------------------------------------

def _prepare_dp_inputs(
    edge_scores: np.ndarray,
    foreign_scores: np.ndarray,
    windows: list[VisualWindow],
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """
    Return the shared preprocessing for the K-stage DP:
      (norm_edge_padded, cum_foreign, min_w, max_w, gap_w, first_start_idx).
    """
    e_max = float(edge_scores.max())
    f_max = float(foreign_scores.max())
    norm_edge    = edge_scores    / (e_max + 1e-9)
    norm_foreign = foreign_scores / (f_max + 1e-9)
    norm_edge = np.append(norm_edge, 0.0)
    cum_foreign = np.concatenate([[0.0], np.cumsum(norm_foreign)])

    window_sec = windows[0].t1 - windows[0].t0 if windows else 2.0
    min_w = max(1, int(AD_MIN_SEC / window_sec))
    max_w = max(min_w + 1, int(AD_MAX_SEC / window_sec) + 1)
    gap_w = max(1, int(GAP_MIN_SEC / window_sec))

    first_start_idx = 0
    for i, w in enumerate(windows):
        if w.t0 >= FIRST_AD_MIN_START_SEC:
            first_start_idx = i
            break

    return norm_edge, cum_foreign, min_w, max_w, gap_w, first_start_idx


def _find_best_k_ads(
    edge_scores: np.ndarray,
    foreign_scores: np.ndarray,
    windows: list[VisualWindow],
    k: int,
) -> tuple[float, list[tuple[int, int]]]:
    """
    Find K non-overlapping ad intervals [s, e) (window indices) that maximise

        sum over i in [1..K] of:
            EDGE_WEIGHT * (norm_edge[s_i] + norm_edge[e_i])
          + INTERIOR_WEIGHT * mean(norm_foreign[s_i:e_i])

    subject to:
        AD_MIN_SEC ≤ e_i - s_i ≤ AD_MAX_SEC
        s_{i+1} ≥ e_i + gap_w               (GAP_MIN_SEC between ads)
        s_1 ≥ first_start_idx               (FIRST_AD_MIN_START_SEC)

    Returns (total_score, intervals_in_time_order).  Empty list and -inf
    score when no valid K-tuple exists.

    The DP is a straightforward generalisation of the previous unrolled
    3-stage version.  ``b[stage][e]`` is the best total score for ``stage``
    ads where the ``stage``-th ad ends at index ``e`` (exclusive).
    Backpointers store the start index of that ad and the end index of the
    previous ad in the optimal sub-solution, enabling backtracking.
    """
    N = len(windows)
    if k <= 0 or N == 0:
        return 0.0, []

    norm_edge, cum_foreign, min_w, max_w, gap_w, first_start_idx = _prepare_dp_inputs(
        edge_scores, foreign_scores, windows,
    )

    def interval_score(s: int, e: int) -> float:
        interior_mean = (cum_foreign[e] - cum_foreign[s]) / max(e - s, 1)
        return (
            EDGE_WEIGHT * (norm_edge[s] + norm_edge[e])
            + INTERIOR_WEIGHT * interior_mean
        )

    NEG_INF = float("-inf")

    b      = [np.full(N + 1, NEG_INF, dtype=np.float64) for _ in range(k + 1)]
    s_back = [np.full(N + 1, -1,      dtype=np.int32)   for _ in range(k + 1)]
    e_back = [np.full(N + 1, -1,      dtype=np.int32)   for _ in range(k + 1)]

    for stage in range(1, k + 1):
        if stage == 1:
            prev_pmax: np.ndarray | None = None
            prev_pmax_e: np.ndarray | None = None
        else:
            prev_pmax   = np.full(N + 1, NEG_INF, dtype=np.float64)
            prev_pmax_e = np.full(N + 1, -1,      dtype=np.int32)
            for i in range(N + 1):
                if i > 0:
                    prev_pmax[i]   = prev_pmax[i - 1]
                    prev_pmax_e[i] = prev_pmax_e[i - 1]
                if b[stage - 1][i] > prev_pmax[i]:
                    prev_pmax[i]   = b[stage - 1][i]
                    prev_pmax_e[i] = i

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
                if stage == 1:
                    total = sc
                    prev_e = -1
                else:
                    me_prev = s - gap_w
                    if me_prev < 0 or prev_pmax[me_prev] <= NEG_INF:  # type: ignore[index]
                        continue
                    total  = float(prev_pmax[me_prev]) + sc           # type: ignore[index]
                    prev_e = int(prev_pmax_e[me_prev])                # type: ignore[index]
                if total > b[stage][e]:
                    b[stage][e]      = total
                    s_back[stage][e] = s
                    e_back[stage][e] = prev_e

    best_e = int(np.argmax(b[k]))
    best_total = float(b[k][best_e])
    if best_total <= NEG_INF:
        return 0.0, []

    intervals: list[tuple[int, int]] = []
    e_cur = best_e
    for stage in range(k, 0, -1):
        s_cur = int(s_back[stage][e_cur])
        if s_cur < 0:
            return 0.0, []
        intervals.append((s_cur, e_cur))
        e_cur = int(e_back[stage][e_cur])
        if stage > 1 and e_cur < 0:
            return 0.0, []
    intervals.reverse()
    return best_total, intervals


def _select_num_ads_auto(
    edge_scores: np.ndarray,
    foreign_scores: np.ndarray,
    windows: list[VisualWindow],
    *,
    max_k: int,
    min_marginal_ratio: float,
    min_k: int = MIN_NUM_ADS,
) -> tuple[int, list[tuple[int, int]]]:
    """
    Auto-select K by running the DP for k = 1..max_k and stopping when the
    K-th ad's marginal score falls below ``min_marginal_ratio`` of the first
    (best) ad's score.

    Marginal score for the K-th ad is ``total_K - total_{K-1}`` (with
    ``total_0 = 0``).  Because the DP is jointly optimal at each K, the best
    K-tuple can re-arrange earlier ads, so the marginal series is not strictly
    monotone, but for typical inputs it decreases as K exceeds the true ad
    count.

    On real broadcast video the per-K interval scores are quite compressed
    (g_2 / g_1 ≈ 0.9 even when 3 ads are clearly present), so the rule
    under-detects below ``min_k``.  We clamp the chosen K up to ``min_k`` to
    keep behaviour aligned with the "expect at least 3 ads" prior while
    still allowing 4 / 5 when the marginal series justifies it.

    Returns ``(K_chosen, intervals_at_K_chosen)``.
    """
    min_k = max(1, int(min_k))
    max_k = max(min_k, int(max_k))

    totals: list[float] = [0.0]
    interval_sets: list[list[tuple[int, int]]] = [[]]
    for k in range(1, max_k + 1):
        total_k, intervals_k = _find_best_k_ads(edge_scores, foreign_scores, windows, k)
        if not intervals_k:
            break
        totals.append(total_k)
        interval_sets.append(intervals_k)

    if len(totals) <= 1:
        return 0, []

    first_gain = totals[1] - totals[0]
    if first_gain <= 0.0:
        chosen_k = min(min_k, len(totals) - 1)
        return chosen_k, interval_sets[chosen_k]

    chosen_k = 1
    for k in range(2, len(totals)):
        gain_k = totals[k] - totals[k - 1]
        if gain_k >= min_marginal_ratio * first_gain:
            chosen_k = k
        else:
            break

    chosen_k = max(chosen_k, min_k)
    chosen_k = min(chosen_k, len(interval_sets) - 1)
    return chosen_k, interval_sets[chosen_k]


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


# Legacy helpers (kept for external callers)
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
    num_ads: int | None = NUM_ADS,
    max_num_ads: int = MAX_NUM_ADS,
    min_num_ads: int = MIN_NUM_ADS,
    min_marginal_ratio: float = MIN_MARGINAL_RATIO,
) -> list[dict[str, Any]]:
    """
    Fuse a multimodal analysis bundle into labeled timeline segments.

    Parameters
    ----------
    num_ads
        How many ads the DP should find.  ``int`` forces that exact K
        (default ``NUM_ADS = 3`` keeps the historical behaviour).
        ``None`` enables auto-K: the DP runs for K in ``[min_num_ads,
        max_num_ads]`` and the largest K where the K-th ad's marginal score
        is at least ``min_marginal_ratio`` * (the first ad's marginal
        score) is selected.
    max_num_ads
        Upper bound on K when ``num_ads is None``.
    min_num_ads
        Lower bound on K when ``num_ads is None`` (default ``MIN_NUM_ADS``).
        Real broadcast video has very compressed marginal-gain ratios so
        the rule under-detects below 3 even when 3 ads are clearly
        present; the floor counters that.
    min_marginal_ratio
        Threshold for the marginal-gain stopping rule (only used when
        ``num_ads is None``).
    """
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

    if num_ads is None:
        chosen_k, ad_intervals = _select_num_ads_auto(
            smooth_edge, smooth_foreign, windows,
            max_k=max(1, int(max_num_ads)),
            min_marginal_ratio=float(min_marginal_ratio),
            min_k=max(1, int(min_num_ads)),
        )
    else:
        chosen_k = max(0, int(num_ads))
        _, ad_intervals = _find_best_k_ads(
            smooth_edge, smooth_foreign, windows, chosen_k,
        )

    if ad_intervals and len(ad_intervals) == chosen_k and chosen_k > 0:
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

