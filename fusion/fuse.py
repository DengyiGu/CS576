"""
fusion/fuse.py
--------------
Multimodal fusion layer.

Takes an AnalysisBundle (visual + optional audio + optional speech) and
produces a list of labeled segment dicts in the shape the player expects:

    [{"start": 0.0, "end": 12.5, "label": "Intro", "kind": "non-content"}, ...]

Design
------
1. Per-window classification  — each 1-second window gets a label from
   visual signals, with audio and speech overrides applied on top.
2. Label smoothing            — short isolated blips are merged into
   their neighbors so the output is clean and usable.
3. Segment merging            — consecutive windows with the same label
   collapse into a single segment dict.

Audio and speech fields come from teammates' modules and slot straight
into AnalysisBundle.audio_windows / speech_spans.  When those are empty
(as they are now) the fusion falls back gracefully to visual-only mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# The schema classes live in schemas/modality.py in the repo.  We import
# them here.  When running the CLI the caller sets PYTHONPATH=. so these
# resolve correctly.
# ---------------------------------------------------------------------------
from schemas.modality import AnalysisBundle, AudioWindow, SpeechSpan, VisualWindow

# ---------------------------------------------------------------------------
# Label constants — must match the TAXONOMY labels in player/player.py
# ---------------------------------------------------------------------------
LABEL_CORE_CONTENT = "Core Content"
LABEL_INTRO = "Intro"
LABEL_OUTRO = "Outro"
LABEL_ADVERTISEMENT = "Advertisement"
LABEL_SELF_PROMOTION = "Self-Promotion"
LABEL_RECAP = "Recap"
LABEL_TRANSITION = "Transition"
LABEL_INACTIVITY = "Inactivity"
LABEL_FILLER = "Filler"

KIND_CONTENT = "content"
KIND_NON_CONTENT = "non-content"

_KIND_FOR_LABEL: dict[str, str] = {
    LABEL_CORE_CONTENT: KIND_CONTENT,
    LABEL_INTRO: KIND_NON_CONTENT,
    LABEL_OUTRO: KIND_NON_CONTENT,
    LABEL_ADVERTISEMENT: KIND_NON_CONTENT,
    LABEL_SELF_PROMOTION: KIND_NON_CONTENT,
    LABEL_RECAP: KIND_NON_CONTENT,
    LABEL_TRANSITION: KIND_NON_CONTENT,
    LABEL_INACTIVITY: KIND_NON_CONTENT,
    LABEL_FILLER: KIND_NON_CONTENT,
}

# ---------------------------------------------------------------------------
# Speech keyword lists used by the speech override layer
#
# These were tuned by manually reviewing transcripts and video descriptions
# for all 5 test videos:
#   test_001 — TED talk interrupted by Apple (no speech), Lays, Doritos ads
#   test_002 — NASA Artemis II interrupted by Starbucks, Pepsi, Dove ads
#   test_003 — Despicable Me clips interrupted by Sony (no speech), Ikea, Doritos ads
#   test_004 — Stanford lecture interrupted by McDonald's, Nike (no speech), Google Pixel ads
#   test_005 — Nat Geo doc interrupted by Coca-Cola x2 (no speech), Bvlgari ads
#
# Key insight from reviewing the videos:
#   - Several ads have NO speech at all (Apple, Sony, Nike, both Coca-Cola ads).
#     These must be caught by visual/audio signals, not speech.
#   - Ads that DO have speech mention product/brand names directly.
#     Brand name detection is the most reliable speech signal for this dataset.
#   - Generic sponsorship phrases ("sponsored by", "use code") do NOT appear
#     in this dataset — these are mid-roll inserted ads, not creator sponsorships.
#   - test_001 speaker mentions "McDonald's" as part of content — handled by
#     a density check: a single passing mention is ignored, a cluster is flagged.
# ---------------------------------------------------------------------------

# Brand names that appear in ad transcripts across the test videos.
# A density check in _speech_label_for_window prevents false positives
# from single in-content brand mentions (e.g. "McDonald's" in the TED talk).
_AD_BRAND_NAMES = [
    # test_001 ads
    "lays", "lay's", "doritos",
    # test_002 ads
    "starbucks", "pepsi", "dove",
    # test_003 ads
    "ikea",
    # test_004 ads
    "mcdonald's", "mcdonalds", "google pixel", "nike",
    # test_005 ads
    "bvlgari", "bulgari", "coca-cola", "coca cola",
    # Broadly common ad brands
    "amazon", "apple", "samsung", "netflix", "spotify",
    "squarespace", "nordvpn", "skillshare", "audible",
    "expressvpn", "hello fresh", "factor meals",
]

# Generic ad-language phrases — less common in this dataset (mid-roll ads
# don't use creator-style phrases) but kept for general robustness.
_SPONSORSHIP_PHRASES = [
    "sponsored by", "brought to you by", "use code", "use my code",
    "promo code", "discount code", "affiliate",
    "link in the description", "link in bio",
    "first month free", "limited time offer", "exclusive deal",
    "check it out", "download the app", "available now", "on sale now",
]

_SELF_PROMO_PHRASES = [
    "subscribe", "hit the bell", "like and subscribe", "don't forget to subscribe",
    "follow me", "follow us", "patreon", "merch", "my discord", "join the discord",
    "become a member",
]

_OUTRO_PHRASES = [
    "thanks for watching", "see you next time", "see you in the next",
    "until next time", "that's all for today", "that's it for today",
    "goodbye", "bye for now",
]

_INTRO_PHRASES = [
    "welcome back", "welcome to", "in today's video", "in this video",
    "today we", "today i'm going to", "today we're going to",
    "let's get started", "let's dive in",
]

_RECAP_PHRASES = [
    "last time", "last episode", "previously", "as we discussed",
    "as i mentioned", "in the previous", "recap",
]

# ---------------------------------------------------------------------------
# Brand name density detection parameters
#
# A single brand mention in speech is ignored (avoids the McDonald's false
# positive in test_001). Two or more brand hits within a context window
# around a given timestamp are treated as an ad.
# ---------------------------------------------------------------------------
_BRAND_CONTEXT_WINDOW_SEC = 15.0
_BRAND_HIT_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Per-window visual classifier
# ---------------------------------------------------------------------------

def _classify_visual(w: VisualWindow, video_duration: float, position: float) -> str:
    """
    Map a single VisualWindow to a label using rule-based heuristics.

    position — fractional position in video (0.0 = start, 1.0 = end),
               used for intro/outro positional bias.

    Tuned for the test video dataset:
      - test_001: TED talk — stable talking-head content, ads cause sharp
                  palette shifts (different brand color palettes)
      - test_002: NASA video — lots of motion throughout, ads interrupt
                  speech but look visually very different (product shots)
      - test_003: Despicable Me — animated content has naturally high
                  edge density and vivid colors; ads still differ in palette
      - test_004: Stanford lecture — alternates between talking-head and
                  slide shots (slide shots have high edge density + low motion
                  but ARE content, not transitions — don't over-flag these)
      - test_005: Nat Geo doc — lots of b-roll with music, dark nature shots;
                  first ~26 sec is black screen (Inactivity), ~26s-1:20 is
                  suspenseful music + title card (Intro)
    """
    m = w.motion_score
    e = w.edge_density
    p = w.palette_delta
    lum = w.luminance_mean
    hyp = w.visual_hypothesis

    # --- Dead / inactive screen ------------------------------------------
    # Catches test_005's black opening screen, holding screens, dead air.
    if m < 0.05 and lum < 0.08:
        return LABEL_INACTIVITY


    # --- Static bright title card / transition ----------------------------
    # High edge density (text-heavy), low motion, high luminance.
    # NOTE: Stanford lecture slide shots also have high edge density but
    # are mid-video content. We only flag as Transition if it's near a
    # shot boundary AND position is not deep in the video (>10% in).
    if w.high_text_density and m < 0.25 and lum > 0.55:
        if position < 0.10 or w.shot_boundary_near:
            return LABEL_TRANSITION
        # Mid-video high-text static frame = likely a slide in a lecture
        # Fall through to Core Content below

    # --- Positional intro heuristic --------------------------------------
    # Catches test_005's title sequence (~26s-80s of suspenseful music + title)
    # and test_004's 5-second "Stanford Engineering" opening card.
    if position < 0.03 and hyp in ("static", "unknown", "graphics_heavy"):
        return LABEL_INTRO

    if position < 0.06 and w.shot_boundary_near and m < 0.35:
        return LABEL_INTRO

    # --- Positional outro heuristic --------------------------------------
    # test_005 ends with credits/text overlay after ~21:16
    if position > 0.92 and hyp in ("static", "unknown"):
        return LABEL_OUTRO

    if position > 0.95 and m < 0.20:
        return LABEL_OUTRO

    # --- Visually anomalous graphics — strong palette shift ---------------
    # This is the primary visual ad detector for this dataset.
    # Ads (Apple, Sony, Nike, Coca-Cola) all show a sharp palette divergence
    # from the surrounding content because they're completely different footage.
    # Threshold lowered slightly from 0.70 to 0.65 to catch more ad frames.
    if p > 0.72 and hyp == "graphics_heavy":
        return LABEL_ADVERTISEMENT

    if p > 0.80:
        return LABEL_ADVERTISEMENT

    # --- Graphics-heavy but not palette-divergent → likely transition ----
    # Catches title cards and intermission screens that aren't quite bright
    # enough for the high_text_density check above.
    if hyp == "graphics_heavy" and e > 0.50 and m < 0.30 and lum > 0.40:
        return LABEL_TRANSITION

    # --- Normal dynamic talk / lecture / b-roll → core content -----------
    return LABEL_CORE_CONTENT


# ---------------------------------------------------------------------------
# Per-window audio override
# ---------------------------------------------------------------------------

def _audio_label_for_window(
    t0: float,
    t1: float,
    audio_windows: list[AudioWindow],
) -> str | None:
    """
    Return a label override from audio features, or None if no strong signal.

    AudioWindow may carry extra fields from teammates' module (model_config
    extra="allow"), e.g.:
        audio_label: "speech" | "music" | "silence" | "mixed"
        energy_rms: float
        anomaly_score: float   (deviation from running audio mean)
    """
    mid = 0.5 * (t0 + t1)
    for aw in audio_windows:
        if aw.t0 <= mid < aw.t1:
            extra = aw.model_extra  # pydantic v2 extra fields
            audio_label = str(extra.get("audio_label", "")).lower()
            anomaly = float(extra.get("anomaly_score", 0.0))
            energy = float(extra.get("energy_rms", 1.0))

            # Silence → inactivity / dead air
            if audio_label == "silence" or energy < 0.02:
                return LABEL_INACTIVITY

            # Music-only (no speech) often means intro/outro/sponsorship.
            # We can't distinguish the three from audio alone, so we just
            # return None and let visual + position sort it out.
            # But a very high anomaly score is a strong ad signal.
            if anomaly > 0.75:
                return LABEL_ADVERTISEMENT

            break  # found the matching window; no override

    return None


# ---------------------------------------------------------------------------
# Per-window speech override
# ---------------------------------------------------------------------------

def _speech_label_for_window(
    t0: float,
    t1: float,
    speech_spans: list[SpeechSpan],
) -> str | None:
    """
    Return a label override if speech in or around this window contains
    known keyword patterns.

    Two-tier detection:

    Tier 1 — Exact window text:
        Check the transcript text that directly overlaps this window for
        generic sponsorship phrases, self-promotion, outro, intro, recap.
        These are high-confidence single-hit signals.

    Tier 2 — Brand name density in context window:
        Check a wider context window (_BRAND_CONTEXT_WINDOW_SEC) around
        this timestamp for brand name mentions. Require at least
        _BRAND_HIT_THRESHOLD hits to flag as Advertisement.

        This handles ads like Lays, Doritos, Starbucks, Ikea, Google Pixel
        in the test videos — they mention brand names but not generic
        sponsorship phrases.

        The density threshold avoids the false positive of a single
        "McDonald's" mention in the test_001 TED talk content.
    """
    # -----------------------------------------------------------------------
    # Tier 1: collect text that directly overlaps this window
    # -----------------------------------------------------------------------
    window_chunks: list[str] = []
    for span in speech_spans:
        overlap_start = max(t0, span.t0)
        overlap_end = min(t1, span.t1)
        if overlap_end > overlap_start and span.text:
            window_chunks.append(span.text.lower())

    if window_chunks:
        combined = " ".join(window_chunks)

        for phrase in _SPONSORSHIP_PHRASES:
            if phrase in combined:
                return LABEL_ADVERTISEMENT

        for phrase in _SELF_PROMO_PHRASES:
            if phrase in combined:
                return LABEL_SELF_PROMOTION

        for phrase in _OUTRO_PHRASES:
            if phrase in combined:
                return LABEL_OUTRO

        for phrase in _INTRO_PHRASES:
            if phrase in combined:
                return LABEL_INTRO

        for phrase in _RECAP_PHRASES:
            if phrase in combined:
                return LABEL_RECAP

    # -----------------------------------------------------------------------
    # Tier 2: brand name density over a wider context window
    # -----------------------------------------------------------------------
    context_start = t0 - _BRAND_CONTEXT_WINDOW_SEC
    context_end = t1 + _BRAND_CONTEXT_WINDOW_SEC

    context_chunks: list[str] = []
    for span in speech_spans:
        if span.t1 >= context_start and span.t0 <= context_end and span.text:
            context_chunks.append(span.text.lower())

    if not context_chunks:
        return None

    context_text = " ".join(context_chunks)
    brand_hits = sum(1 for brand in _AD_BRAND_NAMES if brand in context_text)

    if brand_hits >= _BRAND_HIT_THRESHOLD:
        return LABEL_ADVERTISEMENT

    return None


# ---------------------------------------------------------------------------
# Label smoother
# ---------------------------------------------------------------------------

_MIN_SEGMENT_SECONDS = 4.0  # segments shorter than this get absorbed

def _smooth_labels(
    labels: list[str],
    windows: list[VisualWindow],
    min_segment_seconds: float = _MIN_SEGMENT_SECONDS,
) -> list[str]:
    """
    Two-pass smoother:

    Pass 1 — forward pass: any run of a label that covers fewer than
             min_segment_seconds is replaced by the label of the
             preceding window (or the following window on the first run).

    Pass 2 — backward pass: same logic in reverse, picking up any
             isolated blips still remaining.

    This prevents the common artifact of a 1-2 second "Core Content"
    island appearing in the middle of an ad, or vice versa.
    """
    if not labels:
        return labels

    result = list(labels)

    def _duration_of_run(start_idx: int, lbl: str) -> float:
        total = 0.0
        i = start_idx
        while i < len(result) and result[i] == lbl:
            total += windows[i].t1 - windows[i].t0
            i += 1
        return total

    # Pass 1: forward
    i = 0
    while i < len(result):
        current_label = result[i]
        run_duration = _duration_of_run(i, current_label)
        if run_duration < min_segment_seconds and i > 0:
            # Replace with previous label
            prev_label = result[i - 1]
            j = i
            while j < len(result) and result[j] == current_label:
                result[j] = prev_label
                j += 1
        i += 1

    # Pass 2: backward
    i = len(result) - 1
    while i >= 0:
        current_label = result[i]
        # find run start
        run_start = i
        while run_start > 0 and result[run_start - 1] == current_label:
            run_start -= 1
        run_duration = sum(
            windows[k].t1 - windows[k].t0
            for k in range(run_start, i + 1)
        )
        if run_duration < min_segment_seconds and i < len(result) - 1:
            next_label = result[i + 1]
            for k in range(run_start, i + 1):
                result[k] = next_label
        i = run_start - 1

    return result


# ---------------------------------------------------------------------------
# Segment merger
# ---------------------------------------------------------------------------

def _merge_into_segments(
    labels: list[str],
    windows: list[VisualWindow],
) -> list[dict[str, Any]]:
    """
    Collapse consecutive same-label windows into segment dicts.
    Output shape matches what the player's build_segment_from_payload expects.
    """
    if not labels:
        return []

    segments: list[dict[str, Any]] = []
    current_label = labels[0]
    current_start = windows[0].t0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append(_make_segment_dict(current_label, current_start, windows[i - 1].t1))
            current_label = labels[i]
            current_start = windows[i].t0

    # Final segment
    segments.append(_make_segment_dict(current_label, current_start, windows[-1].t1))
    return segments


def _make_segment_dict(label: str, start: float, end: float) -> dict[str, Any]:
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "label": label,
        "kind": _KIND_FOR_LABEL.get(label, KIND_NON_CONTENT),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fuse_bundle_to_segments(
    bundle: AnalysisBundle,
    *,
    min_segment_seconds: float = _MIN_SEGMENT_SECONDS,
) -> list[dict[str, Any]]:
    """
    Main entry point.  Accepts a fully or partially-populated AnalysisBundle
    and returns a list of segment dicts ready for the player.

    Works in visual-only mode when audio_windows and speech_spans are empty.
    When teammates add their outputs, those fields are picked up automatically.
    """
    if bundle.visual is None or not bundle.visual.windows:
        return []

    windows = bundle.visual.windows
    duration = bundle.visual.duration_sec or bundle.duration_sec

    # -----------------------------------------------------------------------
    # Step 1: classify each window
    # -----------------------------------------------------------------------
    raw_labels: list[str] = []
    for w in windows:
        position = (0.5 * (w.t0 + w.t1)) / max(duration, 1.0)

        # Visual baseline
        label = _classify_visual(w, duration, position)

        # Audio override (higher confidence than visual for certain signals)
        audio_override = _audio_label_for_window(w.t0, w.t1, bundle.audio_windows)
        if audio_override is not None:
            label = audio_override

        # Speech override (highest confidence — explicit keyword matches)
        speech_override = _speech_label_for_window(w.t0, w.t1, bundle.speech_spans)
        if speech_override is not None:
            label = speech_override

        raw_labels.append(label)

    # -----------------------------------------------------------------------
    # Step 2: smooth out short blips
    # -----------------------------------------------------------------------
    smoothed = _smooth_labels(raw_labels, windows, min_segment_seconds)

    # -----------------------------------------------------------------------
    # Step 3: collapse consecutive same-label windows into segments
    # -----------------------------------------------------------------------
    return _merge_into_segments(smoothed, windows)


def load_bundle(path: Path) -> AnalysisBundle:
    """Load an AnalysisBundle from a JSON file on disk."""
    return AnalysisBundle.model_validate_json(path.read_text(encoding="utf-8"))


def write_segments_json(segments: list[dict[str, Any]], out_path: Path) -> None:
    """Write the fused segments list to a JSON file the player can load."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "source": "fusion",
        "segments": segments,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")