"""
Multimodal fusion layer.

Takes an AnalysisBundle (visual + optional audio + optional speech) and
produces labeled segment dicts in the shape the player expects:
    [{"start": 0.0, "end": 12.5, "label": "Intro", "kind": "non-content"}, ...]

Works in three steps:
1. Per-window classification: each window gets a label from visual signals, with audio and speech overrides 
   layered on top.
2. Label smoothing: short isolated blips get absorbed into their neighbors so the output is clean and doesn't 
   flicker.
3. Segment merging: consecutive windows with the same label collapse into a single segment dict.

Falls back to visual-only mode when audio/speech are not available.
Teammate outputs slot in automatically once their fields are populated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemas.modality import AnalysisBundle, AudioWindow, SpeechSpan, VisualWindow

# Label constants — must match the TAXONOMY labels in player/player.py
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


# Speech keyword detection
#
# Brand names and keyword phrases are loaded from fusion/ad_signals.json so the lists can be extended without 
# touching this file.
#
# A density check on brand names prevents false positives — a single brand name mentioned in passing during 
# content won't trigger the threshold. Actual ads cluster multiple brand references together.

def _load_ad_signals() -> tuple[list[str], dict[str, list[str]]]:
    """
    Load brand names and keyword phrases from ad_signals.json. Returns (brand_names, phrases) where phrases is a 
    dict keyed by category. Falls back to empty lists if the file is missing.
    """
    signals_file = Path(__file__).parent / "ad_signals.json"
    if not signals_file.is_file():
        return [], {}
    data = json.loads(signals_file.read_text(encoding="utf-8"))

    # Flatten all brand categories into a single deduplicated list
    brand_names: list[str] = []
    seen: set[str] = set()
    for category_brands in data.get("brands", {}).values():
        for name in category_brands:
            if name not in seen:
                seen.add(name)
                brand_names.append(name)

    phrases: dict[str, list[str]] = data.get("phrases", {})
    return brand_names, phrases


_AD_BRAND_NAMES, _AD_PHRASES = _load_ad_signals()

_SPONSORSHIP_PHRASES = _AD_PHRASES.get("sponsorship", [])
_SELF_PROMO_PHRASES = _AD_PHRASES.get("self_promotion", [])
_OUTRO_PHRASES = _AD_PHRASES.get("outro", [])
_INTRO_PHRASES = _AD_PHRASES.get("intro", [])
_RECAP_PHRASES = _AD_PHRASES.get("recap", [])

# Requiring 2+ brand hits in a 30-second window (15s before + 15s after)
# avoids flagging isolated content mentions as ads.
_BRAND_CONTEXT_WINDOW_SEC = 15.0
_BRAND_HIT_THRESHOLD = 2


def _classify_visual(w: VisualWindow, video_duration: float, position: float) -> str:
    """
    Map a single VisualWindow to a label using rule-based heuristics.
    position is the fractional position in the video (0.0 = start, 1.0 = end), used for intro/outro positional 
    checks.
    """
    m = w.motion_score
    e = w.edge_density
    p = w.palette_delta
    lum = w.luminance_mean
    hyp = w.visual_hypothesis

    # Very dark and very still — black screens, holding screens, dead air. Thresholds kept tight so dark nature 
    # shots and low-light footage aren't incorrectly caught.
    if m < 0.05 and lum < 0.08:
        return LABEL_INACTIVITY

    # Text-heavy, low motion, bright frame — likely a title card or transition. We skip this for mid-video frames 
    # without a nearby shot boundary so that lecture slides and text overlays on content aren't flagged as 
    # transitions.
    if w.high_text_density and m < 0.25 and lum > 0.55:
        if position < 0.10 or w.shot_boundary_near:
            return LABEL_TRANSITION

    # Opening intro — catches title sequences, logo cards, and opening slates
    # in the first few percent of the video.
    if position < 0.03 and hyp in ("static", "unknown", "graphics_heavy"):
        return LABEL_INTRO

    if position < 0.06 and w.shot_boundary_near and m < 0.35:
        return LABEL_INTRO

    # Outro — catches credit sequences and closing cards near the end.
    if position > 0.92 and hyp in ("static", "unknown"):
        return LABEL_OUTRO

    if position > 0.95 and m < 0.20:
        return LABEL_OUTRO

    # Strong palette divergence from the video's running average is the primary visual ad signal. Inserted ads 
    # look completely different from the surrounding content visually, causing a sharp shift in color distribution.
    # Thresholds are conservative to reduce false positives on slides and visually varied content like animation.
    if p > 0.72 and hyp == "graphics_heavy":
        return LABEL_ADVERTISEMENT

    if p > 0.80:
        return LABEL_ADVERTISEMENT

    # Graphics-heavy but not palette-divergent — likely a title card or intermission screen that didn't meet the 
    # brightness threshold above.
    if hyp == "graphics_heavy" and e > 0.50 and m < 0.30 and lum > 0.40:
        return LABEL_TRANSITION

    return LABEL_CORE_CONTENT


def _audio_label_for_window(
    t0: float,
    t1: float,
    audio_windows: list[AudioWindow],
) -> str | None:
    """
    Return a label override from audio features, or None if no strong signal.
    Reads extra fields from the audio teammate's AudioWindow objects:
        audio_label: "speech" | "music" | "silence" | "mixed"
        energy_rms: float (0-1)
        anomaly_score: float (0-1, deviation from running audio mean)
    """
    mid = 0.5 * (t0 + t1)
    for aw in audio_windows:
        if aw.t0 <= mid < aw.t1:
            extra = aw.model_extra
            audio_label = str(extra.get("audio_label", "")).lower()
            anomaly = float(extra.get("anomaly_score", 0.0))
            energy = float(extra.get("energy_rms", 1.0))

            if audio_label == "silence" or energy < 0.02:
                return LABEL_INACTIVITY

            # A very high anomaly score means the audio diverges dramatically from the video baseline 
            # — strong signal for an inserted ad.
            if anomaly > 0.75:
                return LABEL_ADVERTISEMENT

            break

    return None


def _speech_label_for_window(
    t0: float,
    t1: float,
    speech_spans: list[SpeechSpan],
) -> str | None:
    """
    Return a label override if speech around this window matches known patterns.
    Tier 1 checks transcript text that directly overlaps the window for generic sponsorship phrases, 
    self-promotion language, outros, intros, and recap language. A single match is enough to override.
    Tier 2 scans a wider context window for brand name density. Requiring multiple brand hits prevents false 
    positives from isolated brand name mentions in content (e.g. McDonald's in the TED talk).
    """
    # Tier 1: check text that directly overlaps this window
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

    # Tier 2: brand name density over a wider context window
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


_MIN_SEGMENT_SECONDS = 4.0


def _smooth_labels(
    labels: list[str],
    windows: list[VisualWindow],
    min_segment_seconds: float = _MIN_SEGMENT_SECONDS,
) -> list[str]:
    """
    Two-pass smoother that absorbs short runs into their neighbors.
    Forward pass: any run shorter than min_segment_seconds gets replaced by the label of the preceding window.
    Backward pass: same logic in reverse to catch anything the forward pass missed.
    This prevents short "Core Content" blips inside an ad block, or a stray "Advertisement" label appearing 
    mid-content.
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

    # Forward pass
    i = 0
    while i < len(result):
        current_label = result[i]
        run_duration = _duration_of_run(i, current_label)
        if run_duration < min_segment_seconds and i > 0:
            prev_label = result[i - 1]
            j = i
            while j < len(result) and result[j] == current_label:
                result[j] = prev_label
                j += 1
        i += 1

    # Backward pass
    i = len(result) - 1
    while i >= 0:
        current_label = result[i]
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


def _merge_into_segments(
    labels: list[str],
    windows: list[VisualWindow],
) -> list[dict[str, Any]]:
    """
    Collapse consecutive same-label windows into segment dicts.
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

    segments.append(_make_segment_dict(current_label, current_start, windows[-1].t1))
    return segments


def _make_segment_dict(label: str, start: float, end: float) -> dict[str, Any]:
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "label": label,
        "kind": _KIND_FOR_LABEL.get(label, KIND_NON_CONTENT),
    }


def fuse_bundle_to_segments(
    bundle: AnalysisBundle,
    *,
    min_segment_seconds: float = _MIN_SEGMENT_SECONDS,
) -> list[dict[str, Any]]:
    """
    Main entry point. Takes an AnalysisBundle and returns labeled segment dicts ready for the player. Works with 
    just visual data; audio and speech improve results when available.
    """
    if bundle.visual is None or not bundle.visual.windows:
        return []

    windows = bundle.visual.windows
    duration = bundle.visual.duration_sec or bundle.duration_sec

    raw_labels: list[str] = []
    for w in windows:
        position = (0.5 * (w.t0 + w.t1)) / max(duration, 1.0)

        label = _classify_visual(w, duration, position)

        audio_override = _audio_label_for_window(w.t0, w.t1, bundle.audio_windows)
        if audio_override is not None:
            label = audio_override

        # Speech has highest confidence — explicit keyword/brand matches override everything
        speech_override = _speech_label_for_window(w.t0, w.t1, bundle.speech_spans)
        if speech_override is not None:
            label = speech_override

        raw_labels.append(label)

    smoothed = _smooth_labels(raw_labels, windows, min_segment_seconds)
    return _merge_into_segments(smoothed, windows)


def load_bundle(path: Path) -> AnalysisBundle:
    """
    Load an AnalysisBundle from a JSON file on disk.
    """
    return AnalysisBundle.model_validate_json(path.read_text(encoding="utf-8"))


def write_segments_json(segments: list[dict[str, Any]], out_path: Path) -> None:
    """
    Write the fused segments list to a JSON file the player can load.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "source": "fusion",
        "segments": segments,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")