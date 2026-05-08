from __future__ import annotations

import json
import re
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

from schemas.modality import SpeechSpan, VisualWindow

DEFAULT_EDGE_TITLE_CARD_SEC = 30.0
DEFAULT_ENDING_SEQUENCE_MIN_SEC = 18.0
DEFAULT_SEMANTIC_OUTRO_THRESHOLD = 0.62

ENDING_TITLE_TERMS = [
    "thanks for watching",
    "thank you",
    "credits",
    "directed by",
    "produced by",
    "executive producer",
    "copyright",
    "all rights reserved",
    "www.",
    ".com",
]


@lru_cache(maxsize=1)
def _default_outro_phrases() -> tuple[str, ...]:
    signals_file = Path(__file__).with_name("ad_signals.json")
    if not signals_file.is_file():
        return ()
    data = json.loads(signals_file.read_text(encoding="utf-8"))
    phrases = data.get("phrases", {}).get("outro", [])
    return tuple(str(phrase).lower() for phrase in phrases if str(phrase).strip())


def _span_source(span: SpeechSpan) -> str:
    return str((span.model_extra or {}).get("source", "asr"))


def _speech_text_for_range(
    t0: float,
    t1: float,
    speech_spans: Sequence[SpeechSpan],
    *,
    margin_sec: float = 1.5,
) -> str:
    lo = t0 - margin_sec
    hi = t1 + margin_sec
    chunks = [
        span.text.lower()
        for span in speech_spans
        if span.t1 >= lo
        and span.t0 <= hi
        and span.text
        and _span_source(span) not in {"semantic", "semantic_structure"}
    ]
    return " ".join(chunks)


def _ocr_text_for_range(
    t0: float,
    t1: float,
    speech_spans: Sequence[SpeechSpan],
    *,
    margin_sec: float = 1.5,
) -> str:
    lo = t0 - margin_sec
    hi = t1 + margin_sec
    chunks = [
        span.text.lower()
        for span in speech_spans
        if span.t1 >= lo
        and span.t0 <= hi
        and span.text
        and _span_source(span) == "ocr"
    ]
    return " ".join(chunks)


def _asr_spans_in_range(t0: float, t1: float, speech_spans: Sequence[SpeechSpan]) -> list[SpeechSpan]:
    return [
        span
        for span in speech_spans
        if span.t1 >= t0
        and span.t0 <= t1
        and span.text
        and _span_source(span) not in {"ocr", "semantic", "semantic_structure"}
    ]


def _semantic_outro_score_for_range(
    t0: float,
    t1: float,
    speech_spans: Sequence[SpeechSpan],
    *,
    margin_sec: float = 8.0,
) -> float:
    lo = t0 - margin_sec
    hi = t1 + margin_sec
    outro_score = 0.0
    for span in speech_spans:
        if _span_source(span) != "semantic_structure":
            continue
        if span.t1 < lo or span.t0 > hi:
            continue
        extra = span.model_extra or {}
        outro_score = max(outro_score, float(extra.get("semantic_outro_score", 0.0)))
    return outro_score


def _graphics_ratio(windows: Sequence[VisualWindow], indices: Sequence[int], t0: float, t1: float) -> float:
    selected = [
        windows[index]
        for index in indices
        if windows[index].t1 >= t0 and windows[index].t0 <= t1
    ]
    if not selected:
        return 0.0
    hits = sum(
        1
        for window in selected
        if window.high_text_density
        or window.visual_hypothesis == "graphics_heavy"
        or (
            window.visual_hypothesis == "static"
            and window.motion_score < 0.08
            and window.luminance_mean < 0.08
        )
    )
    return hits / len(selected)


def has_ending_title_signal(
    t0: float,
    t1: float,
    windows: Sequence[VisualWindow],
    indices: Sequence[int],
    speech_spans: Sequence[SpeechSpan],
    *,
    outro_phrases: Sequence[str] | None = None,
    semantic_threshold: float = DEFAULT_SEMANTIC_OUTRO_THRESHOLD,
) -> bool:
    phrases = tuple(phrase.lower() for phrase in (outro_phrases or _default_outro_phrases()))
    combined_text = _speech_text_for_range(t0, t1, speech_spans, margin_sec=4.0)
    ocr_text = _ocr_text_for_range(t0, t1, speech_spans, margin_sec=4.0)
    visual_title_ratio = _graphics_ratio(windows, indices, t0, t1)
    semantic_outro_score = _semantic_outro_score_for_range(t0, t1, speech_spans, margin_sec=8.0)

    text_hit = (
        any(term in combined_text for term in ENDING_TITLE_TERMS)
        or any(phrase in combined_text for phrase in phrases)
    )
    semantic_hit = semantic_outro_score >= semantic_threshold and (
        text_hit
        or bool(ocr_text)
        or visual_title_ratio >= 0.30
    )
    return text_hit or (bool(ocr_text) and visual_title_ratio >= 0.30) or semantic_hit


def find_outro_start_time(
    windows: Sequence[VisualWindow],
    run_indices: Sequence[int],
    duration: float,
    speech_spans: Sequence[SpeechSpan],
    *,
    outro_used: bool = False,
    outro_phrases: Sequence[str] | None = None,
    edge_title_card_sec: float = DEFAULT_EDGE_TITLE_CARD_SEC,
    ending_sequence_min_sec: float = DEFAULT_ENDING_SEQUENCE_MIN_SEC,
) -> float | None:
    if outro_used or not run_indices:
        return None

    run_start = windows[run_indices[0]].t0
    run_end = windows[run_indices[-1]].t1
    if duration - run_end > 2.0:
        return None

    asr_spans = _asr_spans_in_range(run_start, run_end, speech_spans)
    last_asr_end = max((span.t1 for span in asr_spans), default=None)
    if last_asr_end is None:
        tail_start = max(run_start, run_end - edge_title_card_sec)
        if has_ending_title_signal(tail_start, run_end, windows, run_indices, speech_spans, outro_phrases=outro_phrases):
            return tail_start
        return None

    if run_end - last_asr_end < ending_sequence_min_sec:
        return None

    tail_start = max(last_asr_end, run_start)
    if has_ending_title_signal(tail_start, run_end, windows, run_indices, speech_spans, outro_phrases=outro_phrases):
        return tail_start
    return None
