from __future__ import annotations

import re
from collections.abc import Sequence

from schemas.modality import SpeechSpan, VisualWindow

DEFAULT_EDGE_TITLE_CARD_SEC = 55.0
DEFAULT_MIN_EDGE_AUXILIARY_SEC = 4.0
DEFAULT_OPENING_SEQUENCE_MAX_SEC = 90.0
DEFAULT_SEMANTIC_INTRO_THRESHOLD = 0.62

OPENING_TITLE_TERMS = [
    "presents",
    "presented by",
    "films",
    "film",
    "documentary",
    "production",
    "productions",
    "studio",
    "episode",
    "official",
    "prize",
]


def _span_source(span: SpeechSpan) -> str:
    return str((span.model_extra or {}).get("source", "asr"))


def _ocr_chunks_for_range(
    t0: float,
    t1: float,
    speech_spans: Sequence[SpeechSpan],
    *,
    margin_sec: float = 1.5,
) -> list[str]:
    lo = t0 - margin_sec
    hi = t1 + margin_sec
    return [
        span.text.lower()
        for span in speech_spans
        if span.t1 >= lo
        and span.t0 <= hi
        and span.text
        and _span_source(span) == "ocr"
    ]


def _asr_spans_in_range(t0: float, t1: float, speech_spans: Sequence[SpeechSpan]) -> list[SpeechSpan]:
    return [
        span
        for span in speech_spans
        if span.t1 >= t0
        and span.t0 <= t1
        and span.text
        and _span_source(span) not in {"ocr", "semantic", "semantic_structure"}
    ]


def _semantic_intro_score_for_range(
    t0: float,
    t1: float,
    speech_spans: Sequence[SpeechSpan],
    *,
    margin_sec: float = 3.0,
) -> float:
    lo = t0 - margin_sec
    hi = t1 + margin_sec
    intro_score = 0.0
    for span in speech_spans:
        if _span_source(span) != "semantic_structure":
            continue
        if span.t1 < lo or span.t0 > hi:
            continue
        extra = span.model_extra or {}
        intro_score = max(intro_score, float(extra.get("semantic_intro_score", 0.0)))
    return intro_score


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


def has_opening_title_signal(
    t0: float,
    t1: float,
    windows: Sequence[VisualWindow],
    indices: Sequence[int],
    speech_spans: Sequence[SpeechSpan],
    *,
    semantic_threshold: float = DEFAULT_SEMANTIC_INTRO_THRESHOLD,
) -> bool:
    ocr_chunks = _ocr_chunks_for_range(t0, t1, speech_spans)
    ocr_text = " ".join(ocr_chunks)
    if not ocr_text:
        return False

    words = re.findall(r"[a-z0-9']+", ocr_text)
    word_count = len(words)
    if word_count == 0:
        return False

    numeric_ratio = sum(1 for word in words if any(ch.isdigit() for ch in word)) / word_count
    distinct_chunks = len({chunk for chunk in ocr_chunks if chunk.strip()})
    title_term_hit = any(term in ocr_text for term in OPENING_TITLE_TERMS)
    visual_title_ratio = _graphics_ratio(windows, indices, t0, t1)
    semantic_intro_score = _semantic_intro_score_for_range(t0, t1, speech_spans)
    title_card_text_shape = (
        2 <= word_count <= 28
        and distinct_chunks >= 3
        and numeric_ratio <= 0.25
        and visual_title_ratio >= 0.22
    )
    semantic_supported_title = (
        semantic_intro_score >= semantic_threshold
        and 2 <= word_count <= 40
        and numeric_ratio <= 0.30
        and visual_title_ratio >= 0.18
    )
    return title_term_hit or title_card_text_shape or semantic_supported_title


def _opening_ocr_text_end(t0: float, t1: float, speech_spans: Sequence[SpeechSpan]) -> float | None:
    ends: list[float] = []
    for span in speech_spans:
        if _span_source(span) != "ocr" or span.t1 <= t0 or span.t0 >= t1 or not span.text:
            continue
        words = re.findall(r"[a-z0-9']+", span.text.lower())
        if not words:
            continue
        alpha_words = sum(1 for word in words if any(ch.isalpha() for ch in word))
        numeric_ratio = sum(1 for word in words if any(ch.isdigit() for ch in word)) / len(words)
        if alpha_words >= 2 and numeric_ratio <= 0.40:
            ends.append(float(span.t1))
    return max(ends) if ends else None


def find_intro_end_time(
    windows: Sequence[VisualWindow],
    run_indices: Sequence[int],
    speech_spans: Sequence[SpeechSpan],
    *,
    intro_used: bool = False,
    edge_title_card_sec: float = DEFAULT_EDGE_TITLE_CARD_SEC,
    min_edge_auxiliary_sec: float = DEFAULT_MIN_EDGE_AUXILIARY_SEC,
    opening_sequence_max_sec: float = DEFAULT_OPENING_SEQUENCE_MAX_SEC,
) -> float | None:
    if intro_used or not run_indices:
        return None

    run_start = windows[run_indices[0]].t0
    run_end = windows[run_indices[-1]].t1
    if run_start > 2.0:
        return None

    asr_spans = _asr_spans_in_range(run_start, run_end, speech_spans)
    first_asr_start = min((span.t0 for span in asr_spans), default=None)

    if first_asr_start is not None and first_asr_start - run_start >= 15.0:
        intro_end = min(first_asr_start, run_start + opening_sequence_max_sec, run_end)
        if has_opening_title_signal(run_start, intro_end, windows, run_indices, speech_spans):
            ocr_intro_end = _opening_ocr_text_end(run_start, intro_end, speech_spans)
            if (
                ocr_intro_end is not None
                and ocr_intro_end - run_start >= min_edge_auxiliary_sec
                and ocr_intro_end < intro_end
            ):
                return ocr_intro_end
            return intro_end

    short_title_end = min(run_start + edge_title_card_sec, run_end)
    if has_opening_title_signal(run_start, short_title_end, windows, run_indices, speech_spans):
        ocr_intro_end = _opening_ocr_text_end(run_start, short_title_end, speech_spans)
        if (
            ocr_intro_end is not None
            and ocr_intro_end - run_start >= min_edge_auxiliary_sec
            and ocr_intro_end < short_title_end
        ):
            return ocr_intro_end
        return short_title_end
    return None
