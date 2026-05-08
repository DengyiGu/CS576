from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from fusion.intro_detector import find_intro_end_time
from fusion.outro_detector import find_outro_start_time
from schemas.modality import AnalysisBundle, AudioWindow, SpeechSpan, VisualWindow

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
LABEL_CORE_CONTENT   = "Core Content"
LABEL_INTRO          = "Intro"
LABEL_OUTRO          = "Outro"
LABEL_ADVERTISEMENT  = "Advertisement"

KIND_CONTENT     = "content"
KIND_NON_CONTENT = "non-content"

_KIND_FOR_LABEL: dict[str, str] = {
    LABEL_CORE_CONTENT:   KIND_CONTENT,
    LABEL_INTRO:          KIND_NON_CONTENT,
    LABEL_OUTRO:          KIND_NON_CONTENT,
    LABEL_ADVERTISEMENT:  KIND_NON_CONTENT,
}

OUTPUT_LABELS = {
    LABEL_CORE_CONTENT,
    LABEL_INTRO,
    LABEL_OUTRO,
    LABEL_ADVERTISEMENT,
}

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

# Practical ad duration bounds (seconds). Keep this broad enough for unseen
# videos; confidence is decided by multimodal signals, not by fixed ad count.
AD_MIN_SEC  = 28.0
AD_MAX_SEC  = 130.0

# Ignore only the first few seconds, where title cards and player warm-up
# frames can look ad-like. Do not assume a fixed number or spacing of ads.
EDGE_SUPPRESSION_SEC = 5.0

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
EDGE_WEIGHT       = 3.5
INTERIOR_WEIGHT   = 1.5
TEXT_WEIGHT       = 1.0
CONTENT_PENALTY_WEIGHT = 3.0
SEMANTIC_AD_THRESHOLD = 0.78
SEMANTIC_WEAK_AD_THRESHOLD = 0.70
SEMANTIC_WEAK_AD_MARGIN = 0.10
DIRECTION_WEIGHT = 2.0
DIRECTION_CONTEXT_SEC = 45.0
CANDIDATE_MIN_ADNESS = 0.10
AD_SELECTION_MIN_SCORE = 6.8
AD_SELECTION_TEXT_SCORE = 6.0
AD_SELECTION_VISUAL_SCORE = 4.8
HIGH_CONTENT_PENALTY_REJECT = 0.28
MEDIUM_CONTENT_PENALTY_REJECT = 0.18
AD_DUPLICATE_IOU = 0.20
AD_DUPLICATE_GAP_SEC = 30.0
MAX_AD_INTERVALS = 10
DURATION_WEIGHT   = 0.0

# Non-ad segmentation rules.  These only split content regions after the ad
# intervals have already been selected, so they do not move ad boundaries.
EDGE_TITLE_CARD_SEC = 55.0
MIN_EDGE_AUXILIARY_SEC = 4.0
OPENING_SEQUENCE_MAX_SEC = 90.0
ENDING_SEQUENCE_MIN_SEC = 18.0
AUXILIARY_MIN_SEC = 12.0
SHORT_CORE_CONTENT_MIN_SEC = 8.0
EDGE_AUXILIARY_KEEP_SEC = 8.0

# ---------------------------------------------------------------------------
# Ad-signal phrase/brand loading
# ---------------------------------------------------------------------------

def _load_ad_signals() -> tuple[list[str], list[str], dict[str, list[str]]]:
    fusion_dir = Path(__file__).parent
    signals_file = fusion_dir / "ad_signals.json"
    if not signals_file.is_file():
        return [], [], {}
    data = json.loads(signals_file.read_text(encoding="utf-8"))
    brand_names: list[str] = []
    extra_brand_names: list[str] = []
    seen: set[str] = set()

    def add_brand(raw_name: str) -> None:
        name = raw_name.strip().lower()
        if not name or name.startswith("#"):
            return
        if name not in seen:
            seen.add(name)
            brand_names.append(name)

    for category_brands in data.get("brands", {}).values():
        for name in category_brands:
            add_brand(name)

    # Optional large external brand dictionary. Keep this out of code so the
    # list can grow without changing the fusion logic. These extra names are
    # intentionally used with stricter context rules than the curated list.
    def add_extra_brand(raw_name: str) -> None:
        name = raw_name.strip().lower()
        if not name or name.startswith("#"):
            return
        if name not in seen:
            seen.add(name)
            extra_brand_names.append(name)

    extra_brand_file = fusion_dir / "extra_brand_names.txt"
    if os.getenv("FUSION_USE_EXTRA_BRANDS", "1") != "0" and extra_brand_file.is_file():
        for line in extra_brand_file.read_text(encoding="utf-8").splitlines():
            for name in line.split("#", 1)[0].split(","):
                add_extra_brand(name)

    return brand_names, extra_brand_names, data.get("phrases", {})


_AD_BRAND_NAMES, _EXTRA_AD_BRAND_NAMES, _AD_PHRASES = _load_ad_signals()
_SPONSORSHIP_PHRASES = _AD_PHRASES.get("sponsorship", [])
_OUTRO_PHRASES       = _AD_PHRASES.get("outro", [])
_TEXT_AD_REGEXES = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b\d{1,2}\s*%\s*off\b",
        r"\b(discount code|promo code|coupon code|limited time offer)\b",
        r"\b(sponsored by|paid for by|brought to you by)\b",
        r"\bisn[鈥?]?t\s+just\b",
        r"\b(can|could)\s+i\s+take\s+your\s+order\b",
        r"\b(to\s+drink|for\s+dessert)\b",
        r"\b(combo|deluxe|cheese\s+curds|curly\s+fries|orange\s+chicken|kung\s+pao)\b",
        r"\b(your\s+)?purchase\s+supports\b",
        r"\bkeep\s+(girls|kids|children|athletes)\s+playing\b",
        r"\b(shop|buy|order|download|get started|sign up)\s+(now|today)\b",
        r"\bfree\s+(trial|shipping|month|months)\b",
        r"\b[A-Z0-9._%+-]+\.com\b",
        r"\bwww\.",
        r"\bqr\s*code\b",
    )
]
_NOISY_BRAND_TERMS = {
    "windows",
    "max",
    "prime",
    "kind",
    "quest",
    "ring",
    "target",
    "teams",
    "zoom",
    "enterprise",
}
_AMBIGUOUS_SINGLE_BRAND_TERMS = {
    "apple",
    "ford",
    "gap",
    "max",
    "prime",
    "shell",
    "target",
    "windows",
}
_GENERIC_CONTENT_CONTEXT_PHRASES = [
    "lecture",
    "lesson",
    "discussion",
    "interview",
    "story",
    "episode",
    "chapter",
    "research",
    "study",
    "experiment",
    "evidence",
    "for example",
    "in this section",
    "let's look",
    "you can see",
    "this means",
    "the question is",
]
_COMMERCE_CONTEXT_WORDS = {
    "buy",
    "shop",
    "order",
    "download",
    "subscribe",
    "sale",
    "deal",
    "offer",
    "discount",
    "code",
    "free",
    "trial",
}
_CORPORATE_OCR_CONTEXT_WORDS = {
    "company",
    "corp",
    "corporation",
    "inc",
    "llc",
    "ltd",
    "copyright",
    "rights",
    "reserved",
}


def _compile_brand_regex(brand_names: list[str], *, include_ambiguous: bool) -> re.Pattern[str]:
    terms: list[str] = []
    for brand in brand_names:
        brand_l = brand.lower().strip()
        if brand_l in _NOISY_BRAND_TERMS:
            continue
        if not include_ambiguous and brand_l in _AMBIGUOUS_SINGLE_BRAND_TERMS:
            continue
        if len(re.sub(r"[^a-z0-9]", "", brand_l)) <= 1:
            continue
        terms.append(re.escape(brand_l))

    if not terms:
        return re.compile(r"a^")

    # One compiled scan is much faster than checking thousands of brands one by one.
    pattern = "|".join(sorted(set(terms), key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])", re.IGNORECASE)


_BRAND_REGEX_ALL = _compile_brand_regex(_AD_BRAND_NAMES, include_ambiguous=True)
_BRAND_REGEX_UNAMBIGUOUS = _compile_brand_regex(_AD_BRAND_NAMES, include_ambiguous=False)
_EXTRA_BRAND_REGEX = _compile_brand_regex(_EXTRA_AD_BRAND_NAMES, include_ambiguous=False)


def _phrase_in_text(phrase: str, text: str) -> bool:
    escaped = re.escape(phrase.lower())
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


def _brand_hit_count(text: str, *, include_ambiguous: bool = True) -> int:
    regex = _BRAND_REGEX_ALL if include_ambiguous else _BRAND_REGEX_UNAMBIGUOUS
    return len({match.group(0).lower() for match in regex.finditer(text)})


def _extra_brand_hit_count(text: str) -> int:
    return len({match.group(0).lower() for match in _EXTRA_BRAND_REGEX.finditer(text)})


def _ocr_text_has_ad_signal(text: str) -> bool:
    if any(pattern.search(text) for pattern in _TEXT_AD_REGEXES):
        return True
    if any(phrase in text for phrase in _SPONSORSHIP_PHRASES):
        return True

    words = re.findall(r"[a-z0-9']+", text)
    brand_hits = _brand_hit_count(text)
    unambiguous_brand_hits = _brand_hit_count(text, include_ambiguous=False)
    extra_brand_hits = _extra_brand_hit_count(text)
    commerce_context = any(word in words for word in _COMMERCE_CONTEXT_WORDS)
    corporate_context = (
        any(word in words for word in _CORPORATE_OCR_CONTEXT_WORDS)
        or "漏" in text
        or bool(re.search(r"\b20\d{2}\b", text))
    )

    if brand_hits >= 2:
        return True
    if unambiguous_brand_hits >= 1 and commerce_context:
        return True
    if unambiguous_brand_hits >= 1 and corporate_context and len(words) <= 9:
        return True
    if extra_brand_hits >= 2 and (commerce_context or corporate_context):
        return True
    if extra_brand_hits >= 1 and (commerce_context or corporate_context) and len(words) <= 8:
        return True
    # Short OCR text can be a brand card or logo. Long OCR text is often a
    # slide, caption, or scene text where a brand name is only topical.
    return unambiguous_brand_hits >= 1 and 1 <= len(words) <= 7

# ---------------------------------------------------------------------------
# Per-window audio helpers
# ---------------------------------------------------------------------------

def _audio_features(
    t0: float, t1: float, audio_windows: list[AudioWindow]
) -> tuple[float, float, str]:
    """Return (anomaly_score, energy_rms, audio_label) for the closest audio window."""
    mid = 0.5 * (t0 + t1)
    best_dist = float("inf")
    anomaly = 0.0
    energy  = 1.0
    label = "unknown"
    for aw in audio_windows:
        d = abs(0.5 * (aw.t0 + aw.t1) - mid)
        if d < best_dist:
            best_dist = d
            extra   = aw.model_extra or {}
            anomaly = float(extra.get("anomaly_score", 0.0))
            energy  = float(extra.get("energy_rms", 1.0))
            label = str(extra.get("audio_label", "unknown"))
    return anomaly, energy, label


def _audio_label_near(t_mid: float, audio_windows: list[AudioWindow]) -> str:
    best_dist = float("inf")
    best_label = "unknown"
    for aw in audio_windows:
        mid = 0.5 * (aw.t0 + aw.t1)
        dist = abs(mid - t_mid)
        if dist < best_dist:
            best_dist = dist
            best_label = str((aw.model_extra or {}).get("audio_label", "unknown"))
    return best_label


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


def _asr_speech_coverage(t0: float, t1: float, speech_spans: list[SpeechSpan]) -> float:
    dur = max(t1 - t0, 1e-6)
    covered = 0.0
    for span in speech_spans:
        if (span.model_extra or {}).get("source") in {"ocr", "semantic", "semantic_structure"}:
            continue
        ov_s = max(t0, span.t0)
        ov_e = min(t1, span.t1)
        if ov_e > ov_s and span.text:
            covered += ov_e - ov_s
    return min(1.0, covered / dur)


def _has_nearby_speech(
    t0: float, t1: float, speech_spans: list[SpeechSpan], context: float
) -> bool:
    lo, hi = t0 - context, t1 + context
    return any(
        s.t1 >= lo
        and s.t0 <= hi
        and s.text
        and (s.model_extra or {}).get("source") not in {"ocr", "semantic", "semantic_structure"}
        for s in speech_spans
    )


def _speech_text_ad_signal(
    t0: float, t1: float, speech_spans: list[SpeechSpan]
) -> float:
    lo, hi = t0 - 6.0, t1 + 6.0
    chunks: list[str] = []
    ocr_chunks: list[str] = []
    semantic_score = 0.0
    semantic_margin = 0.0
    for span in speech_spans:
        if span.t1 < lo or span.t0 > hi:
            continue
        extra = span.model_extra or {}
        source = extra.get("source")
        if source == "semantic" and float(span.t1) - float(span.t0) <= 75.0:
            semantic_score = max(semantic_score, float(extra.get("semantic_ad_score", 0.0)))
            semantic_margin = max(semantic_margin, float(extra.get("semantic_margin", 0.0)))
        if source in {"semantic", "semantic_structure"}:
            continue
        if span.text:
            text = span.text.lower()
            chunks.append(text)
            if source == "ocr":
                ocr_chunks.append(text)

    def semantic_ad_signal() -> float:
        if semantic_score >= SEMANTIC_AD_THRESHOLD:
            return semantic_score
        asr_cov = _asr_speech_coverage(t0, t1, speech_spans)
        if (
            semantic_score >= SEMANTIC_WEAK_AD_THRESHOLD
            and semantic_margin >= SEMANTIC_WEAK_AD_MARGIN
            and asr_cov >= 0.25
        ):
            return 0.35
        return 0.0

    if not chunks:
        return semantic_ad_signal()
    combined = " ".join(chunks)
    for phrase in _SPONSORSHIP_PHRASES:
        if phrase in combined:
            return 0.9
    if any(pattern.search(combined) for pattern in _TEXT_AD_REGEXES):
        return 0.75
    if any(_ocr_text_has_ad_signal(chunk) for chunk in ocr_chunks):
        return 0.55
    brand_hits = _brand_hit_count(combined)
    if brand_hits >= 2:
        return 0.35
    if brand_hits == 1:
        return 0.22
    return semantic_ad_signal()


def _content_text_penalty(
    t0: float,
    t1: float,
    speech_spans: list[SpeechSpan],
    *,
    margin_sec: float = 8.0,
) -> float:
    lo, hi = t0 - margin_sec, t1 + margin_sec
    chunks = [
        span.text.lower()
        for span in speech_spans
        if span.t1 >= lo
        and span.t0 <= hi
        and span.text
        and (span.model_extra or {}).get("source") not in {"ocr", "semantic", "semantic_structure"}
    ]
    combined = " ".join(chunks)
    if not combined:
        return 0.0
    words = re.findall(r"[a-z0-9']+", combined)
    speech_cov = _asr_speech_coverage(t0, t1, speech_spans)
    generic_context_hits = sum(
        1 for phrase in _GENERIC_CONTENT_CONTEXT_PHRASES if _phrase_in_text(phrase, combined)
    )
    has_explicit_ad_hint = (
        any(pattern.search(combined) for pattern in _TEXT_AD_REGEXES)
        or any(phrase in combined for phrase in _SPONSORSHIP_PHRASES)
        or _brand_hit_count(combined) >= 2
    )
    penalty = 0.0
    if len(words) >= 80 and speech_cov >= 0.75:
        penalty += 0.18
    elif len(words) >= 45 and speech_cov >= 0.65:
        penalty += 0.10
    if generic_context_hits >= 2:
        penalty += 0.20
    elif generic_context_hits == 1:
        penalty += 0.08
    if has_explicit_ad_hint:
        penalty *= 0.35
    return min(0.45, penalty)


def _compute_text_ad_scores(
    windows: list[VisualWindow],
    speech_spans: list[SpeechSpan],
) -> np.ndarray:
    return np.array(
        [_speech_text_ad_signal(window.t0, window.t1, speech_spans) for window in windows],
        dtype=np.float64,
    )


def _compute_content_text_penalties(
    windows: list[VisualWindow],
    speech_spans: list[SpeechSpan],
) -> np.ndarray:
    return np.array(
        [_content_text_penalty(window.t0, window.t1, speech_spans) for window in windows],
        dtype=np.float64,
    )

# ---------------------------------------------------------------------------
# Step 1 鈥?per-window foreignness score (interior signal)
# ---------------------------------------------------------------------------

def _compute_foreignness_scores(
    windows: list[VisualWindow],
    audio_windows: list[AudioWindow],
    speech_spans: list[SpeechSpan],
    duration: float,
) -> np.ndarray:
    """
    Per-window foreignness score in [0,1].
    Higher 鈫?more likely to be inside an advertisement.
    """
    N = len(windows)
    scores = np.zeros(N, dtype=np.float64)

    for i, w in enumerate(windows):
        t0, t1 = w.t0, w.t1
        mid = 0.5 * (t0 + t1)

        palette_score = float(w.palette_delta)

        anomaly, energy, audio_label = _audio_features(t0, t1, audio_windows)
        audio_score = float(anomaly)
        if energy < 0.015:
            audio_score = max(audio_score, 0.20)

        cov    = _asr_speech_coverage(t0, t1, speech_spans)
        nearby = _has_nearby_speech(t0, t1, speech_spans, SPEECH_CONTEXT_SEC)
        text_sig = _speech_text_ad_signal(t0, t1, speech_spans)
        content_penalty = _content_text_penalty(t0, t1, speech_spans)

        visual_ad_score = 0.0
        if w.visual_hypothesis == "graphics_heavy":
            visual_ad_score = max(visual_ad_score, 0.60)
        if w.high_text_density:
            visual_ad_score = max(visual_ad_score, 0.45)
        if w.visual_hypothesis == "static" and w.high_text_density:
            visual_ad_score = max(visual_ad_score, 0.55)
        if w.palette_delta > 0.35:
            visual_ad_score = max(visual_ad_score, 0.40)

        has_visual_ad_cue = (
            visual_ad_score >= 0.50
            or w.visual_hypothesis == "graphics_heavy"
            or w.high_text_density
        )
        has_audio_ad_cue = (
            audio_label in {"music", "mixed"}
            and (energy >= 0.08 or audio_score >= 0.45)
        )

        # No speech is only useful when supported by another modality. Static
        # real content can also have little speech, so avoid treating silence
        # alone as an ad signal.
        nospeech_score = 0.0
        if not nearby:
            nospeech_score = 0.35
        elif cov < 0.05:
            nospeech_score = 0.25

        if text_sig > 0:
            audio_score = max(audio_score, text_sig)
            nospeech_score = max(nospeech_score, 0.70)
        elif cov < 0.08 and has_visual_ad_cue and has_audio_ad_cue:
            nospeech_score = max(nospeech_score, 0.65)
        elif cov < 0.08 and (has_visual_ad_cue or has_audio_ad_cue):
            nospeech_score = max(nospeech_score, 0.45)

        if cov < 0.12 and audio_label in {"music", "mixed"}:
            audio_score = max(audio_score, 0.65)
        elif cov < 0.05 and audio_label == "silence" and (w.high_text_density or w.visual_hypothesis == "graphics_heavy"):
            audio_score = max(audio_score, 0.55)

        # Suppress only the very start/end of video where player warm-up,
        # black frames, or closing frames can look ad-like.
        if mid < EDGE_SUPPRESSION_SEC or mid > duration - 20.0:
            palette_score  *= 0.1
            audio_score    *= 0.1
            nospeech_score *= 0.1
            visual_ad_score *= 0.1

        visual_component = max(palette_score, visual_ad_score)
        visual_weight = W_PALETTE
        audio_weight = W_AUDIO
        nospeech_weight = W_NOSPEECH

        quiet_static_content_like = (
            cov < 0.05
            and audio_label == "silence"
            and not w.high_text_density
            and w.visual_hypothesis == "static"
            and w.motion_score < 0.08
            and w.palette_delta < 0.20
        )

        # If there is little/no speech, do not penalize the window just because
        # ASR has no text.  No-dialog ads are often driven by music, graphics,
        # product text, and hard visual/audio changes.
        if cov < 0.08 and (visual_component >= 0.35 or audio_score >= 0.55):
            visual_weight = 0.48
            audio_weight = 0.42
            nospeech_weight = 0.10
            visual_component = min(1.0, visual_component * 1.20)
            audio_score = min(1.0, audio_score * 1.15)
        elif cov < 0.08:
            nospeech_weight = 0.08

        if quiet_static_content_like and text_sig < 0.30:
            visual_component *= 0.45
            audio_score *= 0.30
            nospeech_score *= 0.20

        # Strong content-domain terms should reduce false positives unless
        # explicit ad text is also present.
        if content_penalty > 0.0 and text_sig < 0.75:
            visual_component *= 1.0 - 0.35 * content_penalty
            audio_score *= 1.0 - 0.45 * content_penalty
            nospeech_score *= 1.0 - 0.25 * content_penalty

        scores[i] = (
            visual_weight * visual_component
            + audio_weight * audio_score
            + nospeech_weight * nospeech_score
        )

    return scores


# ---------------------------------------------------------------------------
# Step 2 鈥?per-boundary edge score
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

        # Visual: combine palette change with structural frame changes. Some
        # inserted ads keep a similar palette but switch into text-heavy or
        # graphic-heavy frames, so palette alone misses those boundaries.
        prev = windows[i - 1]
        cur = windows[i]
        edge_density_delta = abs(float(cur.edge_density) - float(prev.edge_density))
        luminance_delta = abs(float(cur.luminance_mean) - float(prev.luminance_mean))
        motion_delta = abs(float(cur.motion_score) - float(prev.motion_score))
        text_density_transition = 1.0 if cur.high_text_density != prev.high_text_density else 0.0
        vis = max(
            float(cur.palette_delta),
            0.70 * edge_density_delta,
            0.45 * luminance_delta,
            0.35 * motion_delta,
            0.50 * text_density_transition,
        )

        # Audio: change in anomaly across boundary
        aud_delta = _audio_delta(t_boundary, audio_windows, half_sec=3.0)
        before_audio_label = _audio_label_near(t_boundary - 1.0, audio_windows)
        after_audio_label = _audio_label_near(t_boundary + 1.0, audio_windows)
        audio_label_transition = 1.0 if before_audio_label != after_audio_label else 0.0

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
        if mid < EDGE_SUPPRESSION_SEC or mid > duration - 20.0:
            vis             *= 0.1
            aud_delta       *= 0.1
            audio_label_transition *= 0.1
            speech_transition *= 0.1

        if not had_speech_before and not has_speech_after:
            edge[i] = 0.60 * vis + 0.30 * aud_delta + 0.10 * audio_label_transition
        else:
            edge[i] = 0.45 * vis + 0.30 * aud_delta + 0.15 * speech_transition + 0.10 * audio_label_transition

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
# Step 3 - candidate ad interval scoring
# ---------------------------------------------------------------------------

def _interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if overlap <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return overlap / max(union, 1)


def _interval_gap(a: tuple[int, int], b: tuple[int, int], windows: list[VisualWindow]) -> float:
    if a[1] <= b[0]:
        return max(0.0, windows[b[0]].t0 - windows[a[1] - 1].t1)
    if b[1] <= a[0]:
        return max(0.0, windows[a[0]].t0 - windows[b[1] - 1].t1)
    return 0.0


def _text_anchor_times(speech_spans: list[SpeechSpan]) -> list[float]:
    anchors: list[float] = []

    for span in speech_spans:
        extra = span.model_extra or {}
        text = span.text.lower().strip()
        if not text:
            continue
        source = extra.get("source")
        is_ocr_ad_text = source == "ocr" and _ocr_text_has_ad_signal(text)
        if is_ocr_ad_text:
            anchors.append(0.5 * (float(span.t0) + float(span.t1)))
    return sorted(set(round(anchor, 3) for anchor in anchors))


def _ad_anchor_times_in_range(
    t0: float,
    t1: float,
    speech_spans: list[SpeechSpan],
) -> list[float]:
    anchors = [
        anchor
        for anchor in _text_anchor_times(speech_spans)
        if t0 - 3.0 <= anchor <= t1 + 3.0
    ]

    for span in speech_spans:
        extra = span.model_extra or {}
        if extra.get("source") != "semantic":
            continue
        if float(span.t1) - float(span.t0) > 75.0:
            continue
        if float(extra.get("semantic_ad_score", 0.0)) < SEMANTIC_WEAK_AD_THRESHOLD:
            continue
        if float(extra.get("semantic_margin", 0.0)) < SEMANTIC_WEAK_AD_MARGIN:
            continue
        if span.t1 >= t0 - 3.0 and span.t0 <= t1 + 3.0:
            anchors.append(0.5 * (float(span.t0) + float(span.t1)))
    return sorted(set(round(anchor, 3) for anchor in anchors))


def _has_text_anchor_in_range(t0: float, t1: float, speech_spans: list[SpeechSpan]) -> bool:
    return bool(_ad_anchor_times_in_range(t0, t1, speech_spans))


def _meaningful_asr_text(text: str) -> bool:
    words = re.findall(r"[a-z0-9']+", text.lower())
    if not words:
        return False
    normalized = " ".join(words)
    if normalized in {"music", "musics", "silence"}:
        return False
    if len(words) == 1 and words[0] in {"music", "silence"}:
        return False
    return True


def _meaningful_asr_bounds(
    t0: float,
    t1: float,
    speech_spans: list[SpeechSpan],
    *,
    min_chars: int = 35,
    min_spans: int = 3,
) -> tuple[float | None, float | None]:
    selected: list[SpeechSpan] = []
    for span in speech_spans:
        if span.t1 < t0 or span.t0 > t1 or not span.text:
            continue
        if _span_source(span) in {"ocr", "semantic", "semantic_structure"}:
            continue
        if _meaningful_asr_text(span.text):
            selected.append(span)
    if not selected:
        return None, None
    combined = " ".join(span.text for span in selected)
    if len(selected) < min_spans and len(combined) < min_chars:
        return None, None
    return min(float(span.t0) for span in selected), max(float(span.t1) for span in selected)


def _semantic_ad_search_range(
    t0: float,
    t1: float,
    speech_spans: list[SpeechSpan],
) -> tuple[float, float] | None:
    starts: list[float] = []
    ends: list[float] = []
    for span in speech_spans:
        extra = span.model_extra or {}
        if extra.get("source") != "semantic":
            continue
        score = float(extra.get("semantic_ad_score", 0.0))
        margin = float(extra.get("semantic_margin", 0.0))
        if score < SEMANTIC_WEAK_AD_THRESHOLD or margin < SEMANTIC_WEAK_AD_MARGIN:
            continue
        if span.t1 < t0 or span.t0 > t1:
            continue
        starts.append(float(span.t0))
        ends.append(float(span.t1))
    if not starts:
        return None
    return min(min(starts), t0), max(max(ends), t1)


def _find_text_anchor_intervals(
    windows: list[VisualWindow],
    speech_spans: list[SpeechSpan],
    edge_scores: np.ndarray,
    foreign_scores: np.ndarray,
    text_scores: np.ndarray,
    content_penalties: np.ndarray,
) -> list[tuple[int, int]]:
    anchors = _text_anchor_times(speech_spans)
    if not anchors or not windows:
        return []

    N = len(windows)
    norm_edge = np.append(edge_scores / (float(edge_scores.max()) + 1e-9), 0.0)
    norm_foreign = foreign_scores / (float(foreign_scores.max()) + 1e-9)
    norm_text = text_scores / (float(text_scores.max()) + 1e-9) if text_scores.max() > 0 else text_scores
    norm_penalty = (
        content_penalties / (float(content_penalties.max()) + 1e-9)
        if content_penalties.max() > 0
        else content_penalties
    )

    def index_at_time(t: float) -> int:
        for i, window in enumerate(windows):
            if window.t0 <= t < window.t1:
                return i
        return max(0, min(N - 1, N - 1))

    common_durations = (30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0)
    anchor_positions = (0.25, 0.50, 0.75, 0.90, 0.95, 1.00)
    intervals: list[tuple[float, int, int]] = []

    for anchor in anchors:
        if anchor < EDGE_SUPPRESSION_SEC or anchor > windows[-1].t1 - 20.0:
            continue
        best: tuple[float, int, int] | None = None
        for duration in common_durations:
            for frac in anchor_positions:
                start_time = max(0.0, anchor - duration * frac)
                end_time = start_time + duration
                # OCR often lands on product/logo text near the final frames of
                # an ad.  Allow the text anchor to sit on the end boundary
                # instead of forcing it toward the middle of the candidate.
                if end_time < anchor - 4.0 or start_time >= anchor:
                    continue
                if end_time > windows[-1].t1:
                    continue
                s = index_at_time(start_time)
                e = min(N, index_at_time(end_time) + 1)
                if e <= s:
                    continue
                actual_duration = windows[e - 1].t1 - windows[s].t0
                if actual_duration < AD_MIN_SEC or actual_duration > AD_MAX_SEC:
                    continue
                penalty_mean = float(norm_penalty[s:e].mean())
                if penalty_mean >= HIGH_CONTENT_PENALTY_REJECT:
                    continue
                text_mean = float(norm_text[s:e].mean())
                foreign_mean = float(norm_foreign[s:e].mean())
                pre_context_w = max(4, int(15.0 / max(windows[0].t1 - windows[0].t0, 1e-6)))
                pre_text_mean = float(norm_text[max(0, s - pre_context_w):s].mean()) if s > 0 else 0.0
                pre_foreign_mean = float(norm_foreign[max(0, s - pre_context_w):s].mean()) if s > 0 else 0.0
                duration_prior = max(
                    np.exp(-abs(actual_duration - target) / max(12.0, target * 0.25))
                    for target in common_durations
                )
                end_anchor_prior = 0.0
                if frac >= 0.85:
                    # OCR often fires on a logo/product card near the end of
                    # an inserted ad. In that case, use the anchor as an end
                    # cue and look backward to a plausible hard boundary.
                    end_anchor_prior += 1.00
                    if actual_duration >= AD_MIN_SEC + 8.0:
                        end_anchor_prior += 0.55
                if frac >= 0.95:
                    end_anchor_prior += 0.35
                if frac >= 0.85 and pre_text_mean >= 0.12 and pre_foreign_mean >= 0.30:
                    # If the region before this candidate already looks like
                    # ad text/graphics, this OCR hit is likely inside the ad,
                    # not a reliable end marker.
                    end_anchor_prior -= 1.25
                short_slice_penalty = (
                    0.45
                    if actual_duration <= AD_MIN_SEC + 4.0 and frac < 0.85
                    else 0.0
                )
                score = (
                    3.2 * (float(norm_edge[s]) + float(norm_edge[e]))
                    + 0.8 * foreign_mean
                    + 0.6 * text_mean
                    + 1.05 * float(duration_prior)
                    + end_anchor_prior
                    - 2.2 * penalty_mean
                    - short_slice_penalty
                )
                if best is None or score > best[0]:
                    best = (float(score), s, e)
        if best is not None and best[0] >= 3.4:
            intervals.append(best)

    return sorted((s, e) for _, s, e in intervals)


def _interval_return_to_content_score(
    start: int,
    end: int,
    windows: list[VisualWindow],
    foreign_scores: np.ndarray,
    text_scores: np.ndarray,
    content_penalties: np.ndarray,
    *,
    context_sec: float = 45.0,
) -> float:
    if not windows or end <= start:
        return 0.0
    window_sec = windows[0].t1 - windows[0].t0 if windows else 1.0
    ctx = max(8, int(context_sec / max(window_sec, 1e-6)))

    def build_matrix() -> np.ndarray:
        norm_foreign = foreign_scores / (float(foreign_scores.max()) + 1e-9)
        norm_text = text_scores / (float(text_scores.max()) + 1e-9) if text_scores.max() > 0 else text_scores
        norm_penalty = (
            content_penalties / (float(content_penalties.max()) + 1e-9)
            if content_penalties.max() > 0
            else content_penalties
        )
        rows: list[list[float]] = []
        for i, window in enumerate(windows):
            rows.append(
                [
                    float(window.motion_score),
                    float(window.luminance_mean),
                    float(window.edge_density),
                    float(window.palette_delta),
                    1.0 if window.high_text_density else 0.0,
                    float(norm_foreign[i]),
                    float(norm_text[i]),
                    float(norm_penalty[i]),
                ]
            )
        return np.asarray(rows, dtype=np.float64)

    features = build_matrix()

    def mean_vec(lo: int, hi: int) -> np.ndarray | None:
        lo = max(0, min(len(features), lo))
        hi = max(0, min(len(features), hi))
        if hi <= lo:
            return None
        return features[lo:hi].mean(axis=0)

    before = mean_vec(start - ctx, start)
    inside = mean_vec(start, end)
    after = mean_vec(end, end + ctx)
    if before is None or inside is None or after is None:
        return 0.0

    before_inside = float(np.linalg.norm(before - inside))
    after_inside = float(np.linalg.norm(after - inside))
    before_after = float(np.linalg.norm(before - after))
    return max(0.0, 0.5 * (before_inside + after_inside) - before_after)


def _find_ad_intervals(
    edge_scores: np.ndarray,
    foreign_scores: np.ndarray,
    windows: list[VisualWindow],
    text_scores: np.ndarray | None = None,
    content_penalties: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """
    Find a variable number of high-confidence ad intervals.

    Candidates are scored independently, then duplicate/overlapping intervals
    are suppressed so only the strongest interval around each ad remains.
    """
    N = len(windows)
    if N == 0:
        return []

    norm_edge = np.append(edge_scores / (float(edge_scores.max()) + 1e-9), 0.0)
    norm_foreign = foreign_scores / (float(foreign_scores.max()) + 1e-9)
    if text_scores is None:
        norm_text = np.zeros(N, dtype=np.float64)
    else:
        norm_text = text_scores / (float(text_scores.max()) + 1e-9)
    if content_penalties is None:
        norm_penalty = np.zeros(N, dtype=np.float64)
    else:
        norm_penalty = content_penalties / (float(content_penalties.max()) + 1e-9)

    cum_foreign = np.concatenate([[0.0], np.cumsum(norm_foreign)])
    cum_text = np.concatenate([[0.0], np.cumsum(norm_text)])
    cum_penalty = np.concatenate([[0.0], np.cumsum(norm_penalty)])
    adness = np.clip(0.65 * norm_foreign + 0.80 * norm_text - 0.90 * norm_penalty, 0.0, 1.5)
    cum_adness = np.concatenate([[0.0], np.cumsum(adness)])

    window_sec = windows[0].t1 - windows[0].t0 if windows else 2.0
    min_w = max(1, int(AD_MIN_SEC / window_sec))
    max_w = max(min_w + 1, int(AD_MAX_SEC / window_sec) + 1)
    direction_context_w = max(2, int(DIRECTION_CONTEXT_SEC / max(window_sec, 1e-6)))
    return_context_w = max(8, int(45.0 / max(window_sec, 1e-6)))

    feature_rows = np.asarray(
        [
            [
                float(window.motion_score),
                float(window.luminance_mean),
                float(window.edge_density),
                float(window.palette_delta),
                1.0 if window.high_text_density else 0.0,
                float(norm_foreign[i]),
                float(norm_text[i]),
                float(norm_penalty[i]),
            ]
            for i, window in enumerate(windows)
        ],
        dtype=np.float64,
    )
    feature_prefix = np.vstack([np.zeros((1, feature_rows.shape[1])), np.cumsum(feature_rows, axis=0)])

    def mean_from_prefix(prefix: np.ndarray, lo: int, hi: int) -> float:
        lo = max(0, min(N, lo))
        hi = max(0, min(N, hi))
        if hi <= lo:
            return 0.0
        return float((prefix[hi] - prefix[lo]) / (hi - lo))

    def direction_score(s: int, e: int) -> float:
        start_inside = mean_from_prefix(cum_adness, s, min(e, s + direction_context_w))
        start_before = mean_from_prefix(cum_adness, s - direction_context_w, s)
        end_inside = mean_from_prefix(cum_adness, max(s, e - direction_context_w), e)
        end_after = mean_from_prefix(cum_adness, e, e + direction_context_w)
        return 0.5 * ((start_inside - start_before) + (end_inside - end_after))

    def mean_feature(lo: int, hi: int) -> np.ndarray | None:
        lo = max(0, min(N, lo))
        hi = max(0, min(N, hi))
        if hi <= lo:
            return None
        return (feature_prefix[hi] - feature_prefix[lo]) / (hi - lo)

    def return_to_content_score(s: int, e: int) -> float:
        before = mean_feature(s - return_context_w, s)
        inside = mean_feature(s, e)
        after = mean_feature(e, e + return_context_w)
        if before is None or inside is None or after is None:
            return 0.0
        before_inside = float(np.linalg.norm(before - inside))
        after_inside = float(np.linalg.norm(after - inside))
        before_after = float(np.linalg.norm(before - after))
        return max(0.0, 0.5 * (before_inside + after_inside) - before_after)

    def candidate_score(s: int, e: int) -> tuple[float, float, float, float, float, float, float]:
        width = max(e - s, 1)
        interior_mean = (cum_foreign[e] - cum_foreign[s]) / width
        text_mean = (cum_text[e] - cum_text[s]) / width
        penalty_mean = (cum_penalty[e] - cum_penalty[s]) / width
        adness_mean = (cum_adness[e] - cum_adness[s]) / width
        if adness_mean < CANDIDATE_MIN_ADNESS:
            return float("-inf"), adness_mean, text_mean, 0.0, penalty_mean, 0.0, 0.0
        boundary_mean = 0.5 * (norm_edge[s] + norm_edge[e])
        text_peak = float(norm_text[s:e].max()) if e > s else 0.0
        direction_signal = direction_score(s, e)
        score = (
            EDGE_WEIGHT * (norm_edge[s] + norm_edge[e])
            + INTERIOR_WEIGHT * interior_mean
            + TEXT_WEIGHT * text_mean
            + DIRECTION_WEIGHT * direction_signal
            - CONTENT_PENALTY_WEIGHT * penalty_mean
        )
        return (
            float(score),
            float(adness_mean),
            float(text_mean),
            float(boundary_mean),
            float(penalty_mean),
            text_peak,
            float(direction_signal),
        )

    candidates: list[tuple[float, int, int]] = []
    for e in range(min_w, N + 1):
        s_lo = max(0, e - max_w)
        s_hi = e - min_w
        for s in range(s_lo, s_hi + 1):
            if windows[s].t0 < EDGE_SUPPRESSION_SEC:
                continue
            duration_sec = windows[e - 1].t1 - windows[s].t0
            if duration_sec < AD_MIN_SEC or duration_sec > AD_MAX_SEC:
                continue
            (
                score,
                adness_mean,
                text_mean,
                boundary_mean,
                penalty_mean,
                text_peak,
                direction_signal,
            ) = candidate_score(s, e)
            if not np.isfinite(score):
                continue
            return_signal = return_to_content_score(s, e)
            # A single OCR/semantic spike inside ordinary content should not
            # override strong content-domain evidence.  Real ads with text tend
            # to keep ad text active across multiple windows; weak text peaks
            # are mostly false positives from scene text, names, or dialogue.
            if penalty_mean >= HIGH_CONTENT_PENALTY_REJECT and not (
                text_mean >= 0.45 and score >= AD_SELECTION_TEXT_SCORE + 0.8
            ):
                continue
            if (
                penalty_mean >= MEDIUM_CONTENT_PENALTY_REJECT
                and text_mean < 0.18
                and adness_mean < 0.45
            ):
                continue

            strong_text_candidate = (
                penalty_mean < HIGH_CONTENT_PENALTY_REJECT
                and text_mean >= 0.32
                and score >= AD_SELECTION_TEXT_SCORE
            ) or (
                penalty_mean < MEDIUM_CONTENT_PENALTY_REJECT
                and
                text_peak >= 0.60
                and text_mean >= 0.025
                and score >= AD_SELECTION_TEXT_SCORE - 0.8
            )
            strong_visual_candidate = (
                text_mean < 0.20
                and adness_mean >= 0.18
                and boundary_mean >= 0.30
                and penalty_mean < MEDIUM_CONTENT_PENALTY_REJECT
                and score >= AD_SELECTION_VISUAL_SCORE
            )
            strong_return_candidate = (
                text_peak < 0.30
                and return_signal >= 0.70
                and duration_sec <= 90.0
                and boundary_mean >= 0.20
                and penalty_mean < MEDIUM_CONTENT_PENALTY_REJECT
            )
            strong_general_candidate = (
                score >= AD_SELECTION_MIN_SCORE
                and penalty_mean < HIGH_CONTENT_PENALTY_REJECT
                and (
                    text_mean >= 0.12
                    or adness_mean >= 0.28
                    or boundary_mean >= 0.42
                )
            )
            if strong_general_candidate or strong_text_candidate or strong_visual_candidate or strong_return_candidate:
                rank_score = score + (2.0 * return_signal if strong_return_candidate else 0.0)
                candidates.append((rank_score, s, e))

    selected: list[tuple[float, int, int]] = []
    for score, s, e in sorted(candidates, key=lambda item: item[0], reverse=True):
        current = (s, e)
        duplicate = False
        for _, selected_s, selected_e in selected:
            selected_interval = (selected_s, selected_e)
            if _interval_iou(current, selected_interval) > AD_DUPLICATE_IOU:
                duplicate = True
                break
            if _interval_gap(current, selected_interval, windows) < AD_DUPLICATE_GAP_SEC:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append((score, s, e))
        if len(selected) >= MAX_AD_INTERVALS:
            break

    return sorted((s, e) for _, s, e in selected)


# ---------------------------------------------------------------------------
# Step 4 鈥?Refine boundaries using local edge maxima
# ---------------------------------------------------------------------------

def _refine_boundary(
    idx: int,
    edge_scores: np.ndarray,
    direction: str,  # "start" or "end"
    windows: list[VisualWindow],
    search_sec: float = 15.0,
) -> int:
    """
    Given a coarse window index, search within 卤search_sec for the window
    with the highest edge score and return that index as the refined boundary.

    For "start": the boundary is the cut INTO the ad 鈫?look for the highest
                 edge score just at/after the coarse start.
    For "end":   the boundary is the cut OUT of the ad 鈫?look for the highest
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


def _window_supports_ad_expansion(
    index: int,
    windows: list[VisualWindow],
    foreign_scores: np.ndarray,
    text_scores: np.ndarray,
    content_penalties: np.ndarray,
    *,
    aggressive_visual: bool = False,
) -> bool:
    window = windows[index]
    foreign = float(foreign_scores[index])
    text = float(text_scores[index])
    penalty = float(content_penalties[index])
    visual_ad_like = (
        window.high_text_density
        or window.visual_hypothesis == "graphics_heavy"
        or window.palette_delta > 0.35
    )
    if penalty >= 0.90 and text < 0.60:
        return False
    if aggressive_visual and visual_ad_like and penalty < 0.80:
        return text >= 0.20 or foreign >= 0.22 or window.palette_delta > 0.45
    return text >= 0.30 or foreign >= 0.42 or (visual_ad_like and foreign >= 0.28)


def _expand_ad_interval(
    start: int,
    end: int,
    windows: list[VisualWindow],
    foreign_scores: np.ndarray,
    text_scores: np.ndarray,
    content_penalties: np.ndarray,
    *,
    search_sec: float = 35.0,
    aggressive_visual: bool = False,
) -> tuple[int, int]:
    window_sec = windows[0].t1 - windows[0].t0 if windows else 2.0
    max_expand_w = max(1, int(search_sec / max(window_sec, 1e-6)))
    rs = start
    while rs > 0 and start - rs < max_expand_w:
        if not _window_supports_ad_expansion(
            rs - 1,
            windows,
            foreign_scores,
            text_scores,
            content_penalties,
            aggressive_visual=aggressive_visual,
        ):
            break
        rs -= 1
    re = end
    while re < len(windows) and re - end < max_expand_w:
        if not _window_supports_ad_expansion(
            re,
            windows,
            foreign_scores,
            text_scores,
            content_penalties,
            aggressive_visual=aggressive_visual,
        ):
            break
        re += 1
    return rs, re


def _interval_support_score(
    interval: tuple[int, int],
    windows: list[VisualWindow],
    foreign_scores: np.ndarray,
    text_scores: np.ndarray,
    content_penalties: np.ndarray,
) -> float:
    start, end = interval
    if end <= start:
        return float("-inf")
    text_mean = float(text_scores[start:end].mean())
    duration_bonus = 0.0
    if text_mean >= 0.25 and windows:
        duration = windows[end - 1].t1 - windows[start].t0
        duration_bonus = 0.0015 * min(duration, AD_MAX_SEC)
    return float(
        foreign_scores[start:end].mean()
        + 1.5 * text_mean
        + duration_bonus
        - 1.2 * content_penalties[start:end].mean()
    )


def _interval_selection_score(
    interval: tuple[int, int],
    windows: list[VisualWindow],
    foreign_scores: np.ndarray,
    text_scores: np.ndarray,
    content_penalties: np.ndarray,
    edge_scores: np.ndarray | None = None,
    speech_spans: list[SpeechSpan] | None = None,
) -> float:
    start, end = interval
    score = _interval_support_score(
        interval,
        windows,
        foreign_scores,
        text_scores,
        content_penalties,
    )
    if not np.isfinite(score) or end <= start or not windows:
        return score

    duration = windows[end - 1].t1 - windows[start].t0
    score += 0.20 * _interval_return_to_content_score(
        start,
        end,
        windows,
        foreign_scores,
        text_scores,
        content_penalties,
    )
    score += 0.15 * _duration_prior(duration)

    if edge_scores is not None:
        start_edge = float(edge_scores[start]) if 0 <= start < len(edge_scores) else 0.0
        end_edge = float(edge_scores[end]) if 0 <= end < len(edge_scores) else 0.0
        score += 0.25 * (0.5 * (start_edge + end_edge))

    if speech_spans is not None:
        start_time = windows[start].t0
        end_time = windows[end - 1].t1
        anchors = [
            anchor
            for anchor in _text_anchor_times(speech_spans)
            if start_time - 3.0 <= anchor <= end_time + 3.0
        ]
        if anchors:
            anchor_pos = max(
                min(1.0, max(0.0, (anchor - start_time) / max(duration, 1e-6)))
                for anchor in anchors
            )
            score += 0.25
            if anchor_pos >= 0.85:
                score += 0.60
                if duration >= AD_MIN_SEC + 8.0:
                    score += 0.60
                window_sec = windows[0].t1 - windows[0].t0
                pre_context_w = max(4, int(15.0 / max(window_sec, 1e-6)))
                pre_start = max(0, start - pre_context_w)
                if start > pre_start:
                    pre_text_mean = float(text_scores[pre_start:start].mean())
                    pre_foreign_mean = float(foreign_scores[pre_start:start].mean())
                    if pre_text_mean >= 0.12 and pre_foreign_mean >= 0.30:
                        score -= 1.25
            elif duration <= AD_MIN_SEC + 3.0:
                score -= 0.25

    return float(score)


def _suppress_close_ad_intervals(
    intervals: list[tuple[int, int]],
    windows: list[VisualWindow],
    foreign_scores: np.ndarray,
    text_scores: np.ndarray,
    content_penalties: np.ndarray,
    edge_scores: np.ndarray | None = None,
    speech_spans: list[SpeechSpan] | None = None,
) -> list[tuple[int, int]]:
    kept: list[tuple[int, int]] = []
    for interval in sorted(intervals):
        if not kept:
            kept.append(interval)
            continue
        previous = kept[-1]
        too_close = (
            _interval_iou(previous, interval) > AD_DUPLICATE_IOU
            or _interval_gap(previous, interval, windows) < AD_DUPLICATE_GAP_SEC
        )
        if not too_close:
            kept.append(interval)
            continue
        interval_support = _interval_support_score(
            interval,
            windows,
            foreign_scores,
            text_scores,
            content_penalties,
        )
        previous_support = _interval_support_score(
            previous,
            windows,
            foreign_scores,
            text_scores,
            content_penalties,
        )
        interval_return = _interval_return_to_content_score(
            interval[0],
            interval[1],
            windows,
            foreign_scores,
            text_scores,
            content_penalties,
        )
        previous_return = _interval_return_to_content_score(
            previous[0],
            previous[1],
            windows,
            foreign_scores,
            text_scores,
            content_penalties,
        )
        # Do not let a broad OCR-anchor guess replace a much stronger
        # visual/audio interval unless the broader interval has its own
        # return-to-content evidence. This avoids turning a mid-ad text hit
        # into an over-extended ad.
        if interval_support < 0.55 * previous_support and interval_return < 0.25:
            continue
        if previous_support < 0.55 * interval_support and previous_return < 0.25:
            kept[-1] = interval
            continue
        interval_duration = windows[interval[1] - 1].t1 - windows[interval[0]].t0
        previous_duration = windows[previous[1] - 1].t1 - windows[previous[0]].t0
        if (
            interval_duration >= previous_duration
            and interval_support >= 0.98 * previous_support
            and interval_return >= previous_return + 0.10
        ):
            kept[-1] = interval
            continue

        if _interval_selection_score(
            interval,
            windows,
            foreign_scores,
            text_scores,
            content_penalties,
            edge_scores,
            speech_spans,
        ) > _interval_selection_score(
            previous,
            windows,
            foreign_scores,
            text_scores,
            content_penalties,
            edge_scores,
            speech_spans,
        ):
            kept[-1] = interval
    return kept


def _trim_confirmed_ad_tails(
    intervals: list[tuple[int, int]],
    windows: list[VisualWindow],
    content_penalties: np.ndarray,
    text_scores: np.ndarray,
) -> list[tuple[int, int]]:
    if not windows:
        return intervals
    window_sec = windows[0].t1 - windows[0].t0
    min_w = max(1, int(AD_MIN_SEC / max(window_sec, 1e-6)))
    trimmed: list[tuple[int, int]] = []
    for start, end in intervals:
        re = end
        while (
            re - start > min_w
            and re > start
            and content_penalties[re - 1] >= 0.30
            and text_scores[re - 1] < 0.60
        ):
            re -= 1
        trimmed.append((start, re))
    return trimmed


def _snap_text_supported_ad_boundaries(
    interval: tuple[int, int],
    windows: list[VisualWindow],
    speech_spans: list[SpeechSpan],
) -> tuple[int, int]:
    start, end = interval
    if not windows or end <= start:
        return interval

    interval_t0 = windows[start].t0
    interval_t1 = windows[end - 1].t1
    semantic_range = _semantic_ad_search_range(interval_t0, interval_t1, speech_spans)
    if semantic_range is not None:
        search_t0 = max(0.0, semantic_range[0] - 3.0)
        search_t1 = min(windows[-1].t1, semantic_range[1] + 3.0)
    else:
        search_t0 = max(0.0, interval_t0 - 3.0)
        search_t1 = min(windows[-1].t1, interval_t1 + 3.0)

    asr_start, asr_end = _meaningful_asr_bounds(search_t0, search_t1, speech_spans)
    if asr_start is None or asr_end is None:
        return interval

    target_start_time = max(search_t0, asr_start)
    target_end_time = min(windows[-1].t1, asr_end + 2.0)
    if target_end_time <= target_start_time:
        return interval

    def index_for_start(t: float) -> int:
        for index, window in enumerate(windows):
            if window.t0 <= t < window.t1:
                return index
        return max(0, min(len(windows) - 1, start))

    def index_for_end(t: float) -> int:
        for index, window in enumerate(windows):
            if window.t0 < t <= window.t1:
                return index + 1
        return max(start + 1, min(len(windows), end))

    snapped_start = index_for_start(target_start_time)
    snapped_end = index_for_end(target_end_time)
    window_sec = windows[0].t1 - windows[0].t0 if windows else 1.0
    min_w = max(1, int(AD_MIN_SEC / max(window_sec, 1e-6)))
    if snapped_end - snapped_start < min_w:
        snapped_end = min(len(windows), snapped_start + min_w)
    return snapped_start, snapped_end


def _trim_text_anchor_tails(
    intervals: list[tuple[int, int]],
    windows: list[VisualWindow],
    speech_spans: list[SpeechSpan],
) -> list[tuple[int, int]]:
    if not windows:
        return intervals
    window_sec = windows[0].t1 - windows[0].t0
    min_w = max(1, int(AD_MIN_SEC / max(window_sec, 1e-6)))
    anchors = _text_anchor_times(speech_spans)
    if not anchors:
        return intervals

    trimmed: list[tuple[int, int]] = []
    for start, end in intervals:
        start_time = windows[start].t0
        end_time = windows[end - 1].t1
        anchors_in_interval = [
            anchor
            for anchor in anchors
            if start_time - 3.0 <= anchor <= end_time + 3.0
        ]
        if not anchors_in_interval:
            trimmed.append((start, end))
            continue

        last_anchor = max(anchors_in_interval)
        if end_time - last_anchor <= 8.0:
            trimmed.append((start, end))
            continue

        target_end_time = min(end_time, last_anchor + 4.0)
        new_end = end
        for index in range(start + min_w, end + 1):
            if windows[index - 1].t1 >= target_end_time:
                new_end = index
                break
        trimmed.append((start, max(start + min_w, new_end)))

    return trimmed


def _semantic_ad_bounds_in_range(
    t0: float,
    t1: float,
    speech_spans: list[SpeechSpan],
) -> tuple[float, float] | None:
    starts: list[float] = []
    ends: list[float] = []
    for span in speech_spans:
        extra = span.model_extra or {}
        if extra.get("source") != "semantic":
            continue
        if float(span.t1) - float(span.t0) > 75.0:
            continue
        if float(extra.get("semantic_ad_score", 0.0)) < SEMANTIC_WEAK_AD_THRESHOLD:
            continue
        if float(extra.get("semantic_margin", 0.0)) < SEMANTIC_WEAK_AD_MARGIN:
            continue
        if span.t1 < t0 - 3.0 or span.t0 > t1 + 3.0:
            continue
        starts.append(float(span.t0))
        ends.append(float(span.t1))
    if not starts:
        return None
    return min(starts), max(ends)


def _trim_semantic_ad_tails(
    intervals: list[tuple[int, int]],
    windows: list[VisualWindow],
    speech_spans: list[SpeechSpan],
) -> list[tuple[int, int]]:
    if not windows:
        return intervals
    window_sec = windows[0].t1 - windows[0].t0
    min_w = max(1, int(AD_MIN_SEC / max(window_sec, 1e-6)))
    trimmed: list[tuple[int, int]] = []
    for start, end in intervals:
        start_time = windows[start].t0
        end_time = windows[end - 1].t1
        semantic_bounds = _semantic_ad_bounds_in_range(start_time, end_time, speech_spans)
        if semantic_bounds is None:
            trimmed.append((start, end))
            continue

        _, semantic_end = semantic_bounds
        target_end_time = min(end_time, semantic_end + 5.0)
        short_semantic_ends = [
            float(span.t1)
            for span in speech_spans
            if (span.model_extra or {}).get("source") == "semantic"
            and float(span.t1) - float(span.t0) <= 20.0
            and float((span.model_extra or {}).get("semantic_ad_score", 0.0)) >= SEMANTIC_WEAK_AD_THRESHOLD
            and float((span.model_extra or {}).get("semantic_margin", 0.0)) >= SEMANTIC_WEAK_AD_MARGIN
            and span.t1 >= start_time - 3.0
            and span.t0 <= end_time + 3.0
        ]
        target_start_time = start_time
        if short_semantic_ends:
            short_end = max(short_semantic_ends)
            if short_end - start_time >= AD_MIN_SEC and end_time - short_end <= 20.0:
                target_end_time = min(target_end_time, max(start_time + AD_MIN_SEC, short_end - 2.0))
                target_start_time = max(start_time, target_end_time - 36.0)

        new_start = start
        if target_start_time > start_time + 4.0:
            for index in range(start, end):
                if windows[index].t0 >= target_start_time:
                    new_start = index
                    break
            if end - new_start < min_w:
                new_start = max(start, end - min_w)

        should_keep_end = end_time - target_end_time < 8.0 and not short_semantic_ends
        if should_keep_end:
            trimmed.append((new_start, end))
            continue

        new_end = end
        for index in range(new_start + min_w, end + 1):
            if windows[index - 1].t1 >= target_end_time:
                new_end = index
                break
        trimmed.append((new_start, max(new_start + min_w, new_end)))
    return trimmed


def _duration_prior(duration_sec: float) -> float:
    common_durations = (30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0)
    return float(
        max(
            np.exp(-abs(duration_sec - target) / max(12.0, target * 0.25))
            for target in common_durations
        )
    )


def _optimize_interval_boundaries(
    interval: tuple[int, int],
    windows: list[VisualWindow],
    edge_scores: np.ndarray,
    foreign_scores: np.ndarray,
    text_scores: np.ndarray,
    content_penalties: np.ndarray,
    speech_spans: list[SpeechSpan],
    *,
    search_sec: float = 28.0,
) -> tuple[int, int]:
    """Local boundary cleanup using generic support signals.

    The candidate finder sometimes lands on the right ad but includes a small
    piece of neighboring content, or picks a high-score slice inside the ad.
    Search nearby boundary pairs and keep the one with the best multimodal
    support, without assuming ad count, brand, or test-specific timing.
    """
    if not windows:
        return interval

    start, end = interval
    window_sec = windows[0].t1 - windows[0].t0
    search_w = max(2, int(search_sec / max(window_sec, 1e-6)))
    min_w = max(1, int(AD_MIN_SEC / max(window_sec, 1e-6)))
    max_w = max(min_w + 1, int(AD_MAX_SEC / max(window_sec, 1e-6)) + 1)
    N = len(windows)

    norm_foreign = foreign_scores / (float(foreign_scores.max()) + 1e-9)
    norm_text = text_scores / (float(text_scores.max()) + 1e-9) if text_scores.max() > 0 else text_scores
    norm_penalty = (
        content_penalties / (float(content_penalties.max()) + 1e-9)
        if content_penalties.max() > 0
        else content_penalties
    )
    feature_rows = np.asarray(
        [
            [
                float(window.motion_score),
                float(window.luminance_mean),
                float(window.edge_density),
                float(window.palette_delta),
                1.0 if window.high_text_density else 0.0,
                float(norm_foreign[index]),
                float(norm_text[index]),
                float(norm_penalty[index]),
            ]
            for index, window in enumerate(windows)
        ],
        dtype=np.float64,
    )
    feature_prefix = np.vstack(
        [np.zeros((1, feature_rows.shape[1])), np.cumsum(feature_rows, axis=0)]
    )
    return_context_w = max(8, int(45.0 / max(window_sec, 1e-6)))

    def mean_vec(lo: int, hi: int) -> np.ndarray | None:
        lo = max(0, min(N, lo))
        hi = max(0, min(N, hi))
        if hi <= lo:
            return None
        return (feature_prefix[hi] - feature_prefix[lo]) / (hi - lo)

    def local_return_to_content_score(s: int, e: int) -> float:
        before = mean_vec(s - return_context_w, s)
        inside = mean_vec(s, e)
        after = mean_vec(e, e + return_context_w)
        if before is None or inside is None or after is None:
            return 0.0
        before_inside = float(np.linalg.norm(before - inside))
        after_inside = float(np.linalg.norm(after - inside))
        before_after = float(np.linalg.norm(before - after))
        return max(0.0, 0.5 * (before_inside + after_inside) - before_after)

    def score(s: int, e: int) -> float:
        if e <= s:
            return float("-inf")
        duration = windows[e - 1].t1 - windows[s].t0
        if duration < AD_MIN_SEC or duration > AD_MAX_SEC:
            return float("-inf")
        text_mean = float(text_scores[s:e].mean())
        text_peak = float(text_scores[s:e].max())
        foreign_mean = float(foreign_scores[s:e].mean())
        penalty_mean = float(content_penalties[s:e].mean())
        if penalty_mean >= HIGH_CONTENT_PENALTY_REJECT and text_mean < 0.18:
            return float("-inf")
        if text_peak < 0.05 and foreign_mean < 0.30:
            return float("-inf")
        support = _interval_support_score((s, e), windows, foreign_scores, text_scores, content_penalties)
        start_edge = float(edge_scores[s]) if 0 <= s < len(edge_scores) else 0.0
        end_edge = float(edge_scores[e]) if 0 <= e < len(edge_scores) else 0.0
        boundary_mean = 0.5 * (start_edge + end_edge)
        return_score = local_return_to_content_score(s, e)
        duration_score = _duration_prior(duration)
        return (
            support
            + 0.45 * boundary_mean
            + 0.25 * return_score
            + 0.12 * duration_score
            - 0.35 * penalty_mean
        )

    best = (score(start, end), start, end)
    s_lo = max(0, start - search_w)
    s_hi = min(N - min_w, start + search_w)
    for s in range(s_lo, s_hi + 1):
        e_lo = max(s + min_w, end - search_w)
        e_hi = min(N, s + max_w, end + search_w)
        for e in range(e_lo, e_hi + 1):
            candidate_score = score(s, e)
            if candidate_score > best[0]:
                best = (candidate_score, s, e)

    return best[1], best[2]


# ---------------------------------------------------------------------------
# Segment building
# ---------------------------------------------------------------------------

def _normalize_output_label(label: str) -> str:
    return label if label in OUTPUT_LABELS else LABEL_CORE_CONTENT


def _make_segment_dict(label: str, start: float, end: float) -> dict[str, Any]:
    label = _normalize_output_label(label)
    return {
        "start": round(start, 3),
        "end":   round(end,   3),
        "label": label,
        "kind":  _KIND_FOR_LABEL.get(label, KIND_NON_CONTENT),
    }


def _span_source(span: SpeechSpan) -> str:
    return str((span.model_extra or {}).get("source", "asr"))


def _merge_adjacent_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for segment in segments:
        if segment["end"] <= segment["start"]:
            continue
        if merged and merged[-1]["label"] == segment["label"]:
            merged[-1]["end"] = segment["end"]
        else:
            merged.append(dict(segment))
    return merged


def _merge_short_auxiliary_segments(segments: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        segment_duration = end - start
        keep_short_edge_auxiliary = (
            segment_duration >= MIN_EDGE_AUXILIARY_SEC
            and (
                (segment["label"] == LABEL_INTRO and start <= EDGE_AUXILIARY_KEEP_SEC)
                or (segment["label"] == LABEL_OUTRO and end >= duration - EDGE_AUXILIARY_KEEP_SEC)
            )
        )
        if (
            segment["label"] not in {LABEL_CORE_CONTENT, LABEL_ADVERTISEMENT}
            and segment_duration < AUXILIARY_MIN_SEC
            and not keep_short_edge_auxiliary
        ):
            segment = _make_segment_dict(LABEL_CORE_CONTENT, start, end)
        normalized.append(segment)
    return _merge_adjacent_segments(normalized)


def _merge_short_core_content_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [dict(segment) for segment in segments]
    auxiliary_labels = {LABEL_INTRO, LABEL_OUTRO}

    for index, segment in enumerate(normalized):
        if segment["label"] != LABEL_CORE_CONTENT:
            continue

        duration = float(segment["end"]) - float(segment["start"])
        if duration >= SHORT_CORE_CONTENT_MIN_SEC:
            continue

        previous_label = normalized[index - 1]["label"] if index > 0 else None
        next_label = normalized[index + 1]["label"] if index < len(normalized) - 1 else None

        if previous_label == next_label and previous_label is not None:
            segment["label"] = previous_label
            segment["kind"] = _KIND_FOR_LABEL[previous_label]
        elif next_label in auxiliary_labels:
            segment["label"] = next_label
            segment["kind"] = _KIND_FOR_LABEL[next_label]
        elif previous_label in auxiliary_labels and next_label != LABEL_ADVERTISEMENT:
            segment["label"] = previous_label
            segment["kind"] = _KIND_FOR_LABEL[previous_label]

    return _merge_adjacent_segments(normalized)


def _build_non_ad_segments(
    windows: list[VisualWindow],
    run_indices: list[int],
    duration: float,
    speech_spans: list[SpeechSpan],
    intro_used: bool,
    outro_used: bool,
) -> tuple[list[dict[str, Any]], bool, bool]:
    if not run_indices:
        return [], intro_used, outro_used

    labels = [LABEL_CORE_CONTENT for _ in run_indices]

    intro_end = find_intro_end_time(
        windows,
        run_indices,
        speech_spans,
        intro_used=intro_used,
        edge_title_card_sec=EDGE_TITLE_CARD_SEC,
        min_edge_auxiliary_sec=MIN_EDGE_AUXILIARY_SEC,
        opening_sequence_max_sec=OPENING_SEQUENCE_MAX_SEC,
    )
    if intro_end is not None:
        for position, index in enumerate(run_indices):
            if windows[index].t0 < intro_end:
                labels[position] = LABEL_INTRO

    outro_start = find_outro_start_time(
        windows,
        run_indices,
        duration,
        speech_spans,
        outro_used=outro_used,
        outro_phrases=_OUTRO_PHRASES,
        edge_title_card_sec=EDGE_TITLE_CARD_SEC,
        ending_sequence_min_sec=ENDING_SEQUENCE_MIN_SEC,
    )
    if outro_start is not None:
        for position, index in enumerate(run_indices):
            if windows[index].t1 > outro_start:
                labels[position] = LABEL_OUTRO

    segments: list[dict[str, Any]] = []
    start_pos = 0
    while start_pos < len(run_indices):
        label = labels[start_pos]
        end_pos = start_pos + 1
        while end_pos < len(run_indices) and labels[end_pos] == label:
            end_pos += 1
        first_index = run_indices[start_pos]
        last_index = run_indices[end_pos - 1]
        segments.append(_make_segment_dict(label, windows[first_index].t0, windows[last_index].t1))
        if label == LABEL_INTRO:
            intro_used = True
        elif label == LABEL_OUTRO:
            outro_used = True
        start_pos = end_pos

    segments = _merge_short_auxiliary_segments(segments, duration)
    intro_used = intro_used or any(segment["label"] == LABEL_INTRO for segment in segments)
    outro_used = outro_used or any(segment["label"] == LABEL_OUTRO for segment in segments)
    return segments, intro_used, outro_used


def _build_segments_from_ad_intervals(
    ad_intervals: list[tuple[int, int]],
    windows: list[VisualWindow],
    duration: float,
    speech_spans: list[SpeechSpan],
) -> list[dict[str, Any]]:
    N = len(windows)

    is_ad = [False] * N
    for s, e in ad_intervals:
        for i in range(s, min(e, N)):
            is_ad[i] = True

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

            non_ad_segments, intro_used, outro_used = _build_non_ad_segments(
                windows,
                run_indices,
                duration,
                speech_spans,
                intro_used=intro_used,
                outro_used=outro_used,
            )
            segments.extend(non_ad_segments)
            i = j

    segments.sort(key=lambda s: s["start"])
    return _merge_short_core_content_segments(_merge_adjacent_segments(segments))


def _passes_final_ad_filters(
    interval: tuple[int, int],
    windows: list[VisualWindow],
    edge_scores: np.ndarray,
    foreign_scores: np.ndarray,
    text_scores: np.ndarray,
    content_penalties: np.ndarray,
    speech_spans: list[SpeechSpan],
) -> bool:
    rs, re = interval
    if re <= rs:
        return False

    final_text_mean = float(text_scores[rs:re].mean())
    final_text_peak = float(text_scores[rs:re].max())
    final_foreign_mean = float(foreign_scores[rs:re].mean())
    final_penalty_mean = float(content_penalties[rs:re].mean())
    interval_start_time = windows[rs].t0
    interval_end_time = windows[re - 1].t1
    has_text_anchor = _has_text_anchor_in_range(
        interval_start_time,
        interval_end_time,
        speech_spans,
    )
    asr_coverage = _asr_speech_coverage(
        interval_start_time,
        interval_end_time,
        speech_spans,
    )
    return_score = _interval_return_to_content_score(
        rs,
        re,
        windows,
        foreign_scores,
        text_scores,
        content_penalties,
    )
    final_start_edge = float(edge_scores[rs]) if 0 <= rs < len(edge_scores) else 0.0
    final_end_edge = float(edge_scores[re]) if 0 <= re < len(edge_scores) else 0.0
    final_boundary_mean = 0.5 * (final_start_edge + final_end_edge)

    if final_penalty_mean >= 0.28 and final_text_mean < 0.18:
        return False
    if final_penalty_mean >= 0.18 and final_text_mean < 0.05 and final_foreign_mean < 0.45:
        return False
    if (
        not has_text_anchor
        and final_text_mean < 0.08
        and final_text_peak <= 0.35
        and final_foreign_mean < 0.68
        and final_boundary_mean < 0.50
    ):
        return False
    if (
        not has_text_anchor
        and asr_coverage >= 0.75
        and final_text_peak < 0.05
        and final_foreign_mean < 0.40
        and final_boundary_mean < 0.50
        and return_score < 0.35
    ):
        return False
    if (
        not has_text_anchor
        and asr_coverage >= 0.75
        and final_penalty_mean >= 0.10
        and final_text_peak <= 0.25
        and return_score < 0.35
    ):
        return False
    if (
        not has_text_anchor
        and asr_coverage < 0.10
        and final_text_peak < 0.05
        and final_boundary_mean < 0.30
    ):
        return False
    if (
        not has_text_anchor
        and final_text_peak < 0.05
        and final_foreign_mean < 0.30
        and asr_coverage >= 0.45
    ):
        return False
    if (
        not has_text_anchor
        and final_text_peak < 0.30
        and not (
            final_foreign_mean >= 0.50
            or (final_foreign_mean >= 0.38 and final_boundary_mean >= 0.55)
            or (
                final_foreign_mean >= 0.38
                and final_boundary_mean >= 0.45
                and return_score >= 0.45
            )
            or return_score >= 0.70
        )
    ):
        return False
    if (
        not has_text_anchor
        and final_text_peak >= 0.60
        and asr_coverage >= 0.30
        and final_boundary_mean < 0.58
        and return_score < 0.25
    ):
        return False
    if final_text_mean < 0.03 and final_text_peak < 0.30 and return_score < 0.12:
        return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fuse_bundle_to_segments(
    bundle: AnalysisBundle,
    *,
    min_segment_seconds: float = 12.0,
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

    # Find high-confidence ad intervals using visual/audio interior,
    # boundary, explicit ad text, and content-domain penalty signals.
    text_scores = _compute_text_ad_scores(windows, bundle.speech_spans)
    content_penalties = _compute_content_text_penalties(windows, bundle.speech_spans)
    ad_intervals = _find_ad_intervals(
        smooth_edge,
        smooth_foreign,
        windows,
        text_scores=text_scores,
        content_penalties=content_penalties,
    )
    text_anchor_intervals = _find_text_anchor_intervals(
        windows,
        bundle.speech_spans,
        smooth_edge,
        smooth_foreign,
        text_scores,
        content_penalties,
    )
    if text_anchor_intervals:
        # Text anchors are strong evidence that an ad exists nearby, but they
        # are not always centered in the ad. Keep visual/audio candidates too
        # and let the interval support score pick the better boundary pair.
        ad_intervals = _suppress_close_ad_intervals(
            ad_intervals + text_anchor_intervals,
            windows,
            smooth_foreign,
            text_scores,
            content_penalties,
            smooth_edge,
            bundle.speech_spans,
        )

    if ad_intervals:
        # Refine each boundary to the nearest local edge maximum
        refined: list[tuple[int, int]] = []
        for s, e in ad_intervals:
            base_interval = (s, e)
            rs = _refine_boundary(s, smooth_edge, "start", windows, search_sec=12.0)
            re = _refine_boundary(e, smooth_edge, "end",   windows, search_sec=12.0)
            rs = min(rs, s)
            re = max(re, e)
            # Ensure minimum duration after refinement
            window_sec = windows[0].t1 - windows[0].t0 if windows else 2.0
            min_w = max(1, int(AD_MIN_SEC / window_sec))
            if re - rs < min_w:
                re = min(len(windows), rs + min_w)
            interval_duration = windows[re - 1].t1 - windows[rs].t0
            interval_text_mean = float(text_scores[rs:re].mean()) if re > rs else 0.0
            interval_foreign_mean = float(smooth_foreign[rs:re].mean()) if re > rs else 0.0
            start_edge = float(smooth_edge[rs]) if 0 <= rs < len(smooth_edge) else 0.0
            end_edge = float(smooth_edge[re]) if 0 <= re < len(smooth_edge) else 0.0
            has_strong_boundary_pair = min(start_edge, end_edge) >= 0.32
            should_expand_text_ad = (
                interval_text_mean >= 0.15
                and (not has_strong_boundary_pair or interval_duration <= AD_MIN_SEC + 2.0)
            )
            should_expand_visual_ad = (
                interval_text_mean < 0.15
                and interval_foreign_mean >= 0.25
            )
            if interval_duration <= 45.0 and (should_expand_text_ad or should_expand_visual_ad):
                pre_expand_rs = rs
                pre_expand_anchor_times = _ad_anchor_times_in_range(
                    windows[rs].t0,
                    windows[re - 1].t1,
                    bundle.speech_spans,
                )
                rs, re = _expand_ad_interval(
                    rs,
                    re,
                    windows,
                    smooth_foreign,
                    text_scores,
                    content_penalties,
                    search_sec=25.0 if should_expand_text_ad else 12.0,
                    aggressive_visual=should_expand_text_ad or should_expand_visual_ad,
                )
                earliest_anchor = min(pre_expand_anchor_times) if pre_expand_anchor_times else None
                if (
                    rs < pre_expand_rs
                    and earliest_anchor is not None
                    and earliest_anchor - windows[pre_expand_rs].t0 >= AD_MIN_SEC
                ):
                    rs = pre_expand_rs
            final_interval = (rs, re)
            if not _passes_final_ad_filters(
                final_interval,
                windows,
                smooth_edge,
                smooth_foreign,
                text_scores,
                content_penalties,
                bundle.speech_spans,
            ):
                base_duration = windows[e - 1].t1 - windows[s].t0
                base_has_reliable_signal = (
                    base_duration >= 40.0
                    or float(text_scores[s:e].max()) >= 0.15
                    or _has_text_anchor_in_range(
                        windows[s].t0,
                        windows[e - 1].t1,
                        bundle.speech_spans,
                    )
                )
                if (
                    base_interval == final_interval
                    or not base_has_reliable_signal
                    or not _passes_final_ad_filters(
                        base_interval,
                        windows,
                        smooth_edge,
                        smooth_foreign,
                        text_scores,
                        content_penalties,
                        bundle.speech_spans,
                    )
                ):
                    continue
                rs, re = base_interval
            has_text_anchor_after_refine = _has_text_anchor_in_range(
                windows[rs].t0,
                windows[re - 1].t1,
                bundle.speech_spans,
            )
            anchor_times_after_refine = _ad_anchor_times_in_range(
                windows[rs].t0,
                windows[re - 1].t1,
                bundle.speech_spans,
            )
            opt_rs, opt_re = _optimize_interval_boundaries(
                (rs, re),
                windows,
                smooth_edge,
                smooth_foreign,
                text_scores,
                content_penalties,
                bundle.speech_spans,
                search_sec=28.0,
            )
            if has_text_anchor_after_refine:
                window_sec = windows[0].t1 - windows[0].t0 if windows else 1.0
                max_anchor_shift_w = 0
                earliest_anchor = min(anchor_times_after_refine) if anchor_times_after_refine else None
                if (
                    opt_rs < rs
                    and earliest_anchor is not None
                    and earliest_anchor - windows[rs].t0 >= AD_MIN_SEC
                ):
                    opt_rs = rs
                if opt_rs > rs:
                    if (
                        earliest_anchor is not None
                        and earliest_anchor - windows[rs].t0 < AD_MIN_SEC
                    ):
                        anchor_index = rs
                        for index, window in enumerate(windows):
                            if window.t0 <= earliest_anchor < window.t1:
                                anchor_index = index
                                break
                        opt_rs = min(opt_rs, anchor_index)
                    else:
                        opt_rs = min(opt_rs, rs + max_anchor_shift_w)
                if opt_re > re:
                    opt_re = min(opt_re, re + max_anchor_shift_w)
                elif opt_re < re:
                    opt_re = max(opt_re, re - max_anchor_shift_w)
            rs, re = opt_rs, opt_re
            if not has_text_anchor_after_refine:
                rs, re = _snap_text_supported_ad_boundaries(
                    (rs, re),
                    windows,
                    bundle.speech_spans,
                )
            if not _passes_final_ad_filters(
                (rs, re),
                windows,
                smooth_edge,
                smooth_foreign,
                text_scores,
                content_penalties,
                bundle.speech_spans,
            ):
                continue
            refined.append((rs, re))

        refined = _suppress_close_ad_intervals(
            refined,
            windows,
            smooth_foreign,
            text_scores,
            content_penalties,
            smooth_edge,
            bundle.speech_spans,
        )
        refined = _trim_text_anchor_tails(refined, windows, bundle.speech_spans)
        refined = _trim_semantic_ad_tails(refined, windows, bundle.speech_spans)
        refined = _trim_confirmed_ad_tails(refined, windows, content_penalties, text_scores)
        return _build_segments_from_ad_intervals(refined, windows, duration, bundle.speech_spans)

    # No confident ads found. Still classify conservative edge-only
    # non-content such as an explicit intro/outro if present.
    return _build_segments_from_ad_intervals([], windows, duration, bundle.speech_spans)


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
