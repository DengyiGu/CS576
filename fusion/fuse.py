from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
from schemas.modality import AnalysisBundle, AudioWindow, SpeechSpan, VisualWindow

# Label constants
LABEL_CORE_CONTENT = "Core Content"
LABEL_INTRO = "Intro"
LABEL_OUTRO = "Outro"
LABEL_ADVERTISEMENT = "Advertisement"
KIND_CONTENT = "content"
KIND_NON_CONTENT = "non-content"

_KIND_FOR_LABEL: dict[str, str] = {
    LABEL_CORE_CONTENT: KIND_CONTENT,
    LABEL_INTRO: KIND_NON_CONTENT,
    LABEL_OUTRO: KIND_NON_CONTENT,
    LABEL_ADVERTISEMENT: KIND_NON_CONTENT,
}

# ---------------------------------------------------------------------------
# Hyper‑parameters – tuned for vlog‑style “Day in the life” content
# ---------------------------------------------------------------------------
AD_MIN_SEC = 28.0
AD_MAX_SEC = 60.0
GAP_MIN_SEC = 190.0          # minimum gap between consecutive ads
FIRST_AD_MIN_START_SEC = 50.0 # ads cannot start earlier than this

W_AUDIO = 1.50               # audio anomaly weight
W_VISUAL_SEMANTIC = 1.20     # visual graphics weight (increased to capture ad UI)

SMOOTH_HALF_WIN = 9          # smoothing radius (windows)

SPEECH_CONTEXT_SEC = 18.0    # look‑ahead for speech coverage

EDGE_WEIGHT = 5.0            # heavily favour strong boundaries (was 2.5)
INTERIOR_WEIGHT = 1.0        # keep interior contribution moderate

# ---------------------------------------------------------------------------
# Ad‑signal phrase / brand loading
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
_SELF_PROMO_PHRASES = _AD_PHRASES.get("self_promotion", [])
_OUTRO_PHRASES = _AD_PHRASES.get("outro", [])
_INTRO_PHRASES = _AD_PHRASES.get("intro", [])
_RECAP_PHRASES = _AD_PHRASES.get("recap", [])


# ===================================================================
# Per‑window audio helpers
# ===================================================================
def _audio_features(
    t0: float, t1: float, audio_windows: list[AudioWindow]
) -> tuple[float, float]:
    mid = 0.5 * (t0 + t1)
    best_dist = float("inf")
    anomaly = 0.0
    energy = 1.0
    for aw in audio_windows:
        d = abs(0.5 * (aw.t0 + aw.t1) - mid)
        if d < best_dist:
            best_dist = d
            extra = aw.model_extra or {}
            anomaly = float(extra.get("anomaly_score", 0.0))
            energy = float(extra.get("energy_rms", 1.0))
    return anomaly, energy


def _audio_delta(
    t_mid: float,
    audio_windows: list[AudioWindow],
    half_sec: float = 4.0,
) -> float:
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
    lo, hi = t0 - 30.0, t1 + 30.0
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
        return 0.75
    if brand_hits == 1:
        return 0.45

    return 0.0


def _visual_semantic_ad_score(w: VisualWindow) -> float:
    score = 0.0
    if w.high_text_density:
        score += 0.35
    if w.visual_hypothesis == "graphics_heavy":
        score += 0.45 * float(w.hypothesis_confidence)
    if w.edge_density > 0.45:
        score += 0.20 * min(1.0, (float(w.edge_density) - 0.45) / 0.35)
    return min(1.0, score)


# ===================================================================
# Step 1 – per‑window foreignness score
# ===================================================================
def _compute_foreignness_scores(
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

        visual_semantic = _visual_semantic_ad_score(w)
        anomaly, energy = _audio_features(t0, t1, audio_windows)
        audio_score = float(anomaly)

        if energy < 0.04:
            audio_score = max(audio_score, 0.92)

        cov = _speech_coverage(t0, t1, speech_spans)
        nearby = _has_nearby_speech(t0, t1, speech_spans, SPEECH_CONTEXT_SEC)
        text_sig = _speech_text_ad_signal(t0, t1, speech_spans)

        nospeech_score = 0.0
        if not nearby:
            nospeech_score = 0.95
        elif cov < 0.18:
            nospeech_score = 0.80

        if text_sig > 0:
            audio_score = max(audio_score, text_sig)

        # dampen intro/outro regions to avoid false positives
        if mid < duration * 0.055 or mid > duration * 0.94:
            visual_semantic *= 0.25
            audio_score *= 0.25
            nospeech_score *= 0.25

        scores[i] = (
            W_VISUAL_SEMANTIC * visual_semantic
            + W_AUDIO * audio_score
            + 0.60 * nospeech_score
        )
    return scores


# ===================================================================
# Step 2 – per‑boundary edge score
# ===================================================================
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
        vis = float(windows[i].palette_delta)
        scene_cut = 1.0 if windows[i].shot_boundary_near else 0.0
        if windows[i].shot_boundary_distance_sec is not None:
            scene_cut = max(scene_cut, max(0.0, 1.0 - float(windows[i].shot_boundary_distance_sec) / 9.0))

        aud_delta = _audio_delta(t_boundary, audio_windows, half_sec=8.0)

        had = _has_nearby_speech(t_boundary - 12.0, t_boundary, speech_spans, 3.0)
        has = _has_nearby_speech(t_boundary, t_boundary + 12.0, speech_spans, 3.0)
        speech_transition = 1.0 if (had != has) else 0.0

        # dampen intro/outro edges
        if t_boundary < duration * 0.06 or t_boundary > duration * 0.93:
            vis *= 0.20
            scene_cut *= 0.20
            aud_delta *= 0.25
            speech_transition *= 0.20

        edge[i] = 0.62 * vis + 0.18 * scene_cut + 0.15 * aud_delta + 0.05 * speech_transition
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


# ===================================================================
# Step 3 – Dynamic‑programming ad placement (generalised)
# ===================================================================
def _find_best_ads(
    edge_scores: np.ndarray,
    foreign_scores: np.ndarray,
    windows: list[VisualWindow],
    max_ads: int = 6,
) -> list[tuple[int, int]]:
    N = len(windows)
    if N == 0:
        return []

    e_max = edge_scores.max()
    f_max = foreign_scores.max()
    norm_edge = edge_scores / (e_max + 1e-9)
    norm_foreign = foreign_scores / (f_max + 1e-9)
    norm_edge = np.append(norm_edge, 0.0)
    cum_foreign = np.concatenate([[0.0], np.cumsum(norm_foreign)])

    def interval_score(s: int, e: int) -> float:
        interior_mean = (cum_foreign[e] - cum_foreign[s]) / max(e - s, 1)
        return (EDGE_WEIGHT * (norm_edge[s] + norm_edge[e])
                + INTERIOR_WEIGHT * interior_mean)

    window_sec = windows[0].t1 - windows[0].t0 if windows else 2.0
    min_w = max(1, int(AD_MIN_SEC / window_sec))
    max_w = max(min_w + 1, int(AD_MAX_SEC / window_sec) + 1)
    gap_w = max(1, int(GAP_MIN_SEC / window_sec))

    first_start_idx = 0
    for i, w in enumerate(windows):
        if w.t0 >= FIRST_AD_MIN_START_SEC:
            first_start_idx = i
            break

    NEG_INF = float("-inf")
    dp = np.full((N + 1, max_ads + 1), NEG_INF, dtype=np.float64)
    prev = np.full((N + 1, max_ads + 1, 2), -1, dtype=np.int32)

    dp[0, 0] = 0.0

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

            # 1‑ad case
            if sc > dp[e, 1]:
                dp[e, 1] = sc
                prev[e, 1] = [s, -1]

            # k‑ads case (k >= 2)
            for k in range(2, max_ads + 1):
                me = s - gap_w
                if me < 0:
                    continue
                for prev_e in range(me + 1):
                    if dp[prev_e, k - 1] > NEG_INF:
                        total = dp[prev_e, k - 1] + sc
                        if total > dp[e, k]:
                            dp[e, k] = total
                            prev[e, k] = [s, prev_e]

    # Find best overall
    best_total = NEG_INF
    best_k = 0
    best_end = 0
    for k in range(1, max_ads + 1):
        for e in range(N + 1):
            if dp[e, k] > best_total:
                best_total = dp[e, k]
                best_k = k
                best_end = e

    if best_total <= NEG_INF:
        return []

    intervals: list[tuple[int, int]] = []
    current_e = best_end
    current_k = best_k
    while current_k > 0 and current_e > 0:
        s = prev[current_e, current_k][0]
        intervals.append((s, current_e))
        current_e = prev[current_e, current_k][1]
        current_k -= 1

    intervals.reverse()
    return intervals


# ===================================================================
# Boundary snapping (ad‑boundary refinement + intro/outro)
# ===================================================================
def _snap_boundary_to_shot(
    windows: list[VisualWindow],
    idx: int,
    direction: str,
    search_sec: float = 8.0,
) -> int:
    """Snap a boundary index to the nearest strong shot‑boundary window."""
    N = len(windows)
    window_sec = windows[0].t1 - windows[0].t0 if windows else 2.0
    half_w = max(1, int(search_sec / window_sec))

    if direction == "start":
        lo = max(0, idx - half_w)
        hi = min(N - 1, idx + half_w)
    else:
        lo = max(0, idx - half_w)
        hi = min(N - 1, idx + half_w)

    best_i = idx
    best_score = -1.0
    for k in range(lo, hi + 1):
        w = windows[k]
        score = float(w.palette_delta)
        if w.shot_boundary_near:
            score += 0.5
        if score > best_score:
            best_score = score
            best_i = k
    return best_i


# ===================================================================
# Intro / Outro detection (visual freeze‑frame / static analysis)
# ===================================================================
def _detect_intro_end(
    windows: list[VisualWindow],
    min_start: float = 10.0,
    max_end: float = 120.0,
) -> float | None:
    """Find intro end by locating the first long static/freeze run after `min_start`."""
    i = 0
    while i < len(windows):
        w = windows[i]
        if w.t0 < min_start:
            i += 1
            continue
        if w.t0 > max_end:
            break
        if w.visual_hypothesis == "static" and float(w.palette_delta) > 0.9:
            return float(w.t0)
        i += 1
    return None


def _detect_outro_start(
    windows: list[VisualWindow],
    min_static_dur: float = 12.0,
    min_start_frac: float = 0.85,
    duration: float = 0.0,
) -> float | None:
    """Detect outro start by finding the last long static block near the end."""
    if duration <= 0:
        return None
    min_start = duration * min_start_frac

    best_start: float | None = None
    i = len(windows) - 1
    while i >= 0:
        w = windows[i]
        if w.t0 < min_start:
            break
        if w.visual_hypothesis == "static" and float(w.palette_delta) > 0.9:
            run_end_t = w.t1
            j = i
            while j >= 0 and windows[j].visual_hypothesis == "static" and float(windows[j].palette_delta) > 0.9:
                j -= 1
            run_start_t = windows[j + 1].t0
            run_dur = run_end_t - run_start_t
            if run_dur >= min_static_dur:
                best_start = run_start_t
            i = j
        else:
            i -= 1

    return best_start


# ===================================================================
# Segment building
# ===================================================================
def _make_segment_dict(label: str, start: float, end: float) -> dict[str, Any]:
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "label": label,
        "kind": _KIND_FOR_LABEL.get(label, KIND_NON_CONTENT),
    }


def _build_segments_from_ad_intervals(
    ad_intervals: list[tuple[int, int]],
    windows: list[VisualWindow],
    duration: float,
    intro_end_sec: float | None = None,
    outro_start_sec: float | None = None,
) -> list[dict[str, Any]]:
    N = len(windows)

    # Detect intro/outro boundaries using visual cues
    if intro_end_sec is None:
        intro_end_sec = _detect_intro_end(windows, min_start=10.0, max_end=120.0)
    if outro_start_sec is None:
        outro_start_sec = _detect_outro_start(windows, min_static_dur=12.0, min_start_frac=0.85, duration=duration)

    # Snap ad boundaries to strong shot cuts
    snapped: list[tuple[int, int]] = []
    for s, e in ad_intervals:
        rs = _snap_boundary_to_shot(windows, s, "start", search_sec=8.0)
        re = _snap_boundary_to_shot(windows, e - 1, "end", search_sec=8.0) + 1
        window_sec = windows[0].t1 - windows[0].t0 if windows else 2.0
        min_w = max(1, int(AD_MIN_SEC / window_sec))
        if re - rs < min_w:
            re = min(N, rs + min_w)
        snapped.append((rs, re))

    is_ad = [False] * N
    for s, e in snapped:
        for i in range(s, min(e, N)):
            is_ad[i] = True

    segments: list[dict[str, Any]] = []
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
            continue

        # Non‑ad run
        j = i
        while j < N and not is_ad[j]:
            j += 1
        run_start = windows[i].t0
        run_end = windows[j - 1].t1

        # Split the run if it crosses an intro / outro boundary
        split_points: list[float] = []
        # Intro split (only for the very first non‑ad run)
        if i == 0 and intro_end_sec is not None and run_start < intro_end_sec < run_end:
            split_points.append(intro_end_sec)
        # Outro split (only for the very last non‑ad run)
        if j == N and outro_start_sec is not None and run_start < outro_start_sec < run_end:
            split_points.append(outro_start_sec)

        if split_points:
            split_points.sort()
            prev_bound = run_start
            for sp in split_points:
                # snap this extra split to a nearby shot for accuracy
                idx_sp = 0
                for k, w in enumerate(windows):
                    if w.t0 <= sp < w.t1:
                        idx_sp = k
                        break
                idx_sp = _snap_boundary_to_shot(windows, idx_sp, "end", search_sec=4.0)
                sp_snapped = windows[idx_sp].t1
                if sp_snapped > prev_bound:
                    # determine label for this sub‑run
                    if i == 0 and sp == intro_end_sec:
                        label = LABEL_INTRO
                    elif j == N and sp == outro_start_sec:
                        label = LABEL_CORE_CONTENT
                    else:
                        label = LABEL_CORE_CONTENT
                    segments.append(_make_segment_dict(label, prev_bound, sp_snapped))
                    prev_bound = sp_snapped
            # Remaining part
            if prev_bound < run_end:
                # last part: if it's after outro split, label outro; else core content
                if j == N and outro_start_sec is not None and prev_bound >= outro_start_sec:
                    label = LABEL_OUTRO
                elif i == 0:
                    label = LABEL_CORE_CONTENT  # already consumed intro
                else:
                    label = LABEL_CORE_CONTENT
                segments.append(_make_segment_dict(label, prev_bound, run_end))
        else:
            # No split – label whole run
            if i == 0:
                label = LABEL_INTRO
            elif j == N:
                label = LABEL_OUTRO
            else:
                label = LABEL_CORE_CONTENT
            segments.append(_make_segment_dict(label, run_start, run_end))

        i = j

    segments.sort(key=lambda s: s["start"])
    return segments


# ===================================================================
# Public API
# ===================================================================
def fuse_bundle_to_segments(
    bundle: AnalysisBundle,
    *,
    min_segment_seconds: float = 12.0,
    enforce_three_ads: bool = True,    # kept for API compatibility, not used now
    intro_end_sec: float | None = None,
    outro_start_sec: float | None = None,
) -> list[dict[str, Any]]:
    if bundle.visual is None or not bundle.visual.windows:
        return []
    windows = bundle.visual.windows
    duration = bundle.visual.duration_sec or bundle.duration_sec

    raw_foreign = _compute_foreignness_scores(
        windows, bundle.audio_windows, bundle.speech_spans, duration
    )
    smooth_foreign = _smooth(raw_foreign, SMOOTH_HALF_WIN)

    raw_edge = _compute_edge_scores(
        windows, bundle.audio_windows, bundle.speech_spans, duration
    )
    smooth_edge = _smooth(raw_edge, SMOOTH_HALF_WIN)

    ad_intervals = _find_best_ads(smooth_edge, smooth_foreign, windows, max_ads=6)

    if ad_intervals:
        return _build_segments_from_ad_intervals(
            ad_intervals, windows, duration,
            intro_end_sec=intro_end_sec,
            outro_start_sec=outro_start_sec,
        )

    return [_make_segment_dict(LABEL_CORE_CONTENT, windows[0].t0, windows[-1].t1)]


def load_bundle(path: Path) -> AnalysisBundle:
    return AnalysisBundle.model_validate_json(path.read_text(encoding="utf-8"))


def write_segments_json(segments: list[dict[str, Any]], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "source": "fusion",
        "segments": segments,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")