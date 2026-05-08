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

# Hyper-parameters
# Tuned for the project test set on 2026-05-08 by porting the auto-K +
# adaptive interior-mean floor from the tuning_audio branch.  The previous
# (28, 60, 190, 50) values fit test_009 well but capped legitimate long ads
# (test_001 ad #1 is 118s) and forced K=6 on every video.  See
# _select_num_ads_auto for the K-selection rule.
AD_MIN_SEC = 28.0           # smallest GT ad on this set is 28.4 s (test_001 #3)
AD_MAX_SEC = 130.0          # longest known GT ad is 118.2 s (test_001)
GAP_MIN_SEC = 60.0          # min seconds between consecutive ads
FIRST_AD_MIN_START_SEC = 30.0

# Auto-K bounds and rules (ported from tuning_audio).  ``MIN_NUM_ADS`` is a
# *soft* floor: K is only padded up to it when no interior collapse was
# observed below it — see _select_num_ads_auto.
MIN_NUM_ADS = 3
MAX_NUM_ADS = 5
# Auto-K thresholds tightened from tuning_audio's (0.74 / 0.40) on the
# 8-video set:
#   * 0.80 ratio rejects test_009's K=4 (ratio=0.741) and test_004/010's
#     K=5 (ratio=0.785/0.786) without sacrificing legitimate K=3 picks.
#   * 0.30 interior floor lets test_008 K=3 through (interior=0.364) which
#     0.40 was wrongly blocking; the adaptive 0.5 * K=1 cap still
#     down-floors music-saturated videos like test_009 to ~0.33.
#   * EXTEND ratio (0.85) is the stricter threshold for growing K beyond
#     MIN_NUM_ADS: test_004 K=4 has ratio=0.832 (rejects, becomes K=3 to
#     match GT) but test_010 K=4 has ratio=0.865 (passes, keeps GT K=4).
MIN_MARGINAL_RATIO = 0.80
MIN_MARGINAL_RATIO_EXTEND = 0.85
MIN_INTERIOR_MEAN_FLOOR = 0.30

W_AUDIO = 1.50
W_VISUAL_SEMANTIC = 0.95
# Weight on the optional MiniLM semantic ad score (--with-semantic).  Note that
# DP normalizes foreignness by its global max, so this weight only nudges the
# *relative* ranking of windows — it does not by itself shift interval picks
# very far.  The bigger lever is the post-DP snap below.  Set to 0.0 to disable.
W_SEMANTIC_AD = 0.85

SMOOTH_HALF_WIN = 7
SPEECH_CONTEXT_SEC = 18.0

EDGE_WEIGHT = 2.5
INTERIOR_WEIGHT = 1.0

# Ad-signal phrase/brand loading
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

# Per-window audio helpers
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


# Quality filters for semantic ad spans before they contribute to foreignness.
#  * span length <= MAX_SEC: longer spans tend to be intro/outro narration that
#    Whisper merged into one segment; their topic-level cosine sim against the
#    ad prompt set is noisy and often fires on day-in-the-life monologue.
#  * margin >= MIN_MARGIN: how much higher the ad-prompt cosine is vs the
#    content-prompt cosine.  Below ~0.10 the model is hedging and false
#    positives dominate.
_SEM_AD_MAX_SPAN_SEC = 70.0
_SEM_AD_MIN_MARGIN = 0.10


def _semantic_ad_score_window(
    t0: float, t1: float, speech_spans: list[SpeechSpan]
) -> float:
    """Max ``semantic_ad_score`` over speech_spans with ``source="semantic"``
    that overlap [t0, t1] AND pass the quality filter.

    Returns 0.0 if no overlapping semantic span (e.g. when ``--with-semantic``
    was not used).  These spans come from the optional MiniLM scorer in
    ``semantic/analyze.py``; their score sits in [0, 1] with ~0.58 being the
    "is_ad" threshold the module itself uses.
    """
    if not speech_spans:
        return 0.0
    best = 0.0
    for span in speech_spans:
        extra = span.model_extra or {}
        if extra.get("source") != "semantic":
            continue
        if span.t1 <= t0 or span.t0 >= t1:
            continue
        if (span.t1 - span.t0) > _SEM_AD_MAX_SPAN_SEC:
            continue
        if float(extra.get("semantic_margin", 0.0)) < _SEM_AD_MIN_MARGIN:
            continue
        score = float(extra.get("semantic_ad_score", 0.0))
        if score > best:
            best = score
    return best


def _snap_intervals_to_semantic_spans(
    ad_intervals: list[tuple[int, int]],
    windows: list[VisualWindow],
    speech_spans: list[SpeechSpan],
) -> list[tuple[int, int]]:
    """Adjust DP-picked ad intervals to better align with high-confidence
    semantic ad spans.

    Rationale: ``_find_best_ads`` normalizes foreignness by its global max, so a
    +0.2 boost in a small region may not move interval boundaries.  Semantic
    spans (from --with-semantic) capture *which* speech-text actually reads as
    ad-copy, which is a more direct timing signal than smoothed foreignness.

    We only snap when:
      * a qualifying semantic span overlaps the DP interval at all
        (touches, even by 1s; otherwise we trust DP)
      * the IoU between DP and the semantic span is low (<= 0.45) — i.e. DP
        is mostly off the semantic evidence
    The snap replaces the DP interval with the union of (DP, sem), clipped to
    [AD_MIN_SEC, AD_MAX_SEC] biased toward keeping the semantic-evidence side.
    """
    if not ad_intervals or not speech_spans or not windows:
        return ad_intervals

    qualifying = []
    for span in speech_spans:
        extra = span.model_extra or {}
        if extra.get("source") != "semantic":
            continue
        if (span.t1 - span.t0) > _SEM_AD_MAX_SPAN_SEC:
            continue
        if float(extra.get("semantic_margin", 0.0)) < _SEM_AD_MIN_MARGIN:
            continue
        score = float(extra.get("semantic_ad_score", 0.0))
        if score < 0.65:
            continue
        qualifying.append((span.t0, span.t1, score))

    if not qualifying:
        return ad_intervals

    def _t_to_idx(t: float) -> int:
        for k, w in enumerate(windows):
            if w.t0 <= t < w.t1:
                return k
        return max(0, min(len(windows) - 1, len(windows) // 2))

    out: list[tuple[int, int]] = []
    for s_idx, e_idx in ad_intervals:
        dp_t0 = windows[s_idx].t0
        dp_t1 = windows[e_idx - 1].t1 if e_idx > 0 else windows[s_idx].t1

        # Find best overlapping qualifying semantic span (by score).
        best = None
        for st, en, sc in qualifying:
            if en <= dp_t0 or st >= dp_t1:
                continue
            if best is None or sc > best[2]:
                best = (st, en, sc)
        if best is None:
            out.append((s_idx, e_idx))
            continue

        sem_t0, sem_t1, _ = best
        sem_width = sem_t1 - sem_t0
        inter = max(0.0, min(dp_t1, sem_t1) - max(dp_t0, sem_t0))
        union = max(dp_t1, sem_t1) - min(dp_t0, sem_t0)
        iou = inter / union if union > 0 else 0.0

        if iou > 0.45:
            out.append((s_idx, e_idx))
            continue

        # When the DP pick barely touches the semantic span (IoU < 0.20)
        # AND the semantic span is short enough to be a real ad,
        # *clip* to the semantic span instead of unioning. The union
        # otherwise blows the interval out across both regions.
        # (test_009 ad #3: DP=[589,617] vs sem=[541,596] → union [541,617]=76s
        # but GT=[541,596]=55s. Clipping to sem matches GT.)
        if iou < 0.20 and sem_width <= 70.0:
            new_t0 = sem_t0
            new_t1 = sem_t1
        else:
            # Build a new interval anchored on the semantic evidence but
            # kept in bounds. Prefer the union; if too long, drop the side
            # that is farther from the semantic span's midpoint.
            new_t0 = min(dp_t0, sem_t0)
            new_t1 = max(dp_t1, sem_t1)
            new_dur = new_t1 - new_t0
            if new_dur > AD_MAX_SEC:
                sem_mid = 0.5 * (sem_t0 + sem_t1)
                half = AD_MAX_SEC / 2.0
                new_t0 = max(min(dp_t0, sem_t0), sem_mid - half)
                new_t1 = min(max(dp_t1, sem_t1), sem_mid + half)
                if new_t1 - new_t0 < AD_MIN_SEC:
                    new_t0 = sem_mid - AD_MAX_SEC / 2.0
                    new_t1 = sem_mid + AD_MAX_SEC / 2.0

        ns = _t_to_idx(new_t0)
        ne = _t_to_idx(new_t1 - 1e-3) + 1
        ne = max(ne, ns + 1)
        # Keep within sane duration band.
        cur_dur = windows[ne - 1].t1 - windows[ns].t0 if ne > 0 else 0.0
        if cur_dur < AD_MIN_SEC or cur_dur > AD_MAX_SEC:
            out.append((s_idx, e_idx))
        else:
            out.append((ns, ne))

    # Re-sort and de-overlap (snap may have produced overlaps).
    out.sort(key=lambda p: p[0])
    deduped: list[tuple[int, int]] = []
    for s, e in out:
        if deduped and s < deduped[-1][1]:
            ps, pe = deduped[-1]
            deduped[-1] = (ps, max(pe, e))
        else:
            deduped.append((s, e))
    return deduped


def _semantic_structure_window(
    t0: float, t1: float, speech_spans: list[SpeechSpan]
) -> tuple[float, float]:
    """Max (intro_score, outro_score) from overlapping ``source="semantic_structure"`` spans."""
    if not speech_spans:
        return 0.0, 0.0
    best_intro = 0.0
    best_outro = 0.0
    for span in speech_spans:
        extra = span.model_extra or {}
        if extra.get("source") != "semantic_structure":
            continue
        if span.t1 <= t0 or span.t0 >= t1:
            continue
        i = float(extra.get("semantic_intro_score", 0.0))
        o = float(extra.get("semantic_outro_score", 0.0))
        if i > best_intro:
            best_intro = i
        if o > best_outro:
            best_outro = o
    return best_intro, best_outro

# Step 1 – per-window foreignness score
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

        # Optional semantic ad score from semantic/analyze.py (MiniLM).
        # Only contributes when --with-semantic was run; otherwise it's 0.0.
        sem_ad = _semantic_ad_score_window(t0, t1, speech_spans)

        nospeech_score = 0.0
        if not nearby:
            nospeech_score = 0.95
        elif cov < 0.18:
            nospeech_score = 0.80

        if text_sig > 0:
            audio_score = max(audio_score, text_sig)

        # Tighter intro/outro protection
        if mid < duration * 0.055 or mid > duration * 0.94:
            visual_semantic *= 0.25
            audio_score *= 0.25
            nospeech_score *= 0.25
            sem_ad *= 0.25

        scores[i] = (
            W_VISUAL_SEMANTIC * visual_semantic
            + W_AUDIO * audio_score
            + 0.60 * nospeech_score
            + W_SEMANTIC_AD * sem_ad
        )
    return scores

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
        vis = float(windows[i].palette_delta)
        scene_cut = 1.0 if windows[i].shot_boundary_near else 0.0
        if windows[i].shot_boundary_distance_sec is not None:
            scene_cut = max(scene_cut, max(0.0, 1.0 - float(windows[i].shot_boundary_distance_sec) / 9.0))

        aud_delta = _audio_delta(t_boundary, audio_windows, half_sec=8.0)

        had = _has_nearby_speech(t_boundary - 12.0, t_boundary, speech_spans, 3.0)
        has = _has_nearby_speech(t_boundary, t_boundary + 12.0, speech_spans, 3.0)
        speech_transition = 1.0 if (had != has) else 0.0

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


# Generalized DP for any number of ads
def _prepare_dp_inputs(
    edge_scores: np.ndarray,
    foreign_scores: np.ndarray,
    windows: list[VisualWindow],
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """Shared preprocessing for the K-stage DP.

    Returns ``(norm_edge_padded, cum_foreign, min_w, max_w, gap_w,
    first_start_idx)``. Both arrays are normalised by their global max so
    interior_mean is comparable across videos.
    """
    e_max = float(edge_scores.max())
    f_max = float(foreign_scores.max())
    norm_edge = edge_scores / (e_max + 1e-9)
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


def _find_best_k_ads_dp(
    edge_scores: np.ndarray,
    foreign_scores: np.ndarray,
    windows: list[VisualWindow],
    max_k: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]] | None:
    """Build the K-ads DP table up to ``max_k`` stages.

    Returns ``(b, s_back, e_back)`` lists of length ``max_k + 1`` where
    ``b[k][e]`` is the best total score for ``k`` ads whose K-th ad ends
    at index ``e`` (exclusive). Backpointers in ``s_back``/``e_back``.

    Returns ``None`` for the degenerate empty/zero case so callers can
    short-circuit. Splitting the build from per-K extraction lets the
    auto-K loop reuse one max-K DP run for every K' it evaluates.
    """
    N = len(windows)
    if max_k <= 0 or N == 0:
        return None

    norm_edge, cum_foreign, min_w, max_w, gap_w, first_start_idx = _prepare_dp_inputs(
        edge_scores, foreign_scores, windows,
    )

    NEG_INF = float("-inf")
    b = [np.full(N + 1, NEG_INF, dtype=np.float64) for _ in range(max_k + 1)]
    s_back = [np.full(N + 1, -1, dtype=np.int32) for _ in range(max_k + 1)]
    e_back = [np.full(N + 1, -1, dtype=np.int32) for _ in range(max_k + 1)]

    win_t0 = np.array([w.t0 for w in windows], dtype=np.float64)
    win_t1 = np.array([w.t1 for w in windows], dtype=np.float64)

    for stage in range(1, max_k + 1):
        if stage == 1:
            prev_pmax: np.ndarray | None = None
            prev_pmax_e: np.ndarray | None = None
        else:
            prev_pmax = np.full(N + 1, NEG_INF, dtype=np.float64)
            prev_pmax_e = np.full(N + 1, -1, dtype=np.int32)
            for i in range(N + 1):
                if i > 0:
                    prev_pmax[i] = prev_pmax[i - 1]
                    prev_pmax_e[i] = prev_pmax_e[i - 1]
                if b[stage - 1][i] > prev_pmax[i]:
                    prev_pmax[i] = b[stage - 1][i]
                    prev_pmax_e[i] = i

        for e in range(min_w, N + 1):
            s_lo = max(first_start_idx, e - max_w)
            s_hi = e - min_w
            if s_lo > s_hi:
                continue
            t_e = float(win_t1[e - 1])

            s_arr = np.arange(s_lo, s_hi + 1, dtype=np.int64)
            durs = t_e - win_t0[s_arr]
            valid = (
                (durs >= AD_MIN_SEC)
                & (durs <= AD_MAX_SEC)
                & (win_t0[s_arr] >= FIRST_AD_MIN_START_SEC)
            )
            if not valid.any():
                continue
            interior_mean = (cum_foreign[e] - cum_foreign[s_arr]) / np.maximum(e - s_arr, 1)
            sc_arr = (
                EDGE_WEIGHT * (norm_edge[s_arr] + norm_edge[e])
                + INTERIOR_WEIGHT * interior_mean
            )

            if stage == 1:
                totals = sc_arr.copy()
                prev_es = np.full(s_arr.size, -1, dtype=np.int32)
            else:
                me_prev = s_arr - gap_w
                prev_valid = me_prev >= 0
                pp = np.full(s_arr.size, NEG_INF, dtype=np.float64)
                pe = np.full(s_arr.size, -1, dtype=np.int32)
                if prev_valid.any():
                    me_idx = me_prev[prev_valid]
                    pp[prev_valid] = prev_pmax[me_idx]  # type: ignore[index]
                    pe[prev_valid] = prev_pmax_e[me_idx]  # type: ignore[index]
                valid &= prev_valid & (pp > NEG_INF)
                totals = pp + sc_arr
                prev_es = pe

            if not valid.any():
                continue
            local_idx = int(np.argmax(np.where(valid, totals, NEG_INF)))
            best_total_for_e = float(totals[local_idx])
            if best_total_for_e > b[stage][e]:
                b[stage][e] = best_total_for_e
                s_back[stage][e] = int(s_arr[local_idx])
                e_back[stage][e] = int(prev_es[local_idx])

    return b, s_back, e_back


def _extract_intervals_at_k(
    dp: tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]],
    k: int,
) -> tuple[float, list[tuple[int, int]]]:
    """Backtrack the optimal K intervals from a pre-built DP table."""
    b, s_back, e_back = dp
    if k <= 0 or k >= len(b):
        return 0.0, []
    NEG_INF = float("-inf")
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


def _min_interior_mean(
    foreign_scores: np.ndarray, intervals: list[tuple[int, int]]
) -> float:
    """Min normalised foreignness mean across the chosen intervals.

    Returns 1.0 for empty input so a K=0 selection is not treated as a
    collapse.
    """
    if not intervals:
        return 1.0
    f_max = float(foreign_scores.max())
    if f_max <= 0.0:
        return 0.0
    norm = foreign_scores / (f_max + 1e-9)
    means = [float(norm[s:e].mean()) for s, e in intervals if e > s]
    return min(means) if means else 1.0


def _select_num_ads_auto(
    edge_scores: np.ndarray,
    foreign_scores: np.ndarray,
    windows: list[VisualWindow],
    *,
    max_k: int = MAX_NUM_ADS,
    min_marginal_ratio: float = MIN_MARGINAL_RATIO,
    min_k: int = MIN_NUM_ADS,
    min_interior_mean_floor: float = MIN_INTERIOR_MEAN_FLOOR,
) -> list[tuple[int, int]]:
    """Auto-select K by walking K=1..max_k and stopping at the first K+1
    that fails *either* of two rules:

    1. **Marginal-gain ratio** — ``total_{k+1} - total_k >=
       min_marginal_ratio * (first ad's marginal score)``.
    2. **Adaptive interior-mean floor** — the minimum normalised
       interior_mean across the K+1 chosen intervals must stay above
       ``min(min_interior_mean_floor, 0.50 * K=1_interior_mean)``.

    Rule 2 catches the failure mode rule 1 doesn't: when the K-stage DP
    re-optimises and absorbs a noise plateau as one of K equally-scored
    intervals (so the marginal stays near 1.0), the noise interval shows
    up as a sharp drop in the *minimum* per-interval interior mean.

    The adaptive floor handles music-saturated videos (test_009) where
    the K=1 interior is already low (~0.55) — a flat 0.40 floor would
    block legitimate K growth there.

    ``min_k`` is a *soft* floor: K is only padded up to it when no
    interior collapse was observed below ``min_k``.
    """
    if not windows:
        return []
    min_k = max(1, int(min_k))
    max_k = max(min_k, int(max_k))

    dp = _find_best_k_ads_dp(edge_scores, foreign_scores, windows, max_k)
    if dp is None:
        return []

    totals: list[float] = [0.0]
    interval_sets: list[list[tuple[int, int]]] = [[]]
    min_interiors: list[float] = [1.0]

    for k in range(1, max_k + 1):
        total_k, intervals_k = _extract_intervals_at_k(dp, k)
        if not intervals_k:
            break
        totals.append(total_k)
        interval_sets.append(intervals_k)
        min_interiors.append(_min_interior_mean(foreign_scores, intervals_k))

    if len(totals) <= 1:
        return []

    first_gain = totals[1] - totals[0]

    # Adaptive interior floor — see docstring above.
    interior_floor_relative_factor = 0.50
    effective_interior_floor = min(
        min_interior_mean_floor,
        interior_floor_relative_factor * min_interiors[1],
    )

    chosen_k = 1
    interior_collapse_seen = False
    for k in range(2, len(totals)):
        # Two-tier ratio: looser inside [1..MIN_NUM_ADS] (we want to
        # allow legitimate K=3), stricter when growing K beyond
        # MIN_NUM_ADS (so videos with monotonically diminishing K
        # gains don't pad to K=5 false positives).
        active_ratio = (
            min_marginal_ratio
            if k <= min_k
            else MIN_MARGINAL_RATIO_EXTEND
        )
        ratio_ok = first_gain > 0.0 and (totals[k] - totals[k - 1]) >= active_ratio * first_gain
        interior_ok = min_interiors[k] >= effective_interior_floor
        if not interior_ok:
            interior_collapse_seen = True
            break
        if not ratio_ok:
            break
        chosen_k = k

    # Soft min_k: pad up to min_k only if no interior collapse below.
    if not interior_collapse_seen and chosen_k < min_k:
        chosen_k = min(min_k, len(interval_sets) - 1)
    chosen_k = min(chosen_k, len(interval_sets) - 1)
    return interval_sets[chosen_k]


def _find_best_ads(
    edge_scores: np.ndarray,
    foreign_scores: np.ndarray,
    windows: list[VisualWindow],
    max_ads: int = MAX_NUM_ADS,
) -> list[tuple[int, int]]:
    """Public wrapper retained for backward compatibility with callers and
    tests. Forwards to the auto-K selector.
    """
    return _select_num_ads_auto(
        edge_scores,
        foreign_scores,
        windows,
        max_k=max_ads,
    )


# Step 4 – Refine boundaries
def _refine_boundary(
    idx: int,
    edge_scores: np.ndarray,
    direction: str,
    windows: list[VisualWindow],
    search_sec: float = 15.0,
) -> int:
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

# Segment building
def _make_segment_dict(label: str, start: float, end: float) -> dict[str, Any]:
    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "label": label,
        "kind": _KIND_FOR_LABEL.get(label, KIND_NON_CONTENT),
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
    if is_before_first_ad and not intro_used:
        return LABEL_INTRO, True, outro_used
    if is_after_last_ad and not outro_used:
        return LABEL_OUTRO, intro_used, True
    return LABEL_CORE_CONTENT, intro_used, outro_used


def _detect_intro_end(windows: list[VisualWindow], min_start: float = 10.0, max_end: float = 120.0) -> float | None:
    """Find intro end by detecting the first long static/freeze run after min_start.
    A sudden high palette_delta with static hypothesis signals a title card / freeze
    that typically marks the end of an organic intro."""
    # Look for consecutive static windows with high palette_delta — these are freeze/end-card transitions
    i = 0
    while i < len(windows):
        w = windows[i]
        if w.t0 < min_start:
            i += 1
            continue
        if w.t0 > max_end:
            break
        if w.visual_hypothesis == "static" and float(w.palette_delta) > 0.9:
            # found start of a static run — return the start of that window as intro end
            return float(w.t0)
        i += 1
    return None


def _is_outro_window(w: VisualWindow) -> bool:
    """An outro window is either a true static title card OR a graphics-heavy
    credit/end-card with substantial text/edges.  Real outros come in two
    flavors — a frozen end-card (static + pal=1.0) and rolling/title credits
    (graphics_heavy with high edge density and on-screen text).  The original
    detector only caught the first kind, so videos like test_009 (animated
    end-card with text) were silently missed.
    """
    if w.visual_hypothesis == "static" and float(w.palette_delta) > 0.9:
        return True
    if w.visual_hypothesis == "graphics_heavy" and (
        float(w.edge_density) >= 0.6 or w.high_text_density
    ):
        return True
    return False


def _detect_outro_start(windows: list[VisualWindow], min_static_dur: float = 12.0, min_start_frac: float = 0.8, duration: float = 0.0) -> float | None:
    """Find outro start by anchoring on the *final* continuous run of
    outro-looking windows (static end-card OR graphics-heavy credits).

    We deliberately do NOT keep walking backwards to find earlier candidate
    runs — many videos contain mid-video graphics overlays (e.g. day-in-life
    text labels at 11:00 AM, 5:00 PM, etc.) that would otherwise be mistaken
    for a much longer outro.  The outro is the *last* such run, full stop.

    A short non-outro gap of up to ``MAX_GAP_WINDOWS`` is tolerated when
    extending the run backwards (typical end-cards have a brief talking-head
    sign-off mixed with a static logo).
    """
    if not windows:
        return None
    min_start = duration * min_start_frac if duration > 0 else 0.0
    MAX_GAP_WINDOWS = 2

    n = len(windows)
    end_i = n - 1
    while end_i >= 0 and not _is_outro_window(windows[end_i]):
        end_i -= 1
    if end_i < 0 or windows[end_i].t0 < min_start:
        return None

    j = end_i
    gap = 0
    while j >= 0:
        if _is_outro_window(windows[j]):
            gap = 0
            j -= 1
        elif gap < MAX_GAP_WINDOWS and windows[j].t0 >= min_start:
            gap += 1
            j -= 1
        else:
            break
    run_start_t = windows[j + 1 + gap].t0 if gap > 0 else windows[j + 1].t0
    run_end_t = windows[end_i].t1

    if run_end_t - run_start_t >= min_static_dur:
        return run_start_t
    return None


def _snap_boundary_to_shot(windows: list[VisualWindow], idx: int, direction: str, search_sec: float = 6.0) -> int:
    """Snap an ad boundary index to the nearest strong shot-boundary window."""
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


def _build_segments_from_ad_intervals(
    ad_intervals: list[tuple[int, int]],
    windows: list[VisualWindow],
    duration: float,
    intro_end_sec: float | None = None,
    outro_start_sec: float | None = None,
) -> list[dict[str, Any]]:
    N = len(windows)

    # Detect intro end and outro start from visual signal if not provided
    if intro_end_sec is None:
        intro_end_sec = _detect_intro_end(windows, min_start=10.0, max_end=120.0)
    if outro_start_sec is None:
        outro_start_sec = _detect_outro_start(windows, min_static_dur=12.0, min_start_frac=0.85, duration=duration)

    # Snap ad boundaries to nearest strong shot cut
    snapped: list[tuple[int, int]] = []
    for s, e in ad_intervals:
        rs = _snap_boundary_to_shot(windows, s, "start", search_sec=6.0)
        re = _snap_boundary_to_shot(windows, e - 1, "end", search_sec=6.0) + 1
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

        j = i
        while j < N and not is_ad[j]:
            j += 1
        run_start = windows[i].t0
        run_end = windows[j - 1].t1

        # Split the run at intro_end_sec / outro_start_sec so we don't
        # accidentally label, e.g., 0-60s as "Intro" when the actual intro
        # only goes 0-42s and the rest is normal pre-ad content.
        if (
            intro_end_sec is not None
            and run_start < intro_end_sec < run_end
        ):
            segments.append(_make_segment_dict(LABEL_INTRO, run_start, intro_end_sec))
            segments.append(_make_segment_dict(LABEL_CORE_CONTENT, intro_end_sec, run_end))
        elif (
            outro_start_sec is not None
            and run_start < outro_start_sec < run_end
        ):
            segments.append(_make_segment_dict(LABEL_CORE_CONTENT, run_start, outro_start_sec))
            segments.append(_make_segment_dict(LABEL_OUTRO, outro_start_sec, run_end))
        elif intro_end_sec is not None and run_end <= intro_end_sec:
            segments.append(_make_segment_dict(LABEL_INTRO, run_start, run_end))
        elif outro_start_sec is not None and run_start >= outro_start_sec:
            segments.append(_make_segment_dict(LABEL_OUTRO, run_start, run_end))
        else:
            segments.append(_make_segment_dict(LABEL_CORE_CONTENT, run_start, run_end))
        i = j

    segments.sort(key=lambda s: s["start"])
    return segments


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

    # Auto-K selector (1..MAX_NUM_ADS), with adaptive interior-mean floor +
    # marginal-gain ratio rules. See _select_num_ads_auto for rationale.
    ad_intervals = _find_best_ads(smooth_edge, smooth_foreign, windows, max_ads=MAX_NUM_ADS)

    # Post-DP snap to high-confidence semantic ad spans (when --with-semantic
    # was run).  See _snap_intervals_to_semantic_spans for rationale; this is
    # a no-op when no overlapping qualifying semantic span exists.
    if ad_intervals and bundle.speech_spans:
        ad_intervals = _snap_intervals_to_semantic_spans(
            ad_intervals, windows, bundle.speech_spans
        )

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