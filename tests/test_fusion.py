from __future__ import annotations

import pytest

from schemas.modality import AnalysisBundle, AudioWindow, SpeechSpan, VisualTrack, VisualWindow
from fusion.fuse import (
    _compute_spectral_flatness_baseline,
    _compute_transcript_density_baseline,
    _compute_yamnet_baselines,
    _count_brand_hits,
    _count_lexicon_hits,
    _has_sponsorship_phrase,
    _loudness_jump_score,
    _speech_text_ad_signal,
    _spectral_flatness_score,
    _transcript_density_score,
    _yamnet_music_score,
    _yamnet_nonspeech_score,
    fuse_bundle_to_segments,
)


def _window_is_inside_any(t0: float, t1: float, intervals: list[tuple[float, float]]) -> bool:
    mid = 0.5 * (t0 + t1)
    return any(start <= mid < end for start, end in intervals)


def _synthetic_bundle_with_ads(
    intervals: list[tuple[float, float]],
    *,
    duration: float = 360.0,
    window_sec: float = 1.0,
    subtle_palette: bool = False,
    include_audio: bool = True,
) -> AnalysisBundle:
    windows: list[VisualWindow] = []
    audio: list[AudioWindow] = []
    t = 0.0
    while t < duration:
        t1 = min(duration, t + window_sec)
        is_ad = _window_is_inside_any(t, t1, intervals)
        is_boundary = any(abs(t - start) < 1e-6 or abs(t - end) < 1e-6 for start, end in intervals)
        windows.append(
            VisualWindow(
                t0=t,
                t1=t1,
                motion_score=0.35 if is_ad else 0.12,
                luminance_mean=0.45,
                edge_density=0.55 if is_ad else 0.20,
                palette_delta=(0.12 if subtle_palette else (0.85 if is_ad else 0.05)),
                shot_boundary_near=is_boundary,
                shot_boundary_distance_sec=0.0 if is_boundary else None,
                high_text_density=is_ad,
                visual_hypothesis="graphics_heavy" if is_ad else "dynamic_talk",
                hypothesis_confidence=0.9 if is_ad else 0.5,
            )
        )
        if include_audio:
            audio.append(
                AudioWindow(
                    t0=t,
                    t1=t1,
                    anomaly_score=0.90 if is_ad else 0.05,
                    energy_rms=0.05,
                )
            )
        t = t1

    return AnalysisBundle(
        video_path="synthetic.mp4",
        duration_sec=duration,
        visual=VisualTrack(
            video_path="synthetic.mp4",
            duration_sec=duration,
            native_fps=30.0,
            fps_sampled=1.0,
            frame_stride=30,
            video_width=640,
            video_height=360,
            window_sec=window_sec,
            windows=windows,
        ),
        audio_windows=audio,
        speech_spans=[],
    )


def test_fusion_keeps_full_ad_blocks_when_interior_signal_stays_high() -> None:
    reference_ads = [(50.0, 110.0), (180.0, 210.0), (290.0, 330.0)]
    bundle = _synthetic_bundle_with_ads(reference_ads)

    segments = fuse_bundle_to_segments(bundle)

    predicted_ads = [s for s in segments if s["label"] == "Advertisement"]
    assert len(predicted_ads) == 3
    for pred, ref in zip(predicted_ads, reference_ads):
        overlap = max(0.0, min(pred["end"], ref[1]) - max(pred["start"], ref[0]))
        ref_duration = ref[1] - ref[0]
        assert overlap / ref_duration >= 0.80


def test_fusion_uses_visual_semantics_when_palette_delta_is_subtle() -> None:
    reference_ads = [(50.0, 110.0), (180.0, 210.0), (290.0, 330.0)]
    bundle = _synthetic_bundle_with_ads(
        reference_ads,
        subtle_palette=True,
        include_audio=False,
    )

    segments = fuse_bundle_to_segments(bundle)

    predicted_ads = [s for s in segments if s["label"] == "Advertisement"]
    assert len(predicted_ads) == 3
    for pred, ref in zip(predicted_ads, reference_ads):
        overlap = max(0.0, min(pred["end"], ref[1]) - max(pred["start"], ref[0]))
        ref_duration = ref[1] - ref[0]
        assert overlap / ref_duration >= 0.70


# ---------------------------------------------------------------------------
# Configurable ad count (num_ads parameter / auto-K)
# ---------------------------------------------------------------------------

def test_fusion_recovers_four_ads_when_num_ads_is_4() -> None:
    """The DP must scale to K=4 ads when explicitly requested."""
    reference_ads = [
        (50.0, 110.0), (180.0, 240.0), (320.0, 380.0), (460.0, 520.0),
    ]
    bundle = _synthetic_bundle_with_ads(reference_ads, duration=600.0)

    segments = fuse_bundle_to_segments(bundle, num_ads=4)

    predicted_ads = [s for s in segments if s["label"] == "Advertisement"]
    assert len(predicted_ads) == 4
    for pred, ref in zip(predicted_ads, reference_ads):
        overlap = max(0.0, min(pred["end"], ref[1]) - max(pred["start"], ref[0]))
        ref_duration = ref[1] - ref[0]
        assert overlap / ref_duration >= 0.80


def test_fusion_recovers_one_ad_when_num_ads_is_1() -> None:
    """Forcing K=1 must return exactly one ad covering the strongest interval."""
    reference_ads = [(80.0, 160.0)]
    bundle = _synthetic_bundle_with_ads(reference_ads, duration=300.0)

    segments = fuse_bundle_to_segments(bundle, num_ads=1)

    predicted_ads = [s for s in segments if s["label"] == "Advertisement"]
    assert len(predicted_ads) == 1
    pred = predicted_ads[0]
    ref = reference_ads[0]
    overlap = max(0.0, min(pred["end"], ref[1]) - max(pred["start"], ref[0]))
    assert overlap / (ref[1] - ref[0]) >= 0.80


def test_fusion_auto_k_picks_four_on_four_ad_bundle() -> None:
    """num_ads=None (auto) should land on K=4 when there are 4 strong ads."""
    reference_ads = [
        (50.0, 110.0), (180.0, 240.0), (320.0, 380.0), (460.0, 520.0),
    ]
    bundle = _synthetic_bundle_with_ads(reference_ads, duration=600.0)

    segments = fuse_bundle_to_segments(bundle, num_ads=None, max_num_ads=6)

    predicted_ads = [s for s in segments if s["label"] == "Advertisement"]
    assert len(predicted_ads) == 4


def test_fusion_auto_k_stays_at_three_on_three_ad_bundle() -> None:
    """Auto-K should not over-detect: a 3-ad bundle stays at K=3."""
    reference_ads = [(50.0, 110.0), (180.0, 210.0), (290.0, 330.0)]
    bundle = _synthetic_bundle_with_ads(reference_ads, duration=420.0)

    segments = fuse_bundle_to_segments(bundle, num_ads=None, max_num_ads=6)

    predicted_ads = [s for s in segments if s["label"] == "Advertisement"]
    assert len(predicted_ads) == 3


# ---------------------------------------------------------------------------
# Brand matching with word boundaries
# ---------------------------------------------------------------------------

def test_brand_match_uses_word_boundaries_no_substring_false_positives() -> None:
    # 'apple' must not fire on 'applesauce'; 'max' must not fire on 'climaxed';
    # 'discover' must not fire on 'discovered'; 'coke' must not fire on 'jacoke'.
    safe, ambig, distinct = _count_brand_hits(
        "applesauce climaxed discovered jacoke offhand naturopath"
    )
    assert distinct == []
    assert safe == 0
    assert ambig == 0


def test_brand_match_word_boundary_positive_cases() -> None:
    safe, ambig, _ = _count_brand_hits("i love doritos and hellofresh today")
    assert safe >= 2  # both are non-ambiguous
    assert ambig == 0


def test_ambiguous_brand_alone_does_not_fire_speech_signal() -> None:
    # 'discover' and 'apple' are both in the ambiguous list; on their own
    # the word should not push the score above zero.
    spans = [SpeechSpan(t0=10.0, t1=12.0, text="to discover unexpected wonders")]
    assert _speech_text_ad_signal(10.0, 12.0, spans) == 0.0
    spans = [SpeechSpan(t0=10.0, t1=12.0, text="i bit into the apple")]
    assert _speech_text_ad_signal(10.0, 12.0, spans) == 0.0


def test_safe_brand_alone_fires_low_signal() -> None:
    spans = [SpeechSpan(t0=10.0, t1=12.0, text="want a doritos")]
    score = _speech_text_ad_signal(10.0, 12.0, spans)
    assert 0.40 <= score <= 0.55  # safe brand single-hit tier


# ---------------------------------------------------------------------------
# TV-ad lexicon
# ---------------------------------------------------------------------------

def test_lexicon_imperative_alone_fires() -> None:
    n_cat, by_cat = _count_lexicon_hits("call now to order today")
    assert n_cat >= 1
    assert "tv_ad_imperative" in by_cat


def test_lexicon_compliance_phrase_fires_strongly_via_text_signal() -> None:
    # Compliance disclaimers ("side effects may include") essentially never
    # appear in non-ad speech, so two-category language should hit the
    # multi-category tier.
    spans = [
        SpeechSpan(
            t0=100.0,
            t1=115.0,
            text=(
                "side effects may include nausea. ask your doctor "
                "if it is right for you."
            ),
        )
    ]
    score = _speech_text_ad_signal(100.0, 115.0, spans)
    assert score >= 0.85  # 2+ categories tier


def test_sponsorship_phrase_still_dominates() -> None:
    spans = [SpeechSpan(t0=10.0, t1=12.0, text="this episode is brought to you by")]
    assert _has_sponsorship_phrase("this episode is brought to you by")
    assert _speech_text_ad_signal(10.0, 12.0, spans) >= 0.95


# ---------------------------------------------------------------------------
# Transcript-density-drop helper
# ---------------------------------------------------------------------------

def test_transcript_density_baseline_uses_global_average() -> None:
    # 600 total chars over 60 seconds -> 10 chars/s baseline.
    spans = [
        SpeechSpan(t0=0.0, t1=10.0, text="x" * 100),
        SpeechSpan(t0=20.0, t1=30.0, text="y" * 200),
        SpeechSpan(t0=50.0, t1=60.0, text="z" * 300),
    ]
    baseline = _compute_transcript_density_baseline(spans, duration_sec=60.0)
    assert 9.0 <= baseline <= 11.0


def test_transcript_density_score_high_when_window_is_quiet() -> None:
    # Steady speech everywhere except a 30 s gap.
    spans: list[SpeechSpan] = []
    for start in range(0, 240, 5):
        spans.append(SpeechSpan(t0=float(start), t1=float(start + 4), text="x" * 40))
    # Drop a quiet stretch at [120, 150].
    spans = [s for s in spans if not (120.0 <= s.t0 < 150.0)]
    baseline = _compute_transcript_density_baseline(spans, duration_sec=240.0)
    quiet = _transcript_density_score(125.0, 145.0, spans, baseline_chars_per_sec=baseline)
    busy = _transcript_density_score(20.0, 40.0, spans, baseline_chars_per_sec=baseline)
    assert quiet >= 0.6
    assert busy <= 0.2


# ---------------------------------------------------------------------------
# Loudness-jump helper
# ---------------------------------------------------------------------------

def _audio_steps(low_db: float, high_db: float, switch_sec: float, *, duration_sec: float = 60.0) -> list[AudioWindow]:
    out: list[AudioWindow] = []
    for t in range(int(duration_sec)):
        rms_db = high_db if t >= switch_sec else low_db
        out.append(AudioWindow(t0=float(t), t1=float(t + 1), rms_db=float(rms_db)))
    return out


def test_loudness_jump_high_on_8db_step() -> None:
    audio = _audio_steps(low_db=-30.0, high_db=-22.0, switch_sec=30.0)
    score = _loudness_jump_score(30.0, audio, half_sec=10.0, edge_skip_sec=1.0)
    # 8 dB step is the reference -> should land near 1.0
    assert 0.85 <= score <= 1.0


def test_loudness_jump_low_when_no_step() -> None:
    audio = _audio_steps(low_db=-30.0, high_db=-30.0, switch_sec=30.0)
    score = _loudness_jump_score(30.0, audio, half_sec=10.0, edge_skip_sec=1.0)
    assert score <= 0.05


def test_loudness_jump_zero_when_no_data_around_boundary() -> None:
    # Boundary far past the audio — both sides empty.
    audio = _audio_steps(low_db=-30.0, high_db=-22.0, switch_sec=30.0, duration_sec=60.0)
    assert _loudness_jump_score(500.0, audio) == 0.0


# ---------------------------------------------------------------------------
# Spectral-flatness helpers
# ---------------------------------------------------------------------------

def _flatness_track(values: list[float]) -> list[AudioWindow]:
    return [
        AudioWindow(t0=float(i), t1=float(i + 1), spectral_flatness=float(v))
        for i, v in enumerate(values)
    ]


def test_spectral_flatness_baseline_returns_per_video_median() -> None:
    audio = _flatness_track([0.10, 0.20, 0.30, 0.40, 0.50])
    assert _compute_spectral_flatness_baseline(audio) == 0.30


def test_spectral_flatness_baseline_returns_none_when_feature_missing() -> None:
    audio = [
        AudioWindow(t0=0.0, t1=1.0),
        AudioWindow(t0=1.0, t1=2.0, anomaly_score=0.5),
    ]
    assert _compute_spectral_flatness_baseline(audio) is None


def test_spectral_flatness_score_zero_when_at_or_below_baseline() -> None:
    assert _spectral_flatness_score(0.30, 0.30) == 0.0
    assert _spectral_flatness_score(0.20, 0.30) == 0.0


def test_spectral_flatness_score_caps_at_one_for_large_excess() -> None:
    # delta = 0.30 with default scale 0.10 -> raw 3.0, clipped to 1.0.
    assert _spectral_flatness_score(0.60, 0.30) == 1.0


def test_spectral_flatness_score_scales_linearly_below_cap() -> None:
    # delta = 0.05, scale = 0.10 -> 0.5 (allow float imprecision).
    assert _spectral_flatness_score(0.35, 0.30) == pytest.approx(0.5)


def test_spectral_flatness_score_returns_zero_on_missing_inputs() -> None:
    assert _spectral_flatness_score(None, 0.30) == 0.0
    assert _spectral_flatness_score(0.30, None) == 0.0
    assert _spectral_flatness_score(None, None) == 0.0


# ---------------------------------------------------------------------------
# YAMNet helpers
# ---------------------------------------------------------------------------

def _yamnet_track(
    music_values: list[float] | None = None,
    speech_values: list[float] | None = None,
) -> list[AudioWindow]:
    n = len(music_values or speech_values or [])
    out: list[AudioWindow] = []
    for i in range(n):
        kwargs: dict[str, float] = {}
        if music_values is not None:
            kwargs["yamnet_music_score"] = float(music_values[i])
        if speech_values is not None:
            kwargs["yamnet_speech_score"] = float(speech_values[i])
        out.append(AudioWindow(t0=float(i), t1=float(i + 1), **kwargs))
    return out


def test_yamnet_baselines_use_per_video_medians() -> None:
    audio = _yamnet_track(
        music_values=[0.0, 0.1, 0.2, 0.3, 0.4],
        speech_values=[0.9, 0.8, 0.7, 0.6, 0.5],
    )
    music_med, speech_med = _compute_yamnet_baselines(audio)
    assert music_med == 0.2
    assert speech_med == 0.7


def test_yamnet_baselines_return_none_when_features_missing() -> None:
    audio = [AudioWindow(t0=0.0, t1=1.0)]
    assert _compute_yamnet_baselines(audio) == (None, None)


def test_yamnet_music_score_zero_at_or_below_baseline() -> None:
    assert _yamnet_music_score(0.30, 0.30) == 0.0
    assert _yamnet_music_score(0.20, 0.30) == 0.0


def test_yamnet_music_score_caps_at_one_for_large_excess() -> None:
    # delta = 0.50 with default scale 0.20 -> raw 2.5, clipped to 1.0.
    assert _yamnet_music_score(0.85, 0.35) == 1.0


def test_yamnet_music_score_scales_linearly_below_cap() -> None:
    # delta = 0.10, scale = 0.20 -> 0.5
    assert _yamnet_music_score(0.40, 0.30) == pytest.approx(0.5)


def test_yamnet_nonspeech_score_fires_when_speech_below_baseline() -> None:
    # speech below baseline -> ad-like silence/music. delta = 0.30 - 0.10 = 0.20.
    # scale = 0.30 -> raw 0.667, clipped.
    assert _yamnet_nonspeech_score(0.10, 0.30) == pytest.approx(0.20 / 0.30)


def test_yamnet_nonspeech_score_zero_when_speech_at_or_above_baseline() -> None:
    assert _yamnet_nonspeech_score(0.30, 0.30) == 0.0
    assert _yamnet_nonspeech_score(0.40, 0.30) == 0.0


def test_yamnet_helpers_return_zero_on_missing_inputs() -> None:
    assert _yamnet_music_score(None, 0.30) == 0.0
    assert _yamnet_music_score(0.30, None) == 0.0
    assert _yamnet_nonspeech_score(None, 0.30) == 0.0
    assert _yamnet_nonspeech_score(0.30, None) == 0.0
