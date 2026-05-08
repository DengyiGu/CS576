from __future__ import annotations

from schemas.modality import AnalysisBundle, AudioWindow, VisualTrack, VisualWindow
from fusion.fuse import fuse_bundle_to_segments


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
