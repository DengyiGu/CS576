"""Tests for the audio modality (no ffmpeg / no real video required)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from audio.analyze import (
    audio_windows_to_payload,
    build_audio_windows,
    write_audio_track_json,
)
from schemas.modality import AnalysisBundle, AudioWindow


_SR = 16_000


def _silence(duration_sec: float, sr: int = _SR) -> np.ndarray:
    return np.zeros(int(duration_sec * sr), dtype=np.float32)


def _white_noise(duration_sec: float, amplitude: float = 0.3, sr: int = _SR) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return (rng.standard_normal(int(duration_sec * sr)).astype(np.float32) * amplitude).clip(-1.0, 1.0)


def _sine(duration_sec: float, freq_hz: float, amplitude: float = 0.4, sr: int = _SR) -> np.ndarray:
    t = np.arange(int(duration_sec * sr), dtype=np.float32) / sr
    return (amplitude * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _formant_speech_like(duration_sec: float, sr: int = _SR) -> np.ndarray:
    """Synthetic formant-style signal: sum of low harmonics with envelope, ZCR resembling voice."""
    f0 = 140.0
    formants = [500.0, 1500.0, 2500.0]
    t = np.arange(int(duration_sec * sr), dtype=np.float32) / sr
    signal = np.zeros_like(t)
    for f in formants:
        signal += np.sin(2.0 * np.pi * f * t) * 0.25
    pitch = 0.4 * np.sin(2.0 * np.pi * f0 * t)
    envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * 4.0 * t)
    return ((signal + pitch) * envelope * 0.5).astype(np.float32)


def test_silence_window_marked_silent_and_low_energy() -> None:
    samples = _silence(3.0)
    windows = build_audio_windows(samples, _SR, window_sec=1.0)
    assert len(windows) >= 3
    for w in windows[:3]:
        assert w.model_extra is not None
        assert w.model_extra["audio_label"] == "silence"
        assert w.model_extra["energy_rms"] < 0.02


def test_window_timestamps_and_count() -> None:
    samples = _white_noise(4.0)
    windows = build_audio_windows(samples, _SR, window_sec=1.0)
    assert len(windows) == 4
    for idx, w in enumerate(windows):
        assert w.t0 == pytest.approx(float(idx), abs=1e-3)
        assert w.t1 == pytest.approx(float(idx + 1), abs=1e-3)


def test_silence_to_loud_anomaly_score() -> None:
    quiet = _silence(2.0)
    loud = _white_noise(2.0, amplitude=0.8)
    samples = np.concatenate([quiet, loud, quiet])
    windows = build_audio_windows(samples, _SR, window_sec=1.0)
    energies = [float(w.model_extra["energy_rms"]) for w in windows if w.model_extra is not None]
    silence_energies = energies[:2] + energies[-2:]
    loud_energies = energies[2:4]
    assert max(silence_energies) < 0.05
    assert min(loud_energies) > 0.5


def test_speech_like_window_classified_as_speech_or_mixed() -> None:
    samples = _formant_speech_like(3.0)
    windows = build_audio_windows(samples, _SR, window_sec=1.0)
    labels = [w.model_extra["audio_label"] for w in windows if w.model_extra is not None]
    assert any(label in ("speech", "music", "mixed") for label in labels)
    assert all(label != "silence" for label in labels)


def test_audio_windows_serialize_into_analysis_bundle() -> None:
    samples = np.concatenate([_silence(1.0), _formant_speech_like(2.0), _white_noise(1.0)])
    windows = build_audio_windows(samples, _SR, window_sec=1.0)

    bundle = AnalysisBundle(
        video_path="/tmp/synthetic.mp4",
        duration_sec=4.0,
        visual=None,
        audio_windows=windows,
        speech_spans=[],
    )
    dumped = json.loads(bundle.model_dump_json())
    restored = AnalysisBundle.model_validate(dumped)
    assert len(restored.audio_windows) == len(windows)
    for original, round_trip in zip(windows, restored.audio_windows):
        assert original.model_extra is not None
        assert round_trip.model_extra is not None
        assert round_trip.model_extra["audio_label"] == original.model_extra["audio_label"]
        assert round_trip.model_extra["energy_rms"] == pytest.approx(
            original.model_extra["energy_rms"], abs=1e-6
        )


def test_write_audio_track_json_roundtrip(tmp_path: Path) -> None:
    samples = _white_noise(2.0)
    windows = build_audio_windows(samples, _SR, window_sec=1.0)
    out = tmp_path / "audio_track.json"
    write_audio_track_json(
        windows,
        out,
        video_path="/tmp/synthetic.mp4",
        duration_sec=2.0,
        window_sec=1.0,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["modality"] == "audio"
    assert payload["window_sec"] == pytest.approx(1.0)
    assert len(payload["audio_windows"]) == len(windows)
    assert "audio_label" in payload["audio_windows"][0]
    assert "energy_rms" in payload["audio_windows"][0]
    assert "anomaly_score" in payload["audio_windows"][0]


def test_audio_window_carries_required_extras_for_fusion() -> None:
    samples = _white_noise(2.0)
    windows = build_audio_windows(samples, _SR, window_sec=1.0)
    payload = audio_windows_to_payload(windows)
    required_keys = {"audio_label", "energy_rms", "anomaly_score"}
    for entry in payload:
        assert required_keys.issubset(entry.keys())
        assert entry["audio_label"] in ("silence", "speech", "music", "mixed")
        assert 0.0 <= entry["energy_rms"] <= 1.0
        assert 0.0 <= entry["anomaly_score"] <= 1.0


def test_audio_window_schema_accepts_extras() -> None:
    window = AudioWindow(
        t0=0.0,
        t1=1.0,
        audio_label="silence",
        energy_rms=0.01,
        anomaly_score=0.0,
    )
    dumped = window.model_dump()
    assert dumped["audio_label"] == "silence"
    assert dumped["energy_rms"] == pytest.approx(0.01)
