"""Audio modality analyzer.

Extracts per-window audio features from a video file and produces
``schemas.modality.AudioWindow`` records consumed by the fusion layer
(``fusion/fuse.py``).

Design choices
--------------
- Audio is demuxed by ``ffmpeg`` to 16 kHz mono PCM s16le, the same recipe used
  in ``scripts/ingest_example.sh``. Loaded with the standard-library ``wave``
  module so we avoid ``soundfile`` as a hard dependency.
- DSP is pure ``numpy`` + ``scipy.signal``; no librosa.
- Window cadence (``window_sec``) defaults to 1.0 second to match the visual
  pipeline so ``fuse_bundle_to_segments`` lines up windows by midpoint.
- For each window we emit, in addition to ``t0``/``t1``, the three extras the
  fusion layer reads:

      audio_label    : "silence" | "speech" | "music" | "mixed"
      energy_rms     : float in [0, 1]; fusion treats < 0.02 as inactivity
      anomaly_score  : float in [0, 1]; fusion treats > 0.75 as advertisement

  Plus auxiliary numeric features (``rms_db``, ``zcr``, ``spectral_centroid``,
  ``spectral_flatness``, ``spectral_rolloff``) that downstream tooling and
  evaluation can use without recomputing them.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import rfft
from scipy.fftpack import dct
from scipy.signal import get_window

from schemas.modality import AudioWindow


_DEFAULT_SAMPLE_RATE = 16_000
_FRAME_MS = 25.0
_HOP_MS = 10.0
_NUM_MEL_FILTERS = 26
_NUM_MFCC = 13
_PRE_EMPHASIS = 0.97
_SILENCE_DB = -40.0
_RMS_NORM_DB_FLOOR = -60.0
_RMS_NORM_DB_CEIL = 0.0
_AUDIO_LABELS = ("silence", "speech", "music", "mixed")


@dataclass
class _WindowFeatures:
    t0: float
    t1: float
    rms_linear: float
    rms_db: float
    zcr_mean: float
    zcr_var: float
    centroid_hz: float
    rolloff_hz: float
    flatness: float
    mfcc_mean: np.ndarray


def _resolve_ffmpeg_executable() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    repo_local = Path(__file__).resolve().parents[1] / "data" / "bin" / "ffmpeg.exe"
    if repo_local.is_file():
        return str(repo_local)

    return "ffmpeg"


def _ffmpeg_extract_wav(video_path: Path, wav_out: Path, sample_rate: int) -> None:
    command = [
        _resolve_ffmpeg_executable(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(wav_out),
    ]
    try:
        subprocess.run(command, check=True, timeout=600)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg is required to extract audio. Install it and ensure it is on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed to extract audio from {video_path}: {exc}") from exc


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        n_channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        n_frames = handle.getnframes()
        raw = handle.readframes(n_frames)
    if sample_width != 2:
        raise RuntimeError(
            f"Expected 16-bit PCM from ffmpeg, got sample width {sample_width} bytes."
        )
    pcm = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    samples = pcm.astype(np.float32) / 32768.0
    return samples, int(sample_rate)


def _frame_signal(signal: np.ndarray, frame_len: int, hop_len: int) -> np.ndarray:
    if signal.size < frame_len:
        padded = np.zeros(frame_len, dtype=signal.dtype)
        padded[: signal.size] = signal
        return padded[np.newaxis, :]
    n_frames = 1 + (signal.size - frame_len) // hop_len
    if n_frames < 1:
        n_frames = 1
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(n_frames, frame_len),
        strides=(signal.strides[0] * hop_len, signal.strides[0]),
        writeable=False,
    )
    return np.ascontiguousarray(frames)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(num_filters: int, n_fft: int, sample_rate: int) -> np.ndarray:
    low_mel = _hz_to_mel(np.array([0.0]))[0]
    high_mel = _hz_to_mel(np.array([sample_rate / 2.0]))[0]
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((num_filters, n_fft // 2 + 1), dtype=np.float32)
    for f_idx in range(num_filters):
        left = bin_indices[f_idx]
        center = bin_indices[f_idx + 1]
        right = bin_indices[f_idx + 2]
        if center == left:
            center = left + 1
        if right == center:
            right = center + 1
        right = min(right, filters.shape[1] - 1)
        center = min(center, filters.shape[1] - 1)
        left = min(left, filters.shape[1] - 1)
        for k in range(left, center):
            filters[f_idx, k] = (k - left) / max(1, center - left)
        for k in range(center, right):
            filters[f_idx, k] = (right - k) / max(1, right - center)
    return filters


def _spectral_features(
    frames: np.ndarray,
    sample_rate: int,
    mel_fb: np.ndarray,
) -> tuple[float, float, float, np.ndarray]:
    """Return centroid_hz, rolloff_hz, flatness, mfcc_mean (length _NUM_MFCC)."""
    if frames.size == 0:
        return 0.0, 0.0, 0.0, np.zeros(_NUM_MFCC, dtype=np.float32)

    n_fft = int(2 ** math.ceil(math.log2(frames.shape[1])))
    window = get_window("hann", frames.shape[1], fftbins=True).astype(np.float32)
    windowed = frames * window
    spectrum = np.abs(rfft(windowed, n=n_fft, axis=1)).astype(np.float32)
    power = (spectrum ** 2) / max(n_fft, 1)
    eps = 1e-10

    freqs = np.linspace(0.0, sample_rate / 2.0, spectrum.shape[1], dtype=np.float32)
    mag_sum = spectrum.sum(axis=1) + eps
    centroid_per_frame = (spectrum * freqs[np.newaxis, :]).sum(axis=1) / mag_sum
    centroid_hz = float(np.mean(centroid_per_frame))

    cumulative = np.cumsum(spectrum, axis=1)
    threshold = 0.95 * cumulative[:, -1:]
    rolloff_idx = np.argmax(cumulative >= threshold, axis=1)
    rolloff_hz = float(np.mean(freqs[rolloff_idx]))

    geo = np.exp(np.mean(np.log(spectrum + eps), axis=1))
    arith = np.mean(spectrum, axis=1) + eps
    flatness = float(np.mean(geo / arith))

    mel_energy = power @ mel_fb.T
    mel_log = np.log(mel_energy + eps)
    mfcc = dct(mel_log, type=2, axis=1, norm="ortho")[:, :_NUM_MFCC]
    mfcc_mean = mfcc.mean(axis=0).astype(np.float32)

    return centroid_hz, rolloff_hz, flatness, mfcc_mean


def _zcr_per_frame(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.zeros(0, dtype=np.float32)
    signs = np.sign(frames)
    signs[signs == 0] = 1
    crossings = np.abs(np.diff(signs, axis=1)) > 0
    return crossings.sum(axis=1).astype(np.float32) / max(1, frames.shape[1] - 1)


def _compute_window_features(
    samples: np.ndarray,
    sample_rate: int,
    window_sec: float,
    mel_fb: np.ndarray,
) -> list[_WindowFeatures]:
    if samples.size == 0:
        return []

    pre = np.empty_like(samples)
    pre[0] = samples[0]
    pre[1:] = samples[1:] - _PRE_EMPHASIS * samples[:-1]

    frame_len = max(1, int(round(_FRAME_MS * 1e-3 * sample_rate)))
    hop_len = max(1, int(round(_HOP_MS * 1e-3 * sample_rate)))
    samples_per_window = max(1, int(round(window_sec * sample_rate)))

    duration_sec = samples.size / float(sample_rate)
    n_windows = max(1, int(math.ceil(duration_sec / window_sec)))

    features: list[_WindowFeatures] = []
    for w in range(n_windows):
        start = w * samples_per_window
        end = min(samples.size, start + samples_per_window)
        if end - start < frame_len:
            t0 = start / sample_rate
            t1 = end / sample_rate if end > start else min(duration_sec, t0 + window_sec)
            features.append(
                _WindowFeatures(
                    t0=float(t0),
                    t1=float(t1),
                    rms_linear=0.0,
                    rms_db=_RMS_NORM_DB_FLOOR,
                    zcr_mean=0.0,
                    zcr_var=0.0,
                    centroid_hz=0.0,
                    rolloff_hz=0.0,
                    flatness=0.0,
                    mfcc_mean=np.zeros(_NUM_MFCC, dtype=np.float32),
                )
            )
            continue

        raw_segment = samples[start:end]
        rms_linear = float(np.sqrt(np.mean(raw_segment.astype(np.float64) ** 2)))
        rms_db = 20.0 * math.log10(max(rms_linear, 1e-6))

        pre_segment = pre[start:end]
        frames = _frame_signal(pre_segment, frame_len, hop_len)
        zcr_per_frame = _zcr_per_frame(frames)
        zcr_mean = float(np.mean(zcr_per_frame)) if zcr_per_frame.size else 0.0
        zcr_var = float(np.var(zcr_per_frame)) if zcr_per_frame.size else 0.0
        centroid_hz, rolloff_hz, flatness, mfcc_mean = _spectral_features(
            frames, sample_rate, mel_fb
        )

        features.append(
            _WindowFeatures(
                t0=float(start / sample_rate),
                t1=float(end / sample_rate),
                rms_linear=rms_linear,
                rms_db=float(rms_db),
                zcr_mean=zcr_mean,
                zcr_var=zcr_var,
                centroid_hz=float(centroid_hz),
                rolloff_hz=float(rolloff_hz),
                flatness=float(flatness),
                mfcc_mean=mfcc_mean,
            )
        )
    return features


def _normalize_rms(rms_db_values: list[float]) -> list[float]:
    out: list[float] = []
    for db in rms_db_values:
        if not math.isfinite(db):
            out.append(0.0)
            continue
        clamped = max(_RMS_NORM_DB_FLOOR, min(_RMS_NORM_DB_CEIL, db))
        out.append((clamped - _RMS_NORM_DB_FLOOR) / (_RMS_NORM_DB_CEIL - _RMS_NORM_DB_FLOOR))
    return out


def _classify_audio_label(features: _WindowFeatures, energy_rms_norm: float) -> str:
    """Lightweight rule-based silence/speech/music/mixed classifier."""
    if features.rms_db <= _SILENCE_DB or energy_rms_norm < 0.02:
        return "silence"

    speech_band = 800.0 <= features.centroid_hz <= 3800.0
    speech_zcr = 0.05 <= features.zcr_mean <= 0.25
    speech_flatness = features.flatness < 0.20
    speech_score = (
        (1.0 if speech_band else 0.0)
        + (1.0 if speech_zcr else 0.0)
        + (1.0 if speech_flatness else 0.0)
        + min(1.0, features.zcr_var * 200.0)
    )

    music_centroid = features.centroid_hz > 2000.0
    music_flatness = 0.10 <= features.flatness <= 0.55
    music_steady = features.zcr_var < 0.005
    music_score = (
        (1.0 if music_centroid else 0.0)
        + (1.0 if music_flatness else 0.0)
        + (1.5 if music_steady else 0.0)
    )

    if speech_score >= max(2.5, music_score + 0.5):
        return "speech"
    if music_score >= max(2.5, speech_score + 0.5):
        return "music"
    return "mixed"


def _robust_normalize(values: np.ndarray, low_q: float = 0.1, high_q: float = 0.9) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(np.quantile(values, low_q))
    hi = float(np.quantile(values, high_q))
    if hi <= lo + 1e-9:
        return np.full_like(values, 0.5, dtype=np.float32)
    scaled = (values - lo) / (hi - lo)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def _anomaly_scores(features: list[_WindowFeatures]) -> list[float]:
    """Distance of each window's MFCC mean from the global median, robustly normalized."""
    if not features:
        return []
    mfcc_matrix = np.stack([f.mfcc_mean for f in features], axis=0).astype(np.float32)
    median = np.median(mfcc_matrix, axis=0, keepdims=True)
    mad = np.median(np.abs(mfcc_matrix - median), axis=0, keepdims=True) + 1e-6
    standardized = (mfcc_matrix - median) / (1.4826 * mad)
    raw = np.linalg.norm(standardized, axis=1)
    normalized = _robust_normalize(raw, low_q=0.5, high_q=0.95)
    return [float(v) for v in normalized.tolist()]


def build_audio_windows(
    samples: np.ndarray,
    sample_rate: int,
    window_sec: float = 1.0,
) -> list[AudioWindow]:
    """Build the public ``AudioWindow`` list from raw mono PCM samples."""
    n_fft = int(2 ** math.ceil(math.log2(max(2, int(round(_FRAME_MS * 1e-3 * sample_rate))))))
    mel_fb = _mel_filterbank(_NUM_MEL_FILTERS, n_fft, sample_rate)

    features = _compute_window_features(samples, sample_rate, window_sec, mel_fb)
    rms_norm = _normalize_rms([f.rms_db for f in features])
    anomalies = _anomaly_scores(features)

    audio_windows: list[AudioWindow] = []
    for idx, feat in enumerate(features):
        energy = rms_norm[idx] if idx < len(rms_norm) else 0.0
        anomaly = anomalies[idx] if idx < len(anomalies) else 0.0
        label = _classify_audio_label(feat, energy)
        audio_windows.append(
            AudioWindow(
                t0=feat.t0,
                t1=feat.t1,
                audio_label=label,
                energy_rms=float(energy),
                anomaly_score=float(anomaly),
                rms_db=float(feat.rms_db),
                zcr=float(feat.zcr_mean),
                zcr_var=float(feat.zcr_var),
                spectral_centroid=float(feat.centroid_hz),
                spectral_rolloff=float(feat.rolloff_hz),
                spectral_flatness=float(feat.flatness),
            )
        )
    return audio_windows


def analyze_audio(
    video_path: Path | str | None = None,
    *,
    window_sec: float = 1.0,
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
    keep_wav: Path | None = None,
    audio_in: Path | str | None = None,
) -> tuple[list[AudioWindow], float]:
    """Return per-window features and duration for a video or pre-extracted WAV.

    Provide exactly one of ``video_path`` or ``audio_in``. ``audio_in`` skips
    ffmpeg entirely and reads a mono PCM WAV directly, matching the planning
    doc's pipeline step 1 ("Frames + audio file") so step 2 can be re-run
    without re-demuxing.
    """
    if (video_path is None) == (audio_in is None):
        raise ValueError("Provide exactly one of video_path or audio_in.")

    if audio_in is not None:
        wav_path = Path(audio_in).expanduser().resolve(strict=False)
        if not wav_path.is_file():
            raise FileNotFoundError(str(wav_path))
        samples, sr = _read_wav_mono(wav_path)
    else:
        path = Path(video_path).expanduser().resolve(strict=False)
        if not path.is_file():
            raise FileNotFoundError(str(path))

        if keep_wav is not None:
            wav_path = Path(keep_wav).expanduser().resolve(strict=False)
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            _ffmpeg_extract_wav(path, wav_path, sample_rate)
            samples, sr = _read_wav_mono(wav_path)
        else:
            with tempfile.TemporaryDirectory(prefix="audio_analyze_") as tmpdir:
                wav_path = Path(tmpdir) / f"{path.stem}.wav"
                _ffmpeg_extract_wav(path, wav_path, sample_rate)
                samples, sr = _read_wav_mono(wav_path)

    duration_sec = float(samples.size) / float(sr) if sr > 0 else 0.0
    audio_windows = build_audio_windows(samples, sr, window_sec=window_sec)
    return audio_windows, duration_sec


def audio_windows_to_payload(windows: list[AudioWindow]) -> list[dict[str, Any]]:
    return [w.model_dump() for w in windows]


def write_audio_track_json(
    windows: list[AudioWindow],
    out_path: Path,
    *,
    video_path: Path | str = "",
    duration_sec: float = 0.0,
    window_sec: float = 1.0,
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "modality": "audio",
        "video_path": str(video_path),
        "duration_sec": float(duration_sec),
        "window_sec": float(window_sec),
        "audio_windows": audio_windows_to_payload(windows),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
