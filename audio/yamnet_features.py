"""YAMNet audio-event features for the fusion pipeline.

YAMNet (Hershey et al., AudioSet) is a MobileNetV1 audio classifier that
labels 0.96-second windows with 521 AudioSet event categories. We use a
pre-exported ONNX build that takes a raw 16 kHz mono float32 waveform
(the same audio we already extract for ``audio/analyze.py`` and
faster-whisper). The model is downloaded once into ``audio/models/`` —
see ``audio/models/yamnet.onnx`` and ``audio/models/yamnet_class_map.csv``.

We only persist a small set of derived per-window scores rather than the
full 521-class probability matrix; the rest of the pipeline is built
around interpretable scalar cues:

  yamnet_music_score    : max over {Music, Background music, Theme music,
                           Jingle (music), Soundtrack music} — the broad
                           "is this a musical bed" indicator that fires
                           on broadcast-TV ads.
  yamnet_jingle_score   : Jingle (music) — strong, narrow ad cue.
  yamnet_theme_score    : Theme music — strong, narrow ad cue.
  yamnet_speech_score   : Speech — useful inverse signal (some ads switch
                           to wordless music; some content is dialog-
                           dense, raising speech).

YAMNet emits one prediction per frame at a hop of 0.48 s; we average the
frames that fall inside each ``AudioWindow``'s ``[t0, t1]`` to align with
the rest of the pipeline's 1 s cadence.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

# YAMNet hard-coded constants from Google's TF model card.
YAMNET_SAMPLE_RATE = 16_000
YAMNET_FRAME_SEC = 0.96
YAMNET_HOP_SEC = 0.48

# Per the AudioSet ontology (see audio/models/yamnet_class_map.csv):
#   132 Music
#   262 Background music
#   263 Theme music
#   264 Jingle (music)
#   265 Soundtrack music
#   0   Speech
_MUSIC_CLASS_IDS = (132, 262, 263, 264, 265)
_JINGLE_CLASS_ID = 264
_THEME_CLASS_ID = 263
_SPEECH_CLASS_ID = 0

# Output keys persisted onto AudioWindow.model_extra.
KEY_MUSIC = "yamnet_music_score"
KEY_JINGLE = "yamnet_jingle_score"
KEY_THEME = "yamnet_theme_score"
KEY_SPEECH = "yamnet_speech_score"
ALL_YAMNET_KEYS = (KEY_MUSIC, KEY_JINGLE, KEY_THEME, KEY_SPEECH)


def default_model_dir() -> Path:
    return Path(__file__).resolve().parent / "models"


def default_model_path() -> Path:
    return default_model_dir() / "yamnet.onnx"


def default_class_map_path() -> Path:
    return default_model_dir() / "yamnet_class_map.csv"


_SESSION_CACHE: dict[str, "object"] = {}


def _load_session(model_path: Path):
    """Lazy singleton ONNXRuntime session keyed by absolute path.

    Loading the 16 MB model takes ~0.5 s; cache it so repeated calls in
    the same process (e.g. running over multiple videos) only pay it once.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for YAMNet. It should already be on the "
            "machine via faster-whisper's deps; if not run "
            "`python -m pip install onnxruntime`."
        ) from exc

    abs_path = str(model_path.resolve())
    cached = _SESSION_CACHE.get(abs_path)
    if cached is not None:
        return cached
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YAMNet ONNX model not found at {model_path}. See audio/yamnet_features.py "
            "module docstring for download instructions."
        )
    session = ort.InferenceSession(abs_path, providers=["CPUExecutionProvider"])
    _SESSION_CACHE[abs_path] = session
    return session


def load_class_names(class_map_path: Path | None = None) -> list[str]:
    """Return the 521 YAMNet display names indexed by class id."""
    path = class_map_path or default_class_map_path()
    if not path.is_file():
        raise FileNotFoundError(f"YAMNet class map not found at {path}")
    names: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            names.append(row[2])
    return names


def _run_yamnet_inference(
    samples_16k: np.ndarray,
    *,
    model_path: Path,
) -> np.ndarray:
    """Return a ``(num_frames, 521)`` float32 score matrix.

    YAMNet's ONNX export takes a 1-D float32 waveform and emits one row
    of class scores every 0.48 s. We pad ultra-short clips up to one
    full frame so the model still produces a valid output.
    """
    session = _load_session(model_path)
    waveform = np.ascontiguousarray(samples_16k, dtype=np.float32)
    if waveform.size == 0:
        return np.zeros((0, 521), dtype=np.float32)

    min_samples = int(YAMNET_FRAME_SEC * YAMNET_SAMPLE_RATE)
    if waveform.size < min_samples:
        padded = np.zeros(min_samples, dtype=np.float32)
        padded[: waveform.size] = waveform
        waveform = padded

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: waveform})
    scores = np.asarray(outputs[0], dtype=np.float32)
    return scores


def _aggregate_per_window(
    scores: np.ndarray,
    *,
    audio_window_bounds: list[tuple[float, float]],
) -> dict[str, list[float]]:
    """Average YAMNet frames that fall inside each ``[t0, t1]`` window.

    Returns a dict whose keys match the schema fields persisted on
    ``AudioWindow.model_extra`` (``KEY_MUSIC`` / ``KEY_JINGLE`` / ...).
    Each value is a list aligned with ``audio_window_bounds``.
    """
    n_frames = scores.shape[0]
    if n_frames == 0:
        empty = [0.0] * len(audio_window_bounds)
        return {key: list(empty) for key in ALL_YAMNET_KEYS}

    music_per_frame = scores[:, list(_MUSIC_CLASS_IDS)].max(axis=1)
    jingle_per_frame = scores[:, _JINGLE_CLASS_ID]
    theme_per_frame = scores[:, _THEME_CLASS_ID]
    speech_per_frame = scores[:, _SPEECH_CLASS_ID]

    frame_centers = (np.arange(n_frames, dtype=np.float64) * YAMNET_HOP_SEC) + 0.5 * YAMNET_FRAME_SEC

    music_out: list[float] = []
    jingle_out: list[float] = []
    theme_out: list[float] = []
    speech_out: list[float] = []

    for t0, t1 in audio_window_bounds:
        mask = (frame_centers >= t0) & (frame_centers < t1)
        if not mask.any():
            nearest = int(np.argmin(np.abs(frame_centers - 0.5 * (t0 + t1))))
            music_out.append(float(music_per_frame[nearest]))
            jingle_out.append(float(jingle_per_frame[nearest]))
            theme_out.append(float(theme_per_frame[nearest]))
            speech_out.append(float(speech_per_frame[nearest]))
        else:
            music_out.append(float(music_per_frame[mask].mean()))
            jingle_out.append(float(jingle_per_frame[mask].mean()))
            theme_out.append(float(theme_per_frame[mask].mean()))
            speech_out.append(float(speech_per_frame[mask].mean()))

    return {
        KEY_MUSIC: music_out,
        KEY_JINGLE: jingle_out,
        KEY_THEME: theme_out,
        KEY_SPEECH: speech_out,
    }


def compute_yamnet_per_window(
    samples_16k: np.ndarray,
    *,
    audio_window_bounds: list[tuple[float, float]],
    model_path: Path | None = None,
) -> dict[str, list[float]]:
    """Run YAMNet on a 16 kHz mono PCM array and aggregate to window cadence.

    Parameters
    ----------
    samples_16k
        Float32 array of mono samples at 16 kHz, values roughly in
        ``[-1, 1]`` (matches what ``audio/analyze.py`` already produces
        from ffmpeg).
    audio_window_bounds
        List of ``(t0, t1)`` pairs in seconds — typically the
        ``[t0, t1]`` of every ``AudioWindow`` in the bundle.
    model_path
        Override the default model path.

    Returns
    -------
    dict[str, list[float]]
        Per-window scores, one list per output key. Lists are aligned
        with ``audio_window_bounds``. Empty inputs return zero-filled
        lists so callers can blanket-assign without checking.
    """
    if not audio_window_bounds:
        return {key: [] for key in ALL_YAMNET_KEYS}
    path = model_path or default_model_path()
    scores = _run_yamnet_inference(samples_16k, model_path=path)
    return _aggregate_per_window(scores, audio_window_bounds=audio_window_bounds)
