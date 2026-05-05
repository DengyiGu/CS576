"""Optional frame OCR for ad detection.

The OCR output is represented as SpeechSpan records with ``source="ocr"`` so the
fusion layer can reuse its existing text and brand matching logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from schemas.modality import SpeechSpan


DEFAULT_OCR_MODEL_DIR = Path(__file__).resolve().parent / "models"


def build_ocr_spans(
    video_path: Path,
    *,
    sample_every_sec: float = 3.0,
    span_sec: float = 3.0,
    candidate_times: Iterable[float] | None = None,
    max_frames: int = 120,
    resize_width: int = 960,
    languages: Iterable[str] = ("en",),
    model_dir: Path = DEFAULT_OCR_MODEL_DIR,
    allow_download: bool = False,
    gpu: bool = False,
    min_confidence: float = 0.35,
) -> list[SpeechSpan]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python or opencv-python-headless to run OCR.") from exc

    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError("Install optional OCR dependency: python -m pip install -r requirements-ocr.txt") from exc

    if sample_every_sec <= 0:
        raise ValueError("sample_every_sec must be positive.")

    model_dir.mkdir(parents=True, exist_ok=True)
    reader = easyocr.Reader(
        list(languages),
        gpu=gpu,
        model_storage_directory=str(model_dir),
        download_enabled=allow_download,
        verbose=False,
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video for OCR: {video_path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

        if candidate_times is None:
            sample_times = []
            t = 0.0
            while duration <= 0.0 or t < duration:
                sample_times.append(t)
                t += sample_every_sec
                if duration <= 0.0 and len(sample_times) >= max_frames:
                    break
        else:
            sample_times = sorted({round(float(t), 3) for t in candidate_times if float(t) >= 0.0})
            if duration > 0.0:
                # Add a sparse global sweep so short brand/logo text is not
                # missed just because the visual pre-filter skipped that frame.
                t = 0.0
                while t < duration:
                    sample_times.append(round(float(t), 3))
                    t += sample_every_sec
                sample_times = sorted(set(sample_times))

        if max_frames > 0 and len(sample_times) > max_frames:
            stride = max(1, round(len(sample_times) / max_frames))
            sample_times = sample_times[::stride][:max_frames]

        spans: list[SpeechSpan] = []
        for t in sample_times:
            if duration > 0.0 and t >= duration:
                continue
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = capture.read()
            if not ok:
                continue

            if resize_width > 0 and frame.shape[1] > resize_width:
                scale = resize_width / float(frame.shape[1])
                frame = cv2.resize(
                    frame,
                    (resize_width, max(1, int(frame.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = reader.readtext(rgb_frame, detail=1, paragraph=False)
            text_chunks = [
                str(text).strip()
                for _box, text, confidence in results
                if confidence >= min_confidence and str(text).strip()
            ]
            if text_chunks:
                spans.append(
                    SpeechSpan(
                        t0=float(t),
                        t1=float(min(t + span_sec, duration) if duration > 0.0 else t + span_sec),
                        text=" ".join(text_chunks),
                        source="ocr",
                    )
                )
        return spans
    finally:
        capture.release()
