from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from schemas.modality import AnalysisBundle, VisualHypothesis, VisualTrack, VisualWindow

try:
    from scenedetect import ContentDetector
    from scenedetect import detect as scenedetect_run

    _HAS_SCENEDETECT = True
except ImportError:
    _HAS_SCENEDETECT = False


def probe_video_metadata(path: Path) -> tuple[float, float, int, int, int]:
    duration = _ffprobe_duration(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    if fps <= 0 and duration > 0 and frame_count > 0:
        fps = frame_count / duration
    if duration <= 0 and fps > 0 and frame_count > 0:
        duration = frame_count / fps
    cap.release()
    return duration, fps, width, height, frame_count


def _ffprobe_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
        return max(0.0, float(completed.stdout.strip()))
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError, subprocess.TimeoutExpired):
        return 0.0


def _detect_scene_cuts_sec(video_path: Path, threshold: float = 27.0) -> list[float]:
    if not _HAS_SCENEDETECT:
        return []
    try:
        scene_list = scenedetect_run(str(video_path), ContentDetector(threshold=threshold), show_progress=False)
    except Exception:
        return []
    cuts: list[float] = []
    si = 0
    n_scenes = len(scene_list)
    while si < n_scenes:
        start_tc, _end_tc = scene_list[si]
        cuts.append(float(start_tc.get_seconds()))
        si += 1
    return sorted(set(cuts))


def _nearest_cut_distance(t_mid: float, cuts: list[float]) -> float | None:
    if not cuts:
        return None
    best = abs(t_mid - cuts[0])
    ci = 1
    n_cuts = len(cuts)
    while ci < n_cuts:
        d = abs(t_mid - cuts[ci])
        if d < best:
            best = d
        ci += 1
    return best


def _normalize_robust(values: list[float], low_q: float = 0.1, high_q: float = 0.9) -> list[float]:
    if not values:
        return []
    arr = np.array(values, dtype=np.float64)
    lo = float(np.quantile(arr, low_q))
    hi = float(np.quantile(arr, high_q))
    if hi <= lo + 1e-9:
        out: list[float] = []
        zi = 0
        zn = len(values)
        while zi < zn:
            out.append(0.5)
            zi += 1
        return out
    scaled = (arr - lo) / (hi - lo)
    raw = scaled.tolist()
    res: list[float] = []
    ri = 0
    rn = len(raw)
    while ri < rn:
        res.append(float(min(1.0, max(0.0, raw[ri]))))
        ri += 1
    return res


_HIST_BINS = [8, 8, 8]      # 8x8x8 = 512 elements
_HIST_SIZE = 8 * 8 * 8      # must match _HIST_BINS product
_HIST_RANGES = [0, 256] * 3


def _small_bgr_hist(bgr: np.ndarray) -> np.ndarray:
    h, w = bgr.shape[:2]
    # Always return _HIST_SIZE-element array so _hist_distance never sees
    # a shape mismatch. Previously returned np.zeros(24) for tiny frames
    # while normal frames returned 512 elements — that mismatch caused the
    # cv2.compareHist assertion error.
    if h < 2 or w < 2:
        return np.zeros(_HIST_SIZE, dtype=np.float64)
    small = cv2.resize(bgr, (32, 32), interpolation=cv2.INTER_AREA)
    hist = cv2.calcHist([small], [0, 1, 2], None, _HIST_BINS, _HIST_RANGES)
    hist = cv2.normalize(hist, None).flatten()
    arr = hist.astype(np.float64)
    # Defensive: ensure exactly _HIST_SIZE elements regardless of cv2 version
    if arr.size != _HIST_SIZE:
        arr = np.zeros(_HIST_SIZE, dtype=np.float64)
    return arr


def _hist_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Guard against any remaining shape mismatch — return 0.0 (no divergence)
    if a.shape != b.shape:
        return 0.0
    return float(cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA))


def _hypothesis_from_features(
    motion: float,
    edge_d: float,
    palette_d: float,
    text_like: float = 0.0,
) -> tuple[VisualHypothesis, float]:
    if text_like > 0.55 and edge_d > 0.35:
        return "graphics_heavy", min(1.0, 0.50 + max(text_like, edge_d) * 0.45)
    if motion < 0.12 and edge_d < 0.25:
        return "static", min(1.0, 0.55 + (0.12 - motion) * 2.0)
    if edge_d > 0.45 or palette_d > 0.55:
        return "graphics_heavy", min(1.0, 0.4 + max(edge_d, palette_d) * 0.55)
    if 0.12 <= motion <= 0.65:
        return "dynamic_talk", min(1.0, 0.35 + motion * 0.6)
    return "unknown", 0.25


def _resize_for_analysis(frame: np.ndarray, resize_max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, resize_max_width / max(w, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _iter_sampled_frames_in_window(
    cap: cv2.VideoCapture,
    native_fps: float,
    frame_stride: int,
    t0: float,
    t1: float,
    resize_max_width: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    start_frame = int(t0 * native_fps)
    end_frame = min(int(math.ceil(t1 * native_fps)) - 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
    if end_frame < start_frame:
        return
    f = start_frame
    while f <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        small = _resize_for_analysis(frame, resize_max_width)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        yield small, gray
        f += frame_stride


@dataclass
class _WindowRaw:
    t0: float
    t1: float
    motion_raw: float
    luminance_mean: float
    edge_raw: float
    text_like_raw: float
    palette_vs_ema_raw: float
    shot_boundary_near: bool
    shot_boundary_distance_sec: float | None
    mean_hist: np.ndarray


def _estimate_text_like_density(gray: np.ndarray, edges_map: np.ndarray) -> float:
    """Estimate caption/title/product-text density without running OCR."""
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0

    kernel_w = max(5, w // 45)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 2))
    merged = cv2.dilate(edges_map, kernel, iterations=1)
    contours, _hierarchy = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_area = 0.0
    frame_area = float(h * w)
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < 5 or ch < 3:
            continue
        aspect = cw / max(ch, 1)
        rel_area = (cw * ch) / frame_area
        if 1.2 <= aspect <= 28.0 and 0.00005 <= rel_area <= 0.08:
            text_area += cw * ch

    return float(min(1.0, (text_area / frame_area) * 8.0))


def _compute_window_raw(
    cap: cv2.VideoCapture,
    native_fps: float,
    frame_stride: int,
    t0: float,
    t1: float,
    cuts_sec: list[float],
    hist_ema: np.ndarray | None,
    resize_max_width: int,
) -> _WindowRaw:
    motions: list[float] = []
    lums: list[float] = []
    edges: list[float] = []
    text_like_scores: list[float] = []
    hist_sum: np.ndarray | None = None
    hist_count = 0
    prev_gray: np.ndarray | None = None
    frame_iter = _iter_sampled_frames_in_window(cap, native_fps, frame_stride, t0, t1, resize_max_width)
    while True:
        try:
            small, gray = next(frame_iter)
        except StopIteration:
            break
        lums.append(float(np.mean(gray)) / 255.0)
        edges_map = cv2.Canny(gray, 60, 140)
        edges.append(float(np.mean(edges_map > 0)))
        text_like_scores.append(_estimate_text_like_density(gray, edges_map))
        hvec = _small_bgr_hist(small)
        hist_sum = hvec if hist_sum is None else hist_sum + hvec
        hist_count += 1
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motions.append(float(np.mean(diff)) / 255.0)
        prev_gray = gray

    motion_raw = float(np.mean(motions)) if motions else 0.0
    lum_mean = float(np.mean(lums)) if lums else 0.0
    edge_mean = float(np.mean(edges)) if edges else 0.0
    text_like_mean = float(np.mean(text_like_scores)) if text_like_scores else 0.0
    mean_hist = (hist_sum / max(1, hist_count)) if hist_sum is not None else np.zeros(_HIST_SIZE, dtype=np.float64)
    if hist_ema is None:
        palette_vs_ema = 0.0
    else:
        palette_vs_ema = _hist_distance(mean_hist, hist_ema)

    mid = 0.5 * (t0 + t1)
    dist = _nearest_cut_distance(mid, cuts_sec)
    boundary_near = dist is not None and dist <= max(0.15, (t1 - t0) * 0.6)

    return _WindowRaw(
        t0=t0,
        t1=t1,
        motion_raw=motion_raw,
        luminance_mean=lum_mean,
        edge_raw=edge_mean,
        text_like_raw=text_like_mean,
        palette_vs_ema_raw=palette_vs_ema,
        shot_boundary_near=boundary_near,
        shot_boundary_distance_sec=dist,
        mean_hist=mean_hist,
    )


def _window_time_ranges(duration_sec: float, window_sec: float) -> list[tuple[float, float]]:
    if duration_sec <= 0:
        return []
    ranges: list[tuple[float, float]] = []
    t = 0.0
    while t < duration_sec - 1e-9:
        t1 = min(t + window_sec, duration_sec)
        if t1 > t + 1e-6:
            ranges.append((t, t1))
        t += window_sec
    return ranges


def analyze_visual(
    video_path: Path | str,
    *,
    sample_fps: float = 2.0,
    window_sec: float = 1.0,
    scenedetect_threshold: float = 27.0,
    resize_max_width: int = 320,
) -> VisualTrack:
    path = Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(str(path))

    duration_sec, native_fps, vw, vh, total_frames = probe_video_metadata(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")

    try:
        if native_fps <= 0 and duration_sec > 0 and total_frames > 0:
            native_fps = total_frames / duration_sec
        if native_fps <= 0:
            native_fps = 24.0
        if duration_sec <= 0:
            oc_fps = float(cap.get(cv2.CAP_PROP_FPS)) or native_fps
            oc_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if oc_fps > 0 and oc_count > 0:
                duration_sec = oc_count / oc_fps

        frame_stride = max(1, int(round(native_fps / max(0.25, sample_fps))))
        effective_fps = native_fps / frame_stride
        window_sec = max(0.25, float(window_sec))

        cuts_sec = _detect_scene_cuts_sec(path, threshold=scenedetect_threshold)
        time_ranges = _window_time_ranges(float(duration_sec), window_sec)

        hist_ema: np.ndarray | None = None
        raw_rows: list[_WindowRaw] = []
        tri = 0
        n_tr = len(time_ranges)
        while tri < n_tr:
            t0, t1 = time_ranges[tri]
            row = _compute_window_raw(
                cap,
                native_fps,
                frame_stride,
                t0,
                t1,
                cuts_sec,
                hist_ema,
                resize_max_width,
            )
            raw_rows.append(row)
            hist_ema = row.mean_hist if hist_ema is None else (0.85 * hist_ema + 0.15 * row.mean_hist)
            tri += 1

        if not raw_rows:
            return VisualTrack(
                video_path=str(path),
                duration_sec=float(duration_sec),
                native_fps=float(native_fps),
                fps_sampled=float(effective_fps),
                frame_stride=int(frame_stride),
                video_width=int(vw),
                video_height=int(vh),
                window_sec=float(window_sec),
                windows=[],
            )

        motion_vals: list[float] = []
        edge_vals: list[float] = []
        text_vals: list[float] = []
        pal_vals: list[float] = []
        vi = 0
        nv = len(raw_rows)
        while vi < nv:
            motion_vals.append(raw_rows[vi].motion_raw)
            edge_vals.append(raw_rows[vi].edge_raw)
            text_vals.append(raw_rows[vi].text_like_raw)
            pal_vals.append(raw_rows[vi].palette_vs_ema_raw)
            vi += 1
        motion_norm = _normalize_robust(motion_vals)
        edge_norm = _normalize_robust(edge_vals)
        text_norm = _normalize_robust(text_vals)
        palette_norm = _normalize_robust(pal_vals)

        windows: list[VisualWindow] = []
        wi = 0
        while wi < nv:
            rw = raw_rows[wi]
            m = motion_norm[wi] if wi < len(motion_norm) else 0.0
            e = edge_norm[wi] if wi < len(edge_norm) else 0.0
            txt = text_norm[wi] if wi < len(text_norm) else 0.0
            p = palette_norm[wi] if wi < len(palette_norm) else 0.0
            hyp, conf = _hypothesis_from_features(m, e, p, txt)
            high_text = (txt > 0.50 and e > 0.32) or (e > 0.62 and m < 0.42)
            windows.append(
                VisualWindow(
                    t0=float(rw.t0),
                    t1=float(rw.t1),
                    motion_score=m,
                    luminance_mean=float(rw.luminance_mean),
                    edge_density=e,
                    shot_boundary_near=rw.shot_boundary_near,
                    shot_boundary_distance_sec=rw.shot_boundary_distance_sec,
                    palette_delta=p,
                    high_text_density=high_text,
                    visual_hypothesis=hyp,
                    hypothesis_confidence=conf,
                )
            )
            wi += 1

        return VisualTrack(
            video_path=str(path),
            duration_sec=float(duration_sec),
            native_fps=float(native_fps),
            fps_sampled=float(effective_fps),
            frame_stride=int(frame_stride),
            video_width=int(vw),
            video_height=int(vh),
            window_sec=float(window_sec),
            windows=windows,
        )
    finally:
        cap.release()


def visual_track_to_json_dict(track: VisualTrack) -> dict[str, Any]:
    return json.loads(track.model_dump_json())


def write_visual_track_json(track: VisualTrack, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(track.model_dump_json(indent=2), encoding="utf-8")


def build_analysis_bundle(video_path: Path | str, track: VisualTrack | None = None) -> AnalysisBundle:
    path = Path(video_path).resolve()
    visual = track if track is not None else analyze_visual(path)
    return AnalysisBundle(
        video_path=str(path),
        duration_sec=visual.duration_sec,
        visual=visual,
        audio_windows=[],
        speech_spans=[],
    )


def write_analysis_bundle_json(bundle: AnalysisBundle, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
