"""Tests for visual analysis (no Qt player)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from schemas.modality import AnalysisBundle, VisualTrack
from schemas.video_info import VideoInfoDoc, load_video_info_doc, reference_ad_segments_player_shape
from visual.analyze import analyze_visual, build_analysis_bundle, write_visual_track_json
from visual.video_info_dataset import find_stitched_video_file


def _write_synthetic_mp4(path: Path, *, fps: float = 10.0, static_sec: float = 2.0, noise_sec: float = 2.0) -> None:
    width, height = 96, 72
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open (codec mp4v).")

    static_frames = int(static_sec * fps)
    noise_frames = int(noise_sec * fps)
    for _ in range(static_frames):
        frame = np.full((height, width, 3), 80, dtype=np.uint8)
        writer.write(frame)
    for _ in range(noise_frames):
        frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_analyze_visual_static_then_noise_motion(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.mp4"
    _write_synthetic_mp4(video, fps=12.0, static_sec=2.0, noise_sec=2.0)

    track = analyze_visual(video, sample_fps=3.0, window_sec=0.5)
    assert track.duration_sec > 3.5
    assert len(track.windows) >= 6

    static_motion = [w.motion_score for w in track.windows if w.t1 <= 2.0 + 1e-3]
    noise_motion = [w.motion_score for w in track.windows if w.t0 >= 2.0 - 1e-3]
    assert static_motion, "expected windows in static region"
    assert noise_motion, "expected windows in noise region"

    assert float(np.mean(static_motion)) < float(np.mean(noise_motion)) - 0.05


def test_visual_track_json_roundtrip(tmp_path: Path) -> None:
    video = tmp_path / "synthetic2.mp4"
    _write_synthetic_mp4(video, fps=8.0, static_sec=1.0, noise_sec=1.0)
    track = analyze_visual(video, sample_fps=2.0, window_sec=1.0)
    out = tmp_path / "track.json"
    write_visual_track_json(track, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    restored = VisualTrack.model_validate(data)
    assert restored.duration_sec == pytest.approx(track.duration_sec, rel=0, abs=0.05)
    assert len(restored.windows) == len(track.windows)


def test_analysis_bundle_schema(tmp_path: Path) -> None:
    video = tmp_path / "synthetic3.mp4"
    _write_synthetic_mp4(video, fps=10.0, static_sec=1.5, noise_sec=1.5)
    track = analyze_visual(video)
    bundle = build_analysis_bundle(video, track=track)
    dumped = json.loads(bundle.model_dump_json())
    again = AnalysisBundle.model_validate(dumped)
    assert again.visual is not None
    assert again.audio_windows == []
    assert again.speech_spans == []


def test_sample_analysis_bundle_on_disk() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sample = repo_root / "sample_data" / "sample_analysis_bundle.json"
    data = json.loads(sample.read_text(encoding="utf-8"))
    AnalysisBundle.model_validate(data)


def test_find_nested_data_input_videos_with_ads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    nested = tmp_path / "data" / "input" / "videos_with_ads"
    nested.mkdir(parents=True)
    clip = nested / "clip.mp4"
    clip.write_bytes(b"x")
    doc = VideoInfoDoc(video_filename="clip.mp4", output_duration_seconds=1.0)
    root = tmp_path / "data" / "input"
    root.mkdir(parents=True, exist_ok=True)
    found, _ = find_stitched_video_file(doc, root)
    assert found == clip.resolve()


def test_find_stitched_video_falls_back_to_videos_with_ad(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "videos_with_ad").mkdir(parents=True)
    clip = tmp_path / "videos_with_ad" / "clip.mp4"
    clip.write_bytes(b"fake")
    doc = VideoInfoDoc(video_filename="clip.mp4", output_duration_seconds=1.0)
    empty_root = tmp_path / "empty_input"
    empty_root.mkdir()
    found, tried = find_stitched_video_file(doc, empty_root)
    assert found == clip.resolve()
    assert tried[0] == (empty_root / "clip.mp4").resolve()


def test_video_info_json_loads_and_reference_ads() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    doc = load_video_info_doc(repo_root / "video_info" / "test_001.json")
    assert doc.video_filename == "test_001.mp4"
    assert doc.num_ads_inserted == 3
    ads = reference_ad_segments_player_shape(doc)
    assert len(ads) == 3
    assert ads[0]["label"] == "Advertisement"
    assert ads[0]["end"] > ads[0]["start"]


def test_cli_video_info_without_mp4_exits_cleanly(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    json_src = repo_root / "video_info" / "test_001.json"
    payload = json.loads(json_src.read_text(encoding="utf-8"))
    payload["video_filename"] = "__missing_dataset_clip__.mp4"
    payload["output_filename"] = "__missing_dataset_clip__.mp4"
    info_copy = tmp_path / "probe.json"
    info_copy.write_text(json.dumps(payload), encoding="utf-8")
    empty_root = tmp_path / "videos"
    empty_root.mkdir()
    cmd = [
        sys.executable,
        "-m",
        "visual_analyze",
        "--video-info",
        str(info_copy),
        "--videos-root",
        str(empty_root),
        "--out",
        str(tmp_path / "out.json"),
    ]
    env = {**__import__("os").environ, "PYTHONPATH": str(repo_root)}
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 2
    combined = proc.stderr + proc.stdout
    assert "stitched video not found" in combined.lower()
    assert "__missing_dataset_clip__" in combined or "missing_dataset" in combined.lower()


def test_cli_exits_cleanly_when_video_missing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    missing = tmp_path / "nope.mp4"
    cmd = [sys.executable, "-m", "visual_analyze", "--video", str(missing), "--out", str(tmp_path / "out.json")]
    env = {**__import__("os").environ, "PYTHONPATH": str(repo_root)}
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 2
    assert "video file not found" in proc.stderr.lower()


def test_cli_module_writes_file(tmp_path: Path) -> None:
    video = tmp_path / "cli.mp4"
    _write_synthetic_mp4(video, fps=10.0, static_sec=1.0, noise_sec=1.0)
    out = tmp_path / "out.json"
    bundle = tmp_path / "bundle.json"
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "visual_analyze",
        "--video",
        str(video),
        "--out",
        str(out),
        "--bundle-out",
        str(bundle),
        "--window-sec",
        "0.5",
    ]
    env = {**__import__("os").environ, "PYTHONPATH": str(repo_root)}
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True, env=env, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.is_file()
    VisualTrack.model_validate(json.loads(out.read_text(encoding="utf-8")))
    AnalysisBundle.model_validate(json.loads(bundle.read_text(encoding="utf-8")))
