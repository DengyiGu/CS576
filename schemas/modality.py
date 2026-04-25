from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

VisualHypothesis = Literal["graphics_heavy", "static", "dynamic_talk", "unknown"]


class VisualWindow(BaseModel):
    t0: float = Field(..., ge=0.0)
    t1: float = Field(..., ge=0.0)
    motion_score: float = Field(..., ge=0.0, le=1.0)
    luminance_mean: float = Field(..., ge=0.0, le=1.0)
    edge_density: float = Field(default=0.0, ge=0.0, le=1.0)
    shot_boundary_near: bool = Field(default=False)
    shot_boundary_distance_sec: float | None = Field(default=None, ge=0.0)
    palette_delta: float = Field(default=0.0, ge=0.0, le=1.0)
    high_text_density: bool = Field(default=False)
    visual_hypothesis: VisualHypothesis = Field(default="unknown")
    hypothesis_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = {"extra": "forbid"}


class VisualTrack(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    modality: Literal["visual"] = "visual"
    video_path: str = ""
    duration_sec: float = Field(..., ge=0.0)
    native_fps: float = Field(default=0.0, ge=0.0)
    fps_sampled: float = Field(..., ge=0.0)
    frame_stride: int = Field(..., ge=1)
    video_width: int = Field(default=0, ge=0)
    video_height: int = Field(default=0, ge=0)
    window_sec: float = Field(default=1.0, gt=0.0)
    windows: list[VisualWindow] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class AudioWindow(BaseModel):
    t0: float = Field(..., ge=0.0)
    t1: float = Field(..., ge=0.0)

    model_config = {"extra": "allow"}


class SpeechSpan(BaseModel):
    t0: float = Field(..., ge=0.0)
    t1: float = Field(..., ge=0.0)
    text: str = ""

    model_config = {"extra": "allow"}


class AnalysisBundle(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    video_path: str = ""
    duration_sec: float = Field(default=0.0, ge=0.0)
    visual: VisualTrack | None = None
    audio_windows: list[AudioWindow] = Field(default_factory=list)
    speech_spans: list[SpeechSpan] = Field(default_factory=list)

    model_config = {"extra": "forbid"}
