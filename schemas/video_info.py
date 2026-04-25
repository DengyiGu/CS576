from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class InsertedAd(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ad_index: int
    ad_filename: str
    ad_duration_seconds: float = 0.0
    final_video_ad_start_seconds: float = Field(..., ge=0.0)
    final_video_ad_end_seconds: float = Field(..., ge=0.0)


class TimelineSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["video_content", "ad"]
    final_video_start_seconds: float = Field(..., ge=0.0)
    final_video_end_seconds: float = Field(..., ge=0.0)
    duration_seconds: float | None = None
    ad_index: int | None = None
    ad_filename: str | None = None


class VideoInfoDoc(BaseModel):
    model_config = ConfigDict(extra="ignore")

    video_filename: str
    output_filename: str | None = None
    output_duration_seconds: float = Field(default=0.0, ge=0.0)
    num_ads_inserted: int | None = None
    original_video_duration_seconds: float | None = None
    inserted_ads: list[InsertedAd] = Field(default_factory=list)
    timeline_segments: list[TimelineSegment] = Field(default_factory=list)

    def primary_video_basename(self) -> str:
        return self.output_filename or self.video_filename


def load_video_info_doc(path: Path | str) -> VideoInfoDoc:
    p = Path(path)
    return VideoInfoDoc.model_validate_json(p.read_text(encoding="utf-8"))


def reference_ad_segments_player_shape(doc: VideoInfoDoc) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    segs = doc.timeline_segments
    si = 0
    sn = len(segs)
    while si < sn:
        seg = segs[si]
        if seg.type == "ad":
            out.append(
                {
                    "start": float(seg.final_video_start_seconds),
                    "end": float(seg.final_video_end_seconds),
                    "label": "Advertisement",
                    "kind": "non-content",
                    "source": "video_info_timeline",
                }
            )
        si += 1
    return out
