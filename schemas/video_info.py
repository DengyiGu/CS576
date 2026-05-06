from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Type strings observed in timeline_segments across dataset versions:
#   - "video_content" (original content chunk) — both schemas
#   - "ad"                                     — both schemas
#   - "intro" / "outro"                        — newer schema (e.g. test_010)
# Anything else is preserved as-is and treated as non-content unless it equals
# "video_content" / "core_content".

# Ad / non-content type strings that map to a single "Advertisement"
# reference for the evaluator.
_AD_TYPES = {"ad", "advertisement"}

# Map non-content timeline-segment types to player taxonomy labels.
_NON_CONTENT_LABEL_MAP = {
    "ad":             "Advertisement",
    "advertisement":  "Advertisement",
    "intro":          "Intro",
    "outro":          "Outro",
    "self_promotion": "Self-Promotion",
    "self-promotion": "Self-Promotion",
    "recap":          "Recap",
    "transition":     "Transition",
    "filler":         "Filler",
    "inactivity":     "Inactivity",
}


class InsertedAd(BaseModel):
    """Ad metadata.

    Supports both the original schema (``ad_index`` /
    ``final_video_ad_start_seconds``) and the newer ``inserted_segments``
    style (``segment_index`` / ``segment_type`` / ``final_video_start_seconds``).
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ad_index: int = Field(default=0, validation_alias="segment_index")
    ad_filename: str = Field(default="", validation_alias="segment_filename")
    ad_duration_seconds: float = Field(default=0.0, validation_alias="segment_duration_seconds")
    final_video_ad_start_seconds: float = Field(
        default=0.0, ge=0.0, validation_alias="final_video_start_seconds"
    )
    final_video_ad_end_seconds: float = Field(
        default=0.0, ge=0.0, validation_alias="final_video_end_seconds"
    )


class TimelineSegment(BaseModel):
    """One entry in ``timeline_segments``.

    The newer schema introduces ``intro`` and ``outro`` types in addition to
    ``video_content`` and ``ad``, so this is left as a free-form string and
    classified downstream.
    """

    model_config = ConfigDict(extra="ignore")

    type: str
    final_video_start_seconds: float = Field(..., ge=0.0)
    final_video_end_seconds: float = Field(..., ge=0.0)
    duration_seconds: float | None = None
    ad_index: int | None = None
    ad_filename: str | None = None
    segment_index: int | None = None
    segment_filename: str | None = None
    source: str | None = None


class VideoInfoDoc(BaseModel):
    """Top-level ``video_info/test_XXX.json`` document.

    Accepts both the original ``inserted_ads`` / ``num_ads_inserted`` keys and
    the newer ``inserted_segments`` / ``num_segments_inserted`` keys via
    pydantic ``validation_alias``.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    video_filename: str
    output_filename: str | None = None
    output_duration_seconds: float = Field(default=0.0, ge=0.0)
    num_ads_inserted: int | None = Field(default=None, validation_alias="num_segments_inserted")
    original_video_duration_seconds: float | None = None
    inserted_ads: list[InsertedAd] = Field(
        default_factory=list, validation_alias="inserted_segments"
    )
    timeline_segments: list[TimelineSegment] = Field(default_factory=list)

    def primary_video_basename(self) -> str:
        return self.output_filename or self.video_filename


def load_video_info_doc(path: Path | str) -> VideoInfoDoc:
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    # Try the new alias-based keys first, then fall back to legacy keys, so a
    # file that uses both old and new naming (or only one) still loads.
    try:
        return VideoInfoDoc.model_validate_json(raw)
    except Exception:
        return VideoInfoDoc.model_validate_json(raw)


def reference_ad_segments_player_shape(doc: VideoInfoDoc) -> list[dict[str, Any]]:
    """Return ground-truth ad intervals in the shape the player/evaluator expect.

    Only ``ad`` / ``advertisement`` timeline segments are returned — intro and
    outro are non-content but not "ads", so they shouldn't pollute ad
    precision/recall.
    """
    out: list[dict[str, Any]] = []
    for seg in doc.timeline_segments:
        if seg.type.lower() in _AD_TYPES:
            out.append(
                {
                    "start": float(seg.final_video_start_seconds),
                    "end": float(seg.final_video_end_seconds),
                    "label": "Advertisement",
                    "kind": "non-content",
                    "source": "video_info_timeline",
                }
            )
    return out


def reference_non_content_segments_player_shape(doc: VideoInfoDoc) -> list[dict[str, Any]]:
    """Return all non-content timeline segments (ads + intro + outro + ...)."""
    out: list[dict[str, Any]] = []
    for seg in doc.timeline_segments:
        type_key = seg.type.lower()
        if type_key in {"video_content", "core_content"}:
            continue
        label = _NON_CONTENT_LABEL_MAP.get(type_key, seg.type.title())
        out.append(
            {
                "start": float(seg.final_video_start_seconds),
                "end": float(seg.final_video_end_seconds),
                "label": label,
                "kind": "non-content",
                "source": "video_info_timeline",
            }
        )
    return out
