from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TAXONOMY = [
    {
        "label": "Core Content",
        "kind": "content",
        "color": "#2f9e44",
    },
    {
        "label": "Intro",
        "kind": "non-content",
        "color": "#1c7ed6",
    },
    {
        "label": "Outro",
        "kind": "non-content",
        "color": "#495057",
    },
    {
        "label": "Advertisement",
        "kind": "non-content",
        "color": "#c92a2a",
    },
    {
        "label": "Self-Promotion",
        "kind": "non-content",
        "color": "#e67700",
    },
    {
        "label": "Recap",
        "kind": "non-content",
        "color": "#f08c00",
    },
    {
        "label": "Transition",
        "kind": "non-content",
        "color": "#5f3dc4",
    },
    {
        "label": "Inactivity",
        "kind": "non-content",
        "color": "#868e96",
    },
    {
        "label": "Filler",
        "kind": "non-content",
        "color": "#a61e4d",
    },
]
EPSILON_SECONDS = 0.04
TIMELINE_TRACK_SIDE_PADDING = 9


@dataclass
class Segment:
    identifier: str
    start: float
    end: float
    label_name: str
    kind: str
    color: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def taxonomy_item_for_label(label_name: str) -> dict[str, str] | None:
    for item in TAXONOMY:
        if item["label"] == label_name:
            return item
    return None


def build_segment_from_payload(payload: dict[str, Any], index: int) -> Segment | None:
    try:
        start = float(payload["start"])
        end = float(payload["end"])
    except (KeyError, TypeError, ValueError):
        return None

    if end <= start:
        return None

    label_name = str(payload.get("label") or "Core Content")
    taxonomy_item = taxonomy_item_for_label(label_name) or {
        "label": label_name,
        "kind": str(payload.get("kind") or "non-content"),
        "color": "#868e96",
    }

    return Segment(
        identifier=str(payload.get("id") or f"segment-{index}-{int(start * 1000)}-{int(end * 1000)}"),
        start=start,
        end=end,
        label_name=str(taxonomy_item["label"]),
        kind=str(payload.get("kind") or taxonomy_item["kind"]),
        color=str(payload.get("color") or taxonomy_item["color"]),
    )


def build_full_content_segment(duration_seconds: float) -> list[Segment]:
    if duration_seconds <= 0:
        raise ValueError("Video duration must be positive before building segments.")

    segment = build_segment_from_payload(
        {
            "id": "segment-0",
            "start": 0.0,
            "end": duration_seconds,
            "label": "Core Content",
        },
        0,
    )
    return [segment] if segment is not None else []


def probe_video_duration_seconds(path: Path) -> float:
    duration_seconds = float(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    if duration_seconds <= 0:
        raise ValueError(f"Unable to determine a positive duration for {path.name}.")
    return duration_seconds
