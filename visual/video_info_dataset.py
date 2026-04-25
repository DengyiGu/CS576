from __future__ import annotations

import json
from pathlib import Path

from schemas.video_info import VideoInfoDoc, load_video_info_doc, reference_ad_segments_player_shape


def _search_roots(videos_root: Path) -> list[Path]:
    roots: list[Path] = []
    candidates = (
        videos_root,
        Path("data/input/videos_with_ads"),
        Path("data/input/videos_with_ad"),
        Path("videos_with_ad"),
        Path("videos_with_ads"),
        Path("data/input"),
    )
    ci = 0
    cn = len(candidates)
    while ci < cn:
        p = candidates[ci]
        try:
            r = p.expanduser().resolve()
        except OSError:
            ci += 1
            continue
        if r not in roots:
            roots.append(r)
        ci += 1
    return roots


def find_stitched_video_file(doc: VideoInfoDoc, videos_root: Path) -> tuple[Path | None, list[Path]]:
    name = doc.primary_video_basename()
    tried: list[Path] = []
    roots = _search_roots(videos_root)
    ri = 0
    rn = len(roots)
    while ri < rn:
        root = roots[ri]
        candidate = (root / name).resolve()
        tried.append(candidate)
        if candidate.is_file():
            return candidate, tried
        ri += 1
    return None, tried


def load_doc_and_resolve_video(
    video_info_json: Path, videos_root: Path
) -> tuple[VideoInfoDoc, Path | None, list[Path]]:
    doc = load_video_info_doc(video_info_json)
    found, tried = find_stitched_video_file(doc, videos_root)
    return doc, found, tried


def list_mp4s_under_data_input(*, max_files: int = 20) -> list[Path]:
    base = Path("data/input")
    if not base.is_dir():
        return []
    found: list[Path] = []
    paths = sorted(base.rglob("*.mp4"))
    pi = 0
    pn = len(paths)
    while pi < pn:
        path = paths[pi]
        if path.is_file():
            found.append(path.resolve())
        if len(found) >= max_files:
            break
        pi += 1
    return found


def write_reference_ads_json(doc: VideoInfoDoc, out_path: Path) -> None:
    segments = reference_ad_segments_player_shape(doc)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "source": "video_info",
        "video_filename": doc.video_filename,
        "segments": segments,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
