from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any

from schemas.modality import AnalysisBundle, VisualWindow, AudioWindow, SpeechSpan

LABEL_CORE_CONTENT = "Core Content"
LABEL_ADVERTISEMENT = "Advertisement"
LABEL_INTRO = "Intro"
LABEL_OUTRO = "Outro"


def _speech_coverage(t0: float, t1: float, speech_spans: list[SpeechSpan]) -> float:
    dur = max(t1 - t0, 1e-6)
    covered = 0.0
    for span in speech_spans:
        extra = getattr(span, 'model_extra', {}) or {}
        if extra.get("source") in {"ocr", "semantic", "semantic_structure"}:
            continue
        ov_s = max(t0, span.t0)
        ov_e = min(t1, span.t1)
        if ov_e > ov_s and span.text:
            covered += ov_e - ov_s
    return min(1.0, covered / dur)


def score_window(w: VisualWindow, audio_windows: list[AudioWindow], speech_spans: list[SpeechSpan], duration: float) -> float:
    t_mid = 0.5 * (w.t0 + w.t1)
    
    # VISUAL - stricter
    palette = float(w.palette_delta)
    graphics = 1.0 if w.visual_hypothesis == "graphics_heavy" else 0.0
    text_dense = 1.0 if getattr(w, 'high_text_density', False) else 0.0

    visual_score = 0.50 * palette + 0.25 * graphics + 0.15 * text_dense

    # AUDIO - strongest signal
    anomaly = 0.0
    energy = 1.0
    audio_label = "unknown"
    if audio_windows:
        best = min(audio_windows, key=lambda aw: abs(0.5*(aw.t0 + aw.t1) - t_mid))
        extra = getattr(best, 'model_extra', {}) or {}
        anomaly = float(extra.get("anomaly_score", 0.0))
        energy = float(extra.get("energy_rms", 1.0))
        audio_label = str(extra.get("audio_label", "unknown"))

    audio_score = anomaly
    if audio_label in {"music", "mixed"}:
        audio_score = max(audio_score, 0.85)
    if energy < 0.15:
        audio_score = max(audio_score, 0.45)

    # SPEECH
    speech_cov = _speech_coverage(w.t0, w.t1, speech_spans)
    speech_score = 0.85 if speech_cov < 0.35 else 0.0

    total_score = (0.32 * visual_score + 0.55 * audio_score + 0.13 * speech_score)

    # Strong edge suppression
    if t_mid < 60 or t_mid > duration - 60:
        total_score *= 0.25

    return float(np.clip(total_score, 0.0, 1.0))


def _post_process(segments: list[dict], duration: float) -> list[dict]:
    if not segments:
        return []

    merged = []
    for seg in segments:
        if merged and merged[-1]["label"] == seg["label"]:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))

    final = []
    intro_done = False
    outro_done = False

    for i, seg in enumerate(merged):
        dur = seg["end"] - seg["start"]
        start, end = seg["start"], seg["end"]

        # Intro - very beginning only
        if not intro_done and start < 160 and dur < 170 and i <= 3:
            final.append({"start": round(start, 3), "end": round(end, 3), "label": LABEL_INTRO, "kind": "non-content"})
            intro_done = True
            continue

        # Outro - very end only
        if not outro_done and end > duration - 130 and dur < 230 and i >= len(merged)-4:
            final.append({"start": round(start, 3), "end": round(end, 3), "label": LABEL_OUTRO, "kind": "non-content"})
            outro_done = True
            continue

        final.append(seg)

    return final


def fuse_bundle_to_segments(bundle: AnalysisBundle, min_segment_seconds: float = 20.0) -> list[dict[str, Any]]:
    if not bundle.visual or not bundle.visual.windows:
        return []

    windows = bundle.visual.windows
    duration = bundle.visual.duration_sec or bundle.duration_sec
    audio_windows = getattr(bundle, 'audio_windows', []) or []
    speech_spans = getattr(bundle, 'speech_spans', []) or []

    scores = np.array([score_window(w, audio_windows, speech_spans, duration) for w in windows])

    is_ad = scores >= 0.57   # Higher threshold

    segments = []
    i = 0
    n = len(windows)

    while i < n:
        if is_ad[i]:
            j = i
            while j < n and is_ad[j]:
                j += 1
            start = windows[i].t0
            end = windows[j-1].t1
            if end - start >= 28.0:   # stricter minimum ad length
                segments.append({
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "label": LABEL_ADVERTISEMENT,
                    "kind": "non-content"
                })
            i = j
        else:
            j = i
            while j < n and not is_ad[j]:
                j += 1
            start = windows[i].t0
            end = windows[j-1].t1
            if end - start >= 15.0:
                segments.append({
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "label": LABEL_CORE_CONTENT,
                    "kind": "content"
                })
            i = j

    return _post_process(segments, duration)


def load_bundle(path: Path) -> AnalysisBundle:
    return AnalysisBundle.model_validate_json(path.read_text(encoding="utf-8"))


def write_segments_json(segments: list[dict[str, Any]], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "source": "simple_fusion_v8",
        "segments": segments,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")