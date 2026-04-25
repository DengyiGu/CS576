After the user clicks “Open Video”, the player will pass the local video file path to the algorithm:

video_path: Path

The algorithm should use this path to load and process the video, audio, frames, transcripts, etc.

The player will not pass preloaded frames or audio arrays, only the video file path.

We recommend that the algorithm returns a list[dict], where each dictionary represents a segment:


[

    {
        "start": 0.0,
        
        "end": 12.5,
        
        "label": "Intro"
        
    },
    
    {
        "start": 12.5,
        
        "end": 180.0,
        
        "label": "Core Content"
    },
    
    {
        "start": 180.0,
        
        "end": 205.0,
        
        "label": "Advertisement"
    }
    
]


The player will use build_segment_from_payload() to convert each payload dictionary into the internal Segment format required by the player.


Requirements:

1. start and end are in seconds and may be decimal values. The player converts them to milliseconds for playback seeking.

2. end must be greater than start

3. Segments should be ordered by time

4. Segments should not overlap

5. Segments should ideally cover the entire video (avoid gaps in the timeline)

6. label must match the predefined labels used in the player


Current Standard Labels:

1. Core Content

2. Intro

3. Outro

4. Advertisement

5. Self-Promotion

6. Recap

7. Transition

8. Inactivity

9. Filler


To see how segments are generated and passed to the player, start from SegmentationWorker. It runs run_video_segmentation(video_path) in the background and emits the resulting segments back to the UI.

---

## Visual pipeline (motion and appearance features)

This branch includes a **visual modality** pass over each video. It downsamples frames in time (configurable target rate, e.g. ~2 fps), resizes frames for speed, and writes **windowed** metrics (typically ~1 s windows) suitable for a downstream **fusion** step. It does **not** assign final player segment labels; those stay the job of fusion + smoothing once audio and text modalities exist.

### What is computed

- **Frame sampling:** OpenCV reads the file; frames are stepped by a stride derived from native FPS and `--sample-fps` so analysis stays tractable on long clips.
- **Motion:** Mean absolute difference of consecutive sampled grayscale frames (normalized per video with robust percentiles).
- **Luminance:** Mean grayscale intensity per window (0–1).
- **Edge density:** Canny edge pixel density per window, normalized for fusion heuristics (graphics / on-screen text hints).
- **Other signals:** Color histogram distance vs a rolling reference (palette change), optional PySceneDetect shot boundaries, coarse `visual_hypothesis` and `high_text_density` flags.

### JSON outputs

- **`VisualTrack`** — Metadata (duration, sampled FPS, stride, resolution) plus a list of **`VisualWindow`** objects (`t0`, `t1`, features above). Typical file: `*_visual_track.json`.
- **`AnalysisBundle`** — Wraps `visual` plus placeholder lists `audio_windows` and `speech_spans` for teammates. Typical file: `*_analysis_bundle.json`. Schema: [schemas/modality.py](schemas/modality.py).

### CLI usage

Install analysis dependencies, then from the **repository root**:

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH=. python3 -m visual_analyze --video /path/to/video.mp4 --out data/output/track.json
```

Optional full bundle for fusion:

```bash
PYTHONPATH=. python3 -m visual_analyze --video /path/to/video.mp4 \
  --out data/output/track.json --bundle-out data/output/bundle.json
```

**Course dataset (`video_info` + stitched `test_*.mp4`):** see [video_info/README.md](video_info/README.md) and [data/README.md](data/README.md). Example:

```bash
PYTHONPATH=. python3 -m visual_analyze --video-info video_info/test_001.json --videos-root data/input \
  --bundle-out data/output/test_001_analysis_bundle.json --reference-out data/output/test_001_reference_ads.json
```

**Player UI** still needs PySide6: `python3 -m pip install PySide6`.

### Ingestion (ffmpeg)

Shared WAV / frame / proxy extraction: [scripts/ingest_example.sh](scripts/ingest_example.sh).

### Fusion handoff

- Teammates fill **audio** and **speech** in the same `AnalysisBundle` or supply separate JSON merged before fusion.
- Example shape: [sample_data/sample_analysis_bundle.json](sample_data/sample_analysis_bundle.json).
- Validate: `AnalysisBundle.model_validate_json(path.read_text())`.

### Tests

```bash
python3 -m pytest tests/
```
