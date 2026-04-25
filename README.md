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

## Visual / motion module (modality output)

The visual analyzer produces **feature windows** and optional **visual_hypothesis** hints for Leena’s fusion layer. It does **not** emit final player `label` strings; fusion + smoothing map modalities to the taxonomy above.

### Install (analysis stack)

```bash
python3 -m pip install -r requirements.txt
```

(Player UI still needs `PySide6` separately: `python3 -m pip install PySide6`.)

### CLI: `visual_track.json`

From the repository root:

```bash
python3 -m visual_analyze --video /path/to/video.mp4 --out visual_track.json
```

If your files live in the repo, use the local layout in [data/README.md](data/README.md) (e.g. `data/input/your.mp4` → `data/output/your_visual_track.json`).

**Dataset with ads:** metadata lives in [video_info/](video_info/) (`test_001.json`, …). Put the matching stitched `test_001.mp4` in **`data/input/`** or **`videos_with_ad/`** (the CLI checks both). Then:

```bash
PYTHONPATH=. python3 -m visual_analyze --video-info video_info/test_001.json --videos-root data/input \
  --bundle-out data/output/test_001_analysis_bundle.json --reference-out data/output/test_001_reference_ads.json
```

See [video_info/README.md](video_info/README.md) for details.

Optional **AnalysisBundle** (visual + empty `audio_windows` / `speech_spans` placeholders for teammates):

```bash
python3 -m visual_analyze --video /path/to/video.mp4 --out visual_track.json --bundle-out analysis_bundle.json
```

### Ingestion (shared `ffmpeg` reference)

See [scripts/ingest_example.sh](scripts/ingest_example.sh) for extracting mono 16 kHz WAV, 1 fps JPEG frames, and a small proxy MP4 into a work directory.

### Schemas (standardization)

Pydantic models live under [schemas/modality.py](schemas/modality.py): `VisualWindow`, `VisualTrack`, `AudioWindow`, `SpeechSpan`, `AnalysisBundle`.

### Fusion handoff

- Teammates append **audio** and **speech** lists into the same `AnalysisBundle`, or fusion loads separate JSON files and merges in memory.
- Example merged payload shape: [sample_data/sample_analysis_bundle.json](sample_data/sample_analysis_bundle.json) (replace `video_path` with a real file path when testing).
- Validate in Python: `AnalysisBundle.model_validate_json(path.read_text())`.

### Tests

```bash
python3 -m pytest tests/
```
