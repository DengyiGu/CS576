Fusion Module
Owner: Leena

This module is the orchestration and fusion layer for the multimodal segmentation
pipeline. It takes outputs from the visual, audio, and speech modules and produces
labeled segments.json files that the player reads.


Directory Structure
fusion/
    __init__.py
    __main__.py       (CLI entry point)
    fuse.py           (core fusion engine)

scripts/
    evaluate.py       (evaluation against ground-truth ad labels)

player_fusion.py      (drop-in replacement for run_video_segmentation in player/player.py)

visual/analyze.py     (bug fix applied: histogram shape mismatch in _hist_distance)


Quick Start
Step 1: Activate the cs576 conda environment (required for speech recognition)

    conda activate cs576

Step 2: Run fusion end-to-end on a video

    PYTHONPATH=. python -m fusion \
        --video videos_with_ad/test_001.mp4 \
        --out data/output/test_001_segments.json \
        --bundle-out data/output/test_001_analysis_bundle.json \
        --min-segment-sec 20 \
        --sample-fps 1.0 \
        --window-sec 2.0

Step 3: Open the player and load the video. Segments load automatically.

Step 4: Evaluate against ground truth

    PYTHONPATH=. python scripts/evaluate.py


Recommended Run Parameters
These parameters gave the best results across all 5 test videos:

    --min-segment-sec 20    absorbs short blips, keeps real ads intact
    --sample-fps 1.0        1 frame per second (faster than default 2.0)
    --window-sec 2.0        2-second analysis windows (faster than default 1.0)

Expected runtime per video: 10-20 minutes depending on length.
Run videos in pairs using & to save time:

    PYTHONPATH=. python -m fusion --video videos_with_ad/test_001.mp4 --out data/output/test_001_segments.json --min-segment-sec 20 --sample-fps 1.0 --window-sec 2.0 &
    PYTHONPATH=. python -m fusion --video videos_with_ad/test_002.mp4 --out data/output/test_002_segments.json --min-segment-sec 20 --sample-fps 1.0 --window-sec 2.0 &
    wait

    PYTHONPATH=. python -m fusion --video videos_with_ad/test_003.mp4 --out data/output/test_003_segments.json --min-segment-sec 20 --sample-fps 1.0 --window-sec 2.0 &
    PYTHONPATH=. python -m fusion --video videos_with_ad/test_004.mp4 --out data/output/test_004_segments.json --min-segment-sec 20 --sample-fps 1.0 --window-sec 2.0 &
    wait

    PYTHONPATH=. python -m fusion --video videos_with_ad/test_005.mp4 --out data/output/test_005_segments.json --min-segment-sec 20 --sample-fps 1.0 --window-sec 2.0


Current Evaluation Results (visual + speech, no audio yet)
    test_001    F1: 0.772    Precision: 0.769    Recall: 0.774    IoU: 0.643
    test_002    F1: 0.109    Precision: 0.092    Recall: 0.134    IoU: 0.111
    test_003    F1: 0.000    Precision: 0.000    Recall: 0.000    IoU: 0.000
    test_004    F1: 0.248    Precision: 0.770    Recall: 0.147    IoU: 0.388
    test_005    F1: 0.397    Precision: 1.000    Recall: 0.247    IoU: 0.864
    MEAN        F1: 0.305    Precision: 0.526    Recall: 0.261    IoU: 0.401

test_001 performs well. test_003 and test_005 are limited by ads with no speech
and visually similar content (animated film, nature documentary). The audio module
is expected to improve these significantly.


How Teammates Plug In

Audio module

    Populate AnalysisBundle.audio_windows before passing the bundle to fusion.
    Each AudioWindow can carry extra fields (pydantic extra="allow"):

        from schemas.modality import AudioWindow

        AudioWindow(
            t0=12.5,
            t1=13.5,
            audio_label="music",    # "speech" | "music" | "silence" | "mixed"
            energy_rms=0.04,        # float 0-1
            anomaly_score=0.82,     # float 0-1; above 0.75 triggers Advertisement
        )

    The fusion layer will:
        - Label windows as Inactivity if audio_label == "silence" or energy_rms < 0.02
        - Label windows as Advertisement if anomaly_score > 0.75

Speech module (already integrated)

    build_speech_spans(video_path) from Automatic_speech_recognition/segment_text_analyzer.py
    is called automatically. No changes needed.


Signal priority (when modalities conflict)

    1. Speech  -- highest confidence, explicit keyword and brand name match
    2. Audio   -- silence triggers Inactivity; high anomaly triggers Advertisement
    3. Visual  -- baseline, always available


Segments JSON Format
    {
        "schema_version": "1.0",
        "source": "fusion",
        "segments": [
            {"start": 0.0,   "end": 12.5,  "label": "Intro",        "kind": "non-content"},
            {"start": 12.5,  "end": 45.0,  "label": "Core Content", "kind": "content"},
            {"start": 45.0,  "end": 75.0,  "label": "Advertisement","kind": "non-content"},
            {"start": 75.0,  "end": 300.0, "label": "Core Content", "kind": "content"}
        ]
    }

Valid labels (must match player TAXONOMY exactly):
    Content:     Core Content
    Non-content: Intro, Outro, Advertisement, Self-Promotion, Recap,
                 Transition, Inactivity, Filler


Player Integration
In player/player.py, run_video_segmentation has been replaced with:

    from player_fusion import run_video_segmentation

The patch checks data/output/<stem>_segments.json first (loads instantly if found),
otherwise runs the full visual, speech, and fusion pipeline live and caches the result,
and falls back to build_full_content_segment() if everything fails.


Notes
- Always run with conda activate cs576 first, otherwise speech recognition will be skipped.
- The cs576 environment requires Python 3.10+. The base anaconda environment (Python 3.8)
  is not compatible with faster-whisper.
- To skip speech recognition for a faster run, add --skip-speech to the fusion command.
- visual/analyze.py has a bug fix applied (histogram shape mismatch). If you pull a newer
  version of analyze.py from a teammate, make sure it includes the _HIST_SIZE fix or
  the pipeline will crash.