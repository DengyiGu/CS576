FUSION MODEL
Owner: Leena

This module is the orchestration and fusion layer for the multimodal segmentation pipeline. It takes outputs from the visual, audio, and speech modules and produces labeled segments.json files that the player reads.

Directory Structure:
fusion/
    __init__.py
    __main__.py  (CLI entry point)
    fuse.py      (core fusion engine)
scripts/
    evaluate.py  (evaluation against ground-truth ad labels)
player_fusion_patch.py  (drop-in replacement for run_video_segmentation in player/player.py)

Quick Start: 
1: Generate a visual analysis bundle
    PYTHONPATH=. python -m visual_analyze \
        --video data/input/test_001.mp4 \
        --bundle-out data/output/test_001_analysis_bundle.json

2: Run fusion
    PYTHONPATH=. python -m fusion \
        --bundle data/output/test_001_analysis_bundle.json \
        --out data/output/test_001_segments.json

 Or run end-to-end from raw video in one command:
    PYTHONPATH=. python -m fusion \
        --video data/input/test_001.mp4 \
        --out data/output/test_001_segments.json \
        --bundle-out data/output/test_001_analysis_bundle.json

3: Open the player
    The player auto-loads data/output/<video_stem>_segments.json when you open a video. No extra step required once the file exists.

4: Evaluate (once you have segments for test_001 through test_005)
    PYTHONPATH=. python scripts/evaluate.py


How Teammates Plug In
Audio module (Teammate 1)
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

Speech / Whisper module (Teammate 2)
    Populate AnalysisBundle.speech_spans. Each SpeechSpan carries text with
    the transcript for that time range. The fusion layer scans for keyword phrases
    (sponsorships, self-promotion, outros, intros, recaps):

        from schemas.modality import SpeechSpan

        SpeechSpan(t0=45.0, t1=60.0, text="This video is sponsored by Squarespace...")
        SpeechSpan(t0=60.0, t1=75.0, text="Use code TECHPOD for 10% off...")

    The fusion layer will automatically label those windows as Advertisement.

Signal priority (when modalities conflict)
    1. Speech  -- highest confidence, explicit keyword match
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
    Non-content: Intro, Outro, Advertisement, Self-Promotion, Recap, Transition, Inactivity, Filler

The video segmentation function checks data/output/<stem>_segments.json first (loads instantly if found),
otherwise runs the full visual and fusion pipeline live and caches the result,
and falls back to build_even_segments() if everything fails so the player never crashes.


Tuning Parameters:
    --min-segment-sec   default 4.0   segments shorter than this get absorbed by neighbors
    --sample-fps        default 2.0   visual analysis frame rate (higher = slower but more detail)
    --window-sec        default 1.0   visual analysis window size in seconds

To adjust how aggressively the classifier detects ads, edit the thresholds in fusion/fuse.py inside the _classify_visual() function.
