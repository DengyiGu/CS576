Fusion Module

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


Expected runtime per video: 10-20 minutes depending on length.

Current Evaluation Results
==========================================================================================
  Advertisement Detection Evaluation
==========================================================================================
Test        Ref Ads   Pred Ads  Ref Sec     Pred Sec    Precision   Recall    F1        Mean Seg IoU 
------------------------------------------------------------------------------------------
test_001    3         3         178.7s      160.0s      0.953       0.854     0.901     0.726        
test_002    3         3         150.3s      84.0s       0.000       0.000     0.000     0.000        
test_003    3         3         186.0s      130.0s      0.625       0.438     0.515     0.469        
test_004    3         3         135.9s      138.0s      0.564       0.572     0.568     0.497        
test_005    3         3         105.2s      74.0s       0.000       0.000     0.000     0.000        
------------------------------------------------------------------------------------------
MEAN (n=5)                                              0.429       0.373     0.397     0.338        
==========================================================================================

Metrics:
  Precision    — of predicted ad time, how much was truly an ad
  Recall       — of reference ad time, how much did we detect
  F1           — harmonic mean of precision and recall
  Mean Seg IoU — average best-match IoU per predicted segment


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


Notes
- Always run with conda activate cs576 first, otherwise speech recognition will be skipped.
- The cs576 environment requires Python 3.10+. The base anaconda environment (Python 3.8)
  is not compatible with faster-whisper.
- To skip speech recognition for a faster run, add --skip-speech to the fusion command.
- visual/analyze.py has a bug fix applied (histogram shape mismatch). If you pull a newer
  version of analyze.py from a teammate, make sure it includes the _HIST_SIZE fix or
  the pipeline will crash.