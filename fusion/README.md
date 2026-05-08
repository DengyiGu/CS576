Fusion Module

This module is the orchestration and fusion layer for the multimodal segmentation
pipeline. It consumes the visual, audio, and speech outputs and produces the
`*_segments.json` files that the player reads.

Current Workflow
----------------
The current day-to-day workflow is the batch pipeline in `scripts/run_pipeline.py`.
OCR is not used in this setup.

Step 1: Activate the conda environment

    conda activate cs576

Step 2: Run the pipeline for all videos in `videos_with_ad`

    PYTHONPATH=. python scripts/run_pipeline.py \
        --input-dir videos_with_ad \
        --model tiny \
        --vad \
        --skip-analysis \
        --skip-audio \
        --skip-speech

    Use `--force-analysis` if you want to rerun the full visual/audio pipeline
    instead of reusing existing `*_analysis_bundle.json` files. That is useful the
    first time you run a video, or if you want to regenerate bundles after making
    code changes. For day-to-day testing, reusing bundles with `--skip-analysis`
    is preferred because it is much faster.

Step 3: Evaluate the generated segments

    PYTHONPATH=. python scripts/evaluate.py

What the pipeline writes
------------------------
The pipeline writes per-video outputs into `data/output/`:

Expected runtime per video: 10-20 minutes depending on length.

Current Evaluation Results

Fusion CLI
----------
`python -m fusion` is still available for direct fusion and ad-hoc runs, but it
is no longer the primary workflow.

If you do use `python -m fusion`, there are CUDA-related options for the text
paths and ASR:

    --cuda-text-models   run OCR and semantic text scoring on CUDA when available
    --asr-device cuda    run speech recognition on CUDA instead of CPU

Those options are only useful on machines with a working CUDA/PyTorch setup.

Key files
---------

fusion/
    __init__.py
    __main__.py       (CLI entry point)
    fuse.py           (core fusion engine)

scripts/
    run_pipeline.py   (batch pipeline used for current testing)
    evaluate.py       (evaluation against ground-truth ad labels)

player_fusion.py      (drop-in replacement for `run_video_segmentation` in `player/player.py`)

Notes
-----
- Always run with `conda activate cs576` first so speech recognition is available.
- The current testing flow does not use OCR.
- If you want to test a single video, you can still pass one file to `scripts/run_pipeline.py` instead of `--input-dir`.
- The bundle-reuse path is the recommended way to iterate quickly; full reruns are mostly for first-time generation or after pipeline changes.

Segments JSON
-------------
The fusion output uses the player taxonomy and writes JSON in this shape:

    {
        "schema_version": "1.0",
        "source": "simple_fusion_v14",
        "segments": [
            {"start": 0.0, "end": 12.5, "label": "Intro", "kind": "non-content"},
            {"start": 12.5, "end": 45.0, "label": "Core Content", "kind": "content"},
            {"start": 45.0, "end": 75.0, "label": "Advertisement", "kind": "non-content"}
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
