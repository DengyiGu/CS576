# Setup and run guide (Songmao branch)

End-to-end recipe to produce labeled segments from a video and play them in
the GUI player. Everything runs from the repository root with `PYTHONPATH=.`.

## 1. Install dependencies

```powershell
python -m pip install -r requirements.txt                                 # analysis core
python -m pip install -r Automatic_speech_recognition/requirements.txt    # speech recognition
python -m pip install -r requirements-player.txt                          # PySide6 (player UI only)
```

## 2. Install ffmpeg (system binary)

The audio analyzer uses `ffmpeg` to demux audio, and Qt's video backend on
Windows can use it for codec coverage.

```powershell
winget install --id Gyan.FFmpeg
# Open a new shell so the new PATH is picked up, or in the same shell:
# $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
ffmpeg -version
```

## 3. Download the speech model (one-time, ~500 MB for `small`)

```powershell
python Automatic_speech_recognition/segment_text_analyzer.py --download-model --model small
```

Other sizes: `tiny`, `base`, `small` (default), `medium`, `large-v3`.

## 3b. Download the YAMNet audio-event model (one-time, ~16 MB)

Fusion uses YAMNet's per-window music / speech probabilities as one of
its strongest ad-interior cues. Pull the ONNX export from Hugging Face
into `audio/models/`:

```powershell
python -c "from huggingface_hub import hf_hub_download; import shutil, os
os.makedirs('audio/models', exist_ok=True)
for fn in ('yamnet.onnx', 'yamnet_class_map.csv'):
    src = hf_hub_download(repo_id='zeropointnine/yamnet-onnx', filename=fn)
    shutil.copyfile(src, os.path.join('audio/models', fn))"
```

After running steps 4–5 once, you can fold YAMNet scores into existing
analysis bundles without re-running the full pipeline:

```powershell
$env:PYTHONPATH = "."
python scripts/add_yamnet_to_bundles.py
# or just one video
python scripts/add_yamnet_to_bundles.py --tests test_001
```

The script re-extracts audio via ffmpeg, runs YAMNet (~2 s of CPU time
per 30-min video), and writes `yamnet_*` fields back into each
`AudioWindow` of the cached bundle.

## 4. Drop a video in

Place a `.mp4` (or any container ffmpeg can decode) under
`videos_with_ad/` or `data/input/`. The course test set uses
`videos_with_ad/test_001.mp4` … `test_005.mp4`.

## 5. Run the pipeline

```powershell
$env:PYTHONPATH = "."
python scripts/run_pipeline.py videos_with_ad/test_001.mp4
```

This runs visual analysis -> audio analysis -> speech recognition ->
fusion and writes:

- `data/output/test_001_analysis_bundle.json` (full per-window features)
- `data/output/test_001_segments.json` (final labeled segments for the player)

Useful flags:

| Flag | Effect |
| --- | --- |
| `--skip-speech` | Skip Whisper transcription (much faster, lower accuracy) |
| `--skip-audio` | Skip our audio modality entirely |
| `--vad` | Enable Whisper voice-activity-detection filter (faster) |
| `--model {tiny\|base\|small\|medium\|large-v3}` | Choose a different Whisper model |
| `--sample-fps 1.0 --window-sec 2.0 --min-segment-sec 20` | Leena's recommended params |

## 6. Launch the player

```powershell
python -m player.player
```

Click "Open Video", pick the file you analyzed in step 5. The player loads
`data/output/<stem>_segments.json` automatically (lookup happens in
`player_fusion.py`).

If you skip step 5, the player still opens — it falls back to running the
full pipeline live (slow), and ultimately to demo segments if all else fails.

## 7. Evaluate against ground truth

For the bundled course test set:

```powershell
python scripts/evaluate.py
# or for a single test
python scripts/evaluate.py --test test_001
```

The evaluator reads `video_info/test_*.json` (ground-truth ad ranges) and
the `data/output/test_*_segments.json` files to report precision / recall /
F1 / IoU per video plus a mean.

## 8. Tests

```powershell
python -m pytest -q
```
