# Local data (not committed)

Put your **source videos** under `data/input/` (optionally in a subfolder like **`data/input/videos_with_ads/`** — the CLI finds `test_001.mp4` there automatically). Write analyzer outputs to `data/output/`. Those directories are listed in `.gitignore` so large files stay off git.

```bash
mkdir -p data/input data/output
# Copy or symlink your .mp4 / .mov / etc. into data/input/
```

Run the visual module from the **repository root**:

```bash
python3 -m visual_analyze \
  --video "data/input/your_video.mp4" \
  --out "data/output/your_video_visual_track.json" \
  --bundle-out "data/output/your_video_analysis_bundle.json"
```

Optional ingestion (WAV + frames) for other teammates:

```bash
./scripts/ingest_example.sh "data/input/your_video.mp4" "data/output/ingest"
```

Share `*_visual_track.json` or the merged `*_analysis_bundle.json` with Leena for fusion when the other modality fields are filled in.

### `video_info` dataset (stitched videos with ads)

If you use the bundled **`video_info/test_*.json`** descriptors, place each **`test_*.mp4`** under **`data/input/`** or under **`videos_with_ad/`** at the repo root (the CLI searches both). Optional: `bash scripts/sync_videos_with_ad_to_data_input.sh` symlinks from `videos_with_ad/` into `data/input/`. Full instructions: [video_info/README.md](../video_info/README.md) and [videos_with_ad/README.md](../videos_with_ad/README.md).
