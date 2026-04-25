# Video dataset metadata (`video_info`)

Each `test_*.json` file describes a **stitched** video: main content plus **inserted ads**, with a `timeline_segments` array (`video_content` vs `ad`) and timestamps in **final video** seconds.

## Pairing with video files

Place the actual `.mp4` file named in `video_filename` (same as `output_filename` in these fixtures) under **`data/input/`** or under **`videos_with_ad/`** at the repo root (the CLI checks both even if `--videos-root` is `data/input`).

| JSON | Expected video path |
|------|---------------------|
| `video_info/test_001.json` | `data/input/test_001.mp4` |
| `video_info/test_002.json` | `data/input/test_002.mp4` |
| … | … |

## Run visual analysis using metadata

From the repository root:

```bash
PYTHONPATH=. python3 -m visual_analyze \
  --video-info video_info/test_001.json \
  --videos-root data/input \
  --out data/output/test_001_visual_track.json \
  --bundle-out data/output/test_001_analysis_bundle.json \
  --reference-out data/output/test_001_reference_ads.json
```

If your MP4s live only in `videos_with_ad/`, you can omit copying: resolution checks **`videos_root` first**, then **`videos_with_ad/`**, then **`data/input/`**. To also populate `data/input/` with symlinks:

```bash
bash scripts/sync_videos_with_ad_to_data_input.sh
```

- **`--reference-out`**: optional JSON with **ground-truth ad intervals** in the same `start` / `end` / `label` shape the player uses (`Advertisement`), built from `timeline_segments` where `type == "ad"`. Useful for scoring fusion against this dataset.

If the `.mp4` is missing, the CLI prints the expected path and exits without a Python traceback.
