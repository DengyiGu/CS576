# Stitched dataset videos (`videos_with_ad`)

Put the **`test_001.mp4` … `test_005.mp4`** files that match [video_info/](../video_info/) here (same names as in each JSON `video_filename`).

The visual CLI looks here **automatically** if the file is not under `--videos-root` (default `data/input`).

## Optional: mirror into `data/input/` (symlinks)

From the repository root:

```bash
bash scripts/sync_videos_with_ad_to_data_input.sh
```

That creates symlinks under `data/input/` pointing at these files so other tools that only read `data/input/` still work.
