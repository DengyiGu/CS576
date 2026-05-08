# Local data (not committed)

The `data/` folder  holds pipeline outputs . Keep generated files in
`data/output/` so they stay out of git.

mkdir -p data/output

When you run the pipeline from the repository root, it writes two JSON files per
video into `data/output/`:

<stem>_analysis_bundle.json
<stem>_segments.json

The analysis bundle contains the merged visual, audio, and speech metadata.
The segments file is the final output from the fusion module.

Source videos live outside `data/`. The `videos_with_ad/` folder is a separate
top-level directory in the repo and is where the batch pipeline reads videos
from when you use `--input-dir videos_with_ad`.

If you are working with the stitched `video_info/test_*.json` dataset, keep the
matching `test_*.mp4` files in `videos_with_ad/` or point the pipeline at a
different directory with `--input-dir`.

Full instructions for the stitched video dataset: [videos_with_ad/README.md](../videos_with_ad/README.md) and [video_info/README.md](../video_info/README.md).
