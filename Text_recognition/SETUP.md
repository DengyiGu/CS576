# Segment Text Analyzer Setup

Install Python dependencies:

```bash
python -m pip install -r Final_project/requirements.txt
```

Download the local speech-to-text and text-classification models:

```bash
python Final_project/segment_text_analyzer.py --download-model
```

The models are stored under the `models/` folder next to `segment_text_analyzer.py`:

```text
models/faster-whisper-small
models/distilbart-mnli-12-1
```

After the models are downloaded, transcription and label suggestion run locally without calling the network.

Optional transcription test:

```bash
python Final_project/segment_text_analyzer.py Final_project/videos_with_ads/test_001.mp4 --ranges-json "[[0, 8]]" --language en
```

Optional label suggestion test:

```bash
python Final_project/segment_text_analyzer.py Final_project/videos_with_ads/test_001.mp4 --ranges-json "[[0, 8]]" --suggest-labels --language en
```

Notes:

- `faster-whisper` is used for offline speech-to-text.
- `transformers`, `torch`, and `torchvision` are used for offline zero-shot text classification support. `torch>=2.6` is required for loading local PyTorch model weights safely.
- `huggingface_hub` is only needed to download the models during setup.
