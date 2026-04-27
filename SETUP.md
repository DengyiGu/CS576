# Segment Text Analyzer Setup

Install Python dependencies:

```bash
python -m pip install -r Text_recognition/requirements.txt
```

Download the local speech-to-text model:

```bash
python Text_recognition/segment_text_analyzer.py --download-model
```

The model is stored under the `models/` folder next to `segment_text_analyzer.py`:

```text
models/faster-whisper-small
```

After the model is downloaded, transcription runs locally without calling the network.

Optional SpeechSpan test:

```bash
python Text_recognition/segment_text_analyzer.py videos_with_ad/test_001.mp4 --language en
```

Notes:

- `faster-whisper` is used for offline speech-to-text.
- `pydantic` is required because the shared fusion schema defines `SpeechSpan` with Pydantic.
- `huggingface_hub` is only needed to download the model during setup.
- `build_speech_spans(video_path)` returns `list[SpeechSpan]`, where each item has `t0`, `t1`, and `text`.
