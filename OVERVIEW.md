# Project overview — pipeline + player (Dengyi branch / demo build)

Audience: every team member. Purpose: a single document everyone can read
end‑to‑end and walk away with a holistic understanding of what each module
does, how the data flows from a raw video to coloured timeline segments
in the desktop player, and which file you'd open first if you had to
debug or extend a given piece. Setup commands live in `SETUP.md`; this
file is about *what is happening* rather than *how to install*.

> Diagram‑first: skip down to **§1 The big picture** if you only want one
> mental model and the file map. **§3 Inside fusion** and **§5 Inside the
> player** are deeper dives.

---

## 1. The big picture

### 1.1 Two front doors, one engine

```
   videos_with_ad/x.mp4
            │
            ├───────────────► CLI:  scripts/run_pipeline.py x.mp4
            │                        (or python -m fusion --video x.mp4)
            │                                  │
            │                                  ▼
            │               ┌───────────────────────────────────┐
            │               │   ANALYSIS PIPELINE               │
            └──────────────►│   visual + audio + ASR + OCR +    │──► data/output/
            (player live)   │   semantic  ──►  fusion           │    x_segments.json
                            └───────────────────────────────────┘            │
                                                                             ▼
                                       ┌─────────────────────────────────────────┐
                                       │   DESKTOP PLAYER  (python -m player.player)
                                       │   reads cached segments (or runs the    │
                                       │   pipeline itself on a worker thread)   │
                                       └─────────────────────────────────────────┘
```

The pipeline is one body of code. There are two front doors that drive it:

1. **CLI** (`scripts/run_pipeline.py` or `python -m fusion`) reads a raw
   video file and writes two JSONs into `data/output/`:
   - `<stem>_analysis_bundle.json` — every per‑window feature we extract
     from any modality (visual, audio, speech, OCR, semantic). Big.
   - `<stem>_segments.json` — the small, player‑facing list of labelled
     time segments. This is the deliverable.
2. **Player** (`python -m player.player`) is a Qt6 desktop app that
   loads a video, paints the segments on a colour‑coded timeline, and
   lets the user scrub / jump / inspect. If a segments JSON for the
   loaded video is missing, the player runs the *same* pipeline itself
   on a background thread and caches the result.

The two surfaces share **only one contract**: the segments JSON. In
both paths the same `fuse_bundle_to_segments` function does the
labelling work — the only difference is where we invoke it from.

**For the demo** we use the CLI path first (it gives us a deterministic,
pre‑validated result on stage and lets us run `scripts/evaluate.py`
before the GUI is even open), then we launch the player which loads
those cached segments instantly. Opening an *unseen* video in the GUI
still works, it just takes the time of a full pipeline run.

### 1.2 What lives where

```
CS576/
├─ scripts/run_pipeline.py        ← CLI: run every stage, write bundle + segments
├─ scripts/evaluate.py            ← P / R / F1 / IoU scorer vs video_info/*.json
├─ visual/analyze.py              ← per‑window visual features
├─ audio/analyze.py               ← per‑window audio features + label
├─ Automatic_speech_recognition/
│  └─ segment_text_analyzer.py    ← Whisper ASR → SpeechSpan list
├─ ocr/analyze.py                 ← easyocr on candidate frames → SpeechSpan
├─ semantic/analyze.py            ← MiniLM sentence embeddings → ad/structure scores
├─ fusion/
│  ├─ __main__.py                 ← CLI: --video (full run) | --bundle (fast re‑fuse)
│  ├─ fuse.py                     ← THE algorithm (~2.4 k LOC)
│  ├─ intro_detector.py           ← find_intro_end_time(): edge + semantic intro
│  ├─ outro_detector.py           ← find_outro_start_time(): edge + semantic outro
│  ├─ ad_signals.json             ← brand list + sponsorship phrases
│  └─ extra_brand_names.txt       ← optional large brand dict
├─ schemas/
│  ├─ modality.py                 ← AnalysisBundle / VisualWindow / SpeechSpan / AudioWindow
│  └─ video_info.py               ← ground‑truth schema (timeline_segments)
├─ player/
│  ├─ player.py                   ← Qt6 desktop player (1,950 LOC)
│  └─ segments.py                 ← Segment dataclass + colour TAXONOMY
└─ player_fusion.py               ← bridge: cached → live pipeline → graceful fallback
```

---

## 2. The five analysis modalities

Each modality reads the video file and emits records that conform to
`schemas/modality.py`. The schema is the lingua franca: any combination of
modalities can be active, and fusion handles missing ones gracefully.

```
                       AnalysisBundle
              ┌───────────────────────────┐
              │ visual: VisualTrack       │   one record per "window" of W sec
              │   ├ duration_sec          │   (1.0 by default, 2.0 in run_pipeline)
              │   └ windows: [VisualWindow]
              │                           │
              │ audio_windows:            │   parallel per‑window audio features
              │   [AudioWindow]           │
              │                           │
              │ speech_spans:             │   ASR + OCR + semantic, all
              │   [SpeechSpan(source=…) ] │   tagged with `source`
              └───────────────────────────┘
```

### 2.1 Visual analyzer — `visual/analyze.py`

Reads frames with OpenCV's `cv2.VideoCapture`, optionally splits scenes
with `PySceneDetect.ContentDetector`, and produces one `VisualWindow` per
fixed‑length time window. The features we emit per window:

| Field                       | Type    | Meaning                                                                         |
| --------------------------- | ------- | ------------------------------------------------------------------------------- |
| `t0`, `t1`                  | float   | window start / end in seconds                                                   |
| `motion_score`              | float   | mean abs‑diff between consecutive sampled frames, robustly normalised           |
| `luminance_mean`            | float   | average grayscale brightness                                                    |
| `edge_density`              | float   | fraction of pixels that are Canny edges                                         |
| `palette_delta`             | float   | Bhattacharyya distance between this window's BGR histogram and the previous one |
| `shot_boundary_near`        | bool    | did PySceneDetect put a cut inside this window?                                 |
| `shot_boundary_distance_sec`| float?  | distance (sec) to nearest shot cut, or `null`                                   |
| `high_text_density`         | bool    | text‑like edge clusters → likely chyron / title card / packshot                 |
| `visual_hypothesis`         | enum    | `static` \| `graphics_heavy` \| `dynamic_talk` \| `unknown` (rule‑based)        |
| `hypothesis_confidence`     | float   | 0…1 confidence in the hypothesis                                                |

The default visual cadence is 1 sec per window in the schema (raised to
2 sec by `scripts/run_pipeline.py` for speed), and the analyzer samples
~`sample_fps` frames inside each window to compute the features.

Why these and not raw pixels? Ads are short, the videos are long
(20–30 min), and we need the fusion stage to run in seconds. About a
half‑dozen floats plus a few flags per window are enough to express
most ad cues (palette switch, graphics heavy, low motion + high
text‑density, etc.).

### 2.2 Audio analyzer — `audio/analyze.py`

Demuxes audio with `ffmpeg` to 16 kHz mono PCM, then a pure NumPy/SciPy
DSP stack computes per‑window features. Each `AudioWindow` carries:

- `t0`, `t1`
- `audio_label`: `"silence" | "speech" | "music" | "mixed"` from a
  rule‑based MFCC / centroid / flatness / ZCR classifier
  (`_classify_audio_label`)
- `energy_rms` ∈ [0,1] (fusion treats < 0.02 as inactivity)
- `anomaly_score` ∈ [0,1] (fusion treats > 0.75 as advertisement)
- auxiliary `rms_db`, `zcr`, `spectral_centroid/flatness/rolloff`

The "anomaly" is not a learned model: it's a per‑video robustness score
(`_anomaly_scores`) that flags windows whose multi‑feature vector is
unusually far from the running median, which is exactly what music‑bed
or jingle ads look like inside a mostly‑speech podcast.

### 2.3 Speech recognition — `Automatic_speech_recognition/segment_text_analyzer.py`

Wraps `faster-whisper` (CPU `int8` by default). One call per video:

```python
build_speech_spans(video, model_name="small", vad=True) → list[SpeechSpan]
```

Each span has `t0`, `t1`, `text`. Voice activity detection is on by
default to skip silent regions. VAD is the single biggest speed lever —
without it Whisper transcribes through music and silence too. Models
live in `Automatic_speech_recognition/models/faster-whisper-<size>/`
and are git‑ignored; the user runs the model download once in setup.

### 2.4 OCR — `ocr/analyze.py`

`easyocr` over a *targeted* set of frames. Crucially, OCR is **not** run
on every frame; it samples only frames that look promising according to
the visual analyzer (`window.high_text_density`,
`visual_hypothesis ∈ {"graphics_heavy","static"}`,
`palette_delta > 0.35`), plus a sparse global sweep so brand text on a
non‑flagged frame still gets seen. Each detected text block is wrapped
as a `SpeechSpan` with `source="ocr"`. The fusion layer then re‑uses its
existing brand / phrase matcher on those spans without needing
OCR‑specific code paths.

### 2.5 Semantic scoring — `semantic/analyze.py`

Local sentence embeddings (`sentence-transformers` MiniLM by default).
First, `merge_text_spans` glues adjacent short ASR/OCR spans together
(max gap 4 s, max window 60 s, min length 25 chars per merged chunk)
so that we score paragraphs of text rather than two‑word fragments.
Then two scoring passes run on those merged chunks, each one comparing
positive vs. negative prompts and emitting cosine‑similarity margins:

1. `build_semantic_ad_spans` — `AD_PROMPTS` vs. `CONTENT_PROMPTS`. Emits
   a synthetic `SpeechSpan` with `source="semantic"`,
   `semantic_ad_score`, `semantic_margin`.
2. `build_semantic_structure_spans` — `INTRO_PROMPTS` vs. `OUTRO_PROMPTS`,
   giving fusion a learned signal for "this is the opening / closing of
   the video". Source tag `source="semantic_structure"`.

Both passes are gated behind a try/except in `fusion/__main__.py`, so
the rest of the pipeline still works if the user has not installed
`sentence-transformers`.

### 2.6 The unifying bundle

All of the above merge into one
`schemas.modality.AnalysisBundle` (visual + audio_windows + speech_spans).
Fusion takes a single bundle as input. The bundle is serialisable:
the `--bundle` flag of `python -m fusion` lets you re‑run fusion on a
cached bundle in seconds, which is how we iterate on the algorithm
without re‑running visual / OCR / Whisper every time.

---

## 3. Inside fusion (`fusion/fuse.py` + `intro_detector.py` + `outro_detector.py`)

This is the largest part of the project (`fuse.py` alone is ≈ 2.4 k LOC,
with intro/outro logic factored into ~200‑LOC sibling modules). The
30‑second explanation: we score every window, pick a small number of
high‑confidence ad intervals, refine each interval's boundaries, then
label the gaps between them as Intro / Core Content / Outro. The
30‑minute explanation follows.

### 3.1 Step 1 — interior "foreignness" score

`_compute_foreignness_scores(windows, audio_windows, speech_spans, duration)`
returns a `np.ndarray` of length `N` (one per visual window). Score
in [0, 1] where higher means "this window looks like the inside of an
ad". Inputs that contribute:

- **Visual**: palette change, `graphics_heavy`, `high_text_density`,
  `static + text` combination
- **Audio**: anomaly_score, energy floor, label = music / mixed
- **Speech coverage / no‑speech**: a window with no nearby ASR text is
  more ad‑like *only when* something else (visuals or audio) corroborates
- **Text signals**: brand / phrase / regex matches in nearby ASR / OCR
  spans, plus the MiniLM `semantic_ad_score`
- **Content penalty**: lecture / discussion / "for example" terminology
  reduces the score so we don't false‑positive a calculus lecture

The inputs are weighted, edges of the video are suppressed, and a
"quiet static content" rule explicitly down‑weights low‑entropy
non‑ad segments. The whole thing is pure NumPy and runs in < 1 sec
even on 30‑min videos.

### 3.2 Step 2 — boundary "edge" score

`_compute_edge_scores` produces a parallel array, but indexed *between*
windows. A high edge value at boundary *i* means "a hard cut likely
happened here". Inputs:

- visual delta across the boundary (`palette_delta` of *cur*, plus
  edge / luminance / motion / text‑density transitions)
- audio anomaly delta and audio label transition
- speech presence transition (had speech before but not after, or
  vice versa)

Edge scores are what the candidate scorer uses to pick the *boundaries*
of an ad, and foreignness is what it uses to pick the *interior*.

### 3.3 Step 3 — candidate ad intervals (`_find_ad_intervals`)

This is the workhorse. We **brute‑force enumerate** every `(s, e)`
window pair whose duration is between `AD_MIN_SEC = 28` s and
`AD_MAX_SEC = 130` s (with cumulative‑sum tricks so the inner loop is
constant‑time), score each candidate, and keep the strong ones. The
score combines:

- `EDGE_WEIGHT * (norm_edge[s] + norm_edge[e])` — sharp cuts at both ends
- `INTERIOR_WEIGHT * mean(foreignness[s:e])` — interior looks ad‑like
- `TEXT_WEIGHT * mean(text_scores[s:e])` — explicit ad text inside
- `DIRECTION_WEIGHT * direction_score` — adness rises into the
  interval and falls after it (the "this is bracketed by content" cue)
- `- CONTENT_PENALTY_WEIGHT * mean(content_penalties[s:e])` — punish
  candidates that lie inside lecture / discussion vocabulary

There's a parallel `_find_text_anchor_intervals` that anchors candidates
on strong ASR / OCR text matches even when the boundary edges aren't
sharp — this catches voice‑over ads with no scene cut.

### 3.4 Step 4 — suppress, refine, expand

After collecting candidates `fuse_bundle_to_segments` runs a sequence of
post‑processing passes (each is its own function in `fuse.py`):

1. **First dedup pass** — if `_find_text_anchor_intervals` returned
   anything, combine those with the visual/audio candidates and run
   `_suppress_close_ad_intervals` once on the union to drop overlaps.
2. **Per‑candidate refinement loop** (in order):
   1. `_refine_boundary` for the start, then for the end — slide each
      boundary up to ±12 s onto the nearest local edge maximum.
   2. `_expand_ad_interval` — short ads with strong text anchors or
      strong interior visuals are permitted to grow outward (up to
      ~25 s for text‑driven, ~12 s for visual‑driven) as long as the
      new windows keep "ad‑like" feature signatures.
   3. `_passes_final_ad_filters` — if the expanded interval no longer
      passes (minimum support, edges, etc.) we either revert to the
      pre‑expansion base interval or drop the candidate entirely.
   4. `_optimize_interval_boundaries` — local search ±28 s around each
      refined boundary for a higher‑scoring pair.
   5. `_passes_final_ad_filters` once more on the optimised interval.
3. **Second dedup pass** — `_suppress_close_ad_intervals` again on the
   refined set, in case refinement collapsed two candidates onto the
   same ad.
4. `_trim_text_anchor_tails` and `_trim_confirmed_ad_tails` — pull the
   tail back when the post‑ad text shows obvious "back to content"
   cues (lecture vocabulary, content phrases, etc.).

### 3.5 Step 5 — emit segments (`_build_segments_from_ad_intervals`)

The kept ad intervals split the video into runs. Each run is either:

- An **ad** run → labelled `Advertisement`.
- A non‑ad run → fed to `_build_non_ad_segments`, which decides
  per‑window whether the head of the run should be `Intro` (using
  `find_intro_end_time` from `fusion/intro_detector.py` — combines
  edge cues, opening‑title vocabulary, and `semantic_structure`
  spans) or the tail should be `Outro` (`find_outro_start_time` from
  `fusion/outro_detector.py` — symmetric logic with closing‑title
  vocabulary). Everything else stays as `Core Content`.

A pair of cleanup passes (`_merge_adjacent_segments`,
`_merge_short_core_content_segments`) collapses tiny gaps so the player
gets clean blocks.

### 3.6 The final JSON

`fuse_bundle_to_segments` returns a list of `dict` records. They are
written by `write_segments_json`:

```json
{
  "schema_version": "1.0",
  "source": "fusion",
  "segments": [
    { "start": 0.0,   "end": 12.0,  "label": "Intro",          "kind": "non-content" },
    { "start": 12.0,  "end": 122.0, "label": "Core Content",   "kind": "content"     },
    { "start": 122.0, "end": 152.0, "label": "Advertisement",  "kind": "non-content" },
    …
  ]
}
```

`label` is one of *Core Content*, *Intro*, *Outro*, *Advertisement*.
`kind` is `content` or `non-content`. That is the **only** contract the
player needs.

---

## 4. The orchestrators

The same fusion code can be driven from four entry points. They differ
only in *which stages they actually run* before they call into
`fuse_bundle_to_segments`:

| Entry point                                | Runs visual? | Runs audio? | Runs ASR/OCR/semantic? | Use it when                                                                 |
| ------------------------------------------ | :----------: | :---------: | :--------------------: | --------------------------------------------------------------------------- |
| `scripts/run_pipeline.py V`                | yes          | yes         | yes                    | One‑shot end‑to‑end CLI run with stage‑by‑stage progress printing.         |
| `python -m fusion --video V`               | yes          | yes         | yes                    | Same, but uses fusion's CLI shape (`--bundle-out`, `--cuda-text-models`).  |
| `python -m fusion --bundle B`              | bundle reuse | only if missing | only if missing    | Fast re‑fuse: visual must already be in the bundle; audio/ASR/OCR/semantic only run if their lists are empty in the bundle. |
| Player → `run_video_segmentation(video)`   | yes          | yes         | yes                    | The player itself drives the full pipeline on a Qt worker thread when no cached segments JSON is found. |

All four end up writing the same `data/output/<stem>_segments.json`,
and the player picks it up the next time you open the video.

---

## 5. Inside the player (`player/player.py` + `player/segments.py` + `player_fusion.py`)

The player is a Qt6 desktop app (`PySide6`). It's intentionally
self‑contained: pip install one requirements file, run one command,
nothing else.

### 5.1 Files

- `player/segments.py` — the `Segment` dataclass (id, start, end,
  label, kind, color), the `TAXONOMY` table that maps each label to
  the colour shown on the timeline, and the helper that turns a raw
  segments JSON record into a `Segment`.
- `player_fusion.py` — `run_video_segmentation(video_path)`, the
  function the GUI calls when a video is opened. **This is the bridge
  that lets the player drive the whole pipeline by itself.** Lookup
  order:
  1. `data/output/<stem>_segments.json` ← happy path, < 1 s load
  2. same file next to the video itself
  3. **live fusion** via `_run_fusion_live` (visual → audio → speech →
     OCR → semantic → fuse), then *writes the segments JSON back* to
     `data/output/` so the next open is instant
  4. graceful fallback: a single "Core Content" segment so the video
     still plays even if some modality fails

  The third path means the player is a one‑stop tool: a non‑technical
  user can hand the player a brand‑new MP4 and the player orchestrates
  every modality on its own. The CLI pipeline (`run_pipeline.py`,
  `python -m fusion`) is just the same code path exposed for batch
  runs and evaluation.
- `player/player.py` — the GUI itself.

### 5.2 GUI structure (top‑level Qt widgets)

```
PlayerWindow (QMainWindow)
├─ video_widget : PlayerVideoWidget (QVideoWidget subclass)
│      └─ click toggles play, F toggles fullscreen
├─ position_slider : PositionSlider (custom QSlider)
│      └─ click‑to‑seek anywhere on the track
├─ SegmentTimelineWidget
│      └─ paints all segments as coloured boxes,
│         playhead as a vertical line,
│         hover / selected / active states
├─ segment_table : QTableWidget
│      └─ list of segments with current one highlighted
├─ segment_badge_label : QLabel "Now playing: Advertisement (12.0–32.0s)"
└─ transport row: Play, Prev/Next segment, Volume, Speed, Fullscreen
```

`SegmentTimelineWidget.paintEvent` is the visual centrepiece: it
proportionally maps each segment's `[start, end]` to a coloured
rectangle on the track, blends the playhead line, and keeps the hovered
segment highlighted. Colour comes straight from
`player.segments.TAXONOMY`.

### 5.3 What happens when you press "Open Video"

```
open_video()
  └ load_video_path(path)
       ├ self.player.setSource(QUrl.fromLocalFile(path))   ← Qt starts decoding
       ├ self.show_processing_overlay()                    ← spinner + grey overlay
       └ QTimer.singleShot(0, self.process_current_video)
                ├ SegmentationWorker(path) on a QThread
                │      └ run_video_segmentation(path)      ← player_fusion.py
                │             ├─ cache hit  → return cached segments  (< 1 s)
                │             ├─ no cache   → live full pipeline:
                │             │     visual → audio → ┐
                │             │                      ├─ Whisper ASR (CPU)
                │             │                      └─ easyocr     (GPU/CPU)
                │             │           ↑ ASR + OCR run in parallel via
                │             │             ThreadPoolExecutor(max_workers=2)
                │             │     → semantic → fuse → write JSON
                │             └─ fallback   → one "Core Content" segment
                ├ on success → on_processing_finished(segments)
                │      └ apply_segments(...) → refresh_ui() → repaint timeline
                └ on failure → on_processing_failed(msg) → toast error
```

The processing happens on a `QThread` so the UI never freezes — even
during a 10‑min live fusion call. The overlay is dismissed only after
`apply_segments` has run, so the user never sees a half‑populated UI.

A subtle but important property: the live‑fusion path **caches** its
result back into `data/output/<stem>_segments.json`. So if a teammate
hands you a new video and you run it through the GUI once, the next
open of that same file is instant — and `scripts/evaluate.py` can
score the result without re‑running anything.

### 5.4 Navigation

- Position slider: drag, click, or mousewheel to seek.
- Segment table: clicking a row jumps the playhead to that segment.
- `Prev / Next segment` buttons: walk through `navigation_segments()`
  (which respects the "Content only" toggle, see below).
- Keyboard: `Space` toggles play, `F` toggles fullscreen, `Esc` exits.

### 5.5 Content‑only toggle

`on_content_only_toggled` filters `navigation_segments` to only
`kind == "content"` items. That gives the user a "skip the ads" mode
that uses *our* labels, not Qt's. This is the demo‑worthy moment for
the project: clicking that toggle and seeing the video skip past every
detected `Advertisement`/`Intro`/`Outro` block.

---

## 6. Evaluation (`scripts/evaluate.py`)

Reads ground truth from `video_info/test_*.json`
(see `schemas/video_info.py`: each ground truth has a list of
`timeline_segments` with `type ∈ {"video_content", "ad"}`). For each
test it computes:

- **Temporal precision** at 0.1 s resolution — of all the time we
  predicted as "ad", how much actually was an ad?
- **Temporal recall** — of all real ad time, how much did we catch?
- **F1** — harmonic mean of the two
- **Mean segment IoU** — for each predicted ad segment, the maximum IoU
  with any reference ad, averaged across predictions

Run `PYTHONPATH=. python scripts/evaluate.py` from repo root for a per‑
test table plus a mean across the eight test videos. This is the number
the deck will quote on the "results" slide.

---

## 7. Where to start if you have to debug

| Symptom                                              | First file to open                                                                |
| ---------------------------------------------------- | --------------------------------------------------------------------------------- |
| Player crashes on open                               | `player/player.py` `PlayerWindow.__init__`                                        |
| Player loads but no segments shown                   | `player_fusion.py` `_find_segments_file`                                          |
| Wrong number of ads detected                         | `fusion/fuse.py` `_find_ad_intervals` + `_suppress_close_ad_intervals`            |
| Ad boundaries slightly off                           | `fusion/fuse.py` `_refine_boundary` / `_expand_ad_interval` / `_optimize_…`       |
| Intro / Outro mislabelled                            | `fusion/intro_detector.py` / `fusion/outro_detector.py` (called from `_build_non_ad_segments` in `fuse.py`) |
| OCR / semantic missing                               | check `try/except` in `fusion/__main__.py` and `_try_add_*`                       |
| Whisper failing                                      | `Automatic_speech_recognition/segment_text_analyzer.py`, run with `--download-model` |
| Audio analyzer can't decode                          | `audio/analyze.py` → likely `ffmpeg` not on PATH (`SETUP.md` step 2)              |

---

## 8. One‑page recap (paste into a slide)

- **Inputs**: any decode‑able video file, no annotations required.
- **Outputs**: a small `<stem>_segments.json` with `Intro / Core Content
  / Advertisement / Outro` blocks. Native granularity is the visual
  window length (1–2 sec); refinement and optimisation can place
  boundaries at sub‑window times.
- **5 modalities feed 1 fuser**: visual features (motion / edges /
  palette / text‑density / hypothesis), audio features (RMS / MFCC /
  spectral / anomaly), Whisper ASR, easyocr text, MiniLM sentence
  embeddings (ad / content / intro / outro prompts).
- **Fusion is rule‑based scoring + heuristic refinement**: per‑window
  foreignness score + per‑boundary edge score → brute‑force enumerate
  candidate ad intervals → suppress / refine / expand / optimise →
  emit labelled segments. Pure NumPy, ~1 s on a 30‑min video once the
  bundle is cached.
- **Player is Qt6 (PySide6)**: `python -m player.player`. Loads the
  cached segments JSON (or falls back to live fusion). Coloured
  timeline + segment table + "Content only" navigation toggle.
- **Evaluation**: `scripts/evaluate.py` measures temporal P / R / F1 +
  mean segment IoU against `video_info/test_*.json`.
