# Session notes — last updated 2026-05-07

Working notes for the ad-detection / segmentation pipeline. Tracked on
`tuning_audio` so it syncs across machines.

---

## Where we left off (2026-05-07 00:15 PT)

### Pipeline state
- **Mean F1 = 0.648** across the 6 cached test videos (raw baseline 0.380 → +0.268, **+70% relative**).
- All work this session lives on branch `tuning_audio`, ahead of origin by 8 commits as of this note.
- 36/36 unit tests passing.
- Demo date is **May 6/7/8 2026** (Project_Spring2026.pdf). The demo
  video is supposed to drop **20 hours before our slot**, giving us a
  buffer to retune if needed.

### Today's commits (in order)
- `7dc3994` spectral-flatness foreignness signal (mean F1 0.480 → 0.510)
- `396f121` YAMNet music/speech with per-video baseline subtraction (0.510 → 0.552)
- `d04b94b` Auto-K interior-mean floor + soft MIN_NUM_ADS (0.552 → 0.584)
- `ffca0c8` Boundary extension along foreignness signal (0.584 → **0.648**)
- `af8e54b` `scripts/inspect_video.py` consolidated triage report

### Per-video F1 right now

| Video    | start of session | now       | delta     |
|----------|------------------|-----------|-----------|
| test_001 |          0.501   |    0.935  |   +0.434  |
| test_002 |          0.421   |    0.728  |   +0.307  |
| test_003 |          0.272   |    0.421  |   +0.149  |
| test_004 |          0.498   |    0.949  |   +0.451  |
| test_005 |          0.358   |    0.426  |   +0.068  |
| test_010 |          0.301   |    0.431  |   +0.130  |
| **mean** |       **0.380**  | **0.648** |**+0.268** |

### Outstanding (in priority order for the demo window)

**Pre-demo (now):**
1. Demo polish in the player: confidence per segment, "Why was this an
   ad?" inspector panel, hotkeys (`A`=jump-to-next-ad, `C`=jump-to-next-content).
2. 1-page README/writeup: pipeline diagram, taxonomy, F1 table,
   modality-contribution ablation.
3. Soft-floor `MIN_NUM_ADS` from 3 → 1 so the pipeline gracefully
   handles videos with 0/1/2 ads (podcasts, lectures, vlogs). The
   interior collapse rule + the soft floor already partially handle
   this — test_002 correctly drops to K=2 — but we don't have a clean
   "K=0, no ads found" code path.

**During the 20-hour window after the demo video drops:**
1. `python scripts/inspect_video.py --video videos_with_ad/<name>.mp4`
   first; read the diagnostic hints. They encode the heuristics built
   up over this session ("brand list 0 hits → edit BRAND_NAMES",
   "shot-boundary density >50% → scene_cut saturating", etc.).
2. Re-tune knobs (brand list, EXTEND_KEEP_RATIO, MIN_INTERIOR_MEAN_FLOOR)
   only if a hint says so. Don't touch the DP scoring shape this close
   to demo.

**Don't do (these were considered and deferred):**
- Sentence-embedding discontinuity (MiniLM ~90 MB) — structural change,
  ~3-5 h, can introduce regressions, unlikely to move demo grade.
- Tiny learned classifier over per-window features — needs LOO-CV on
  6 videos to avoid over-fit, risk vs reward is bad pre-demo.
- DP rescore (`edges + λ·sum_foreign − μ·K`) — would resolve the
  test_005 K=4-vs-K=5 question and the under-extension that boundary-
  extension currently patches post-hoc, but requires re-tuning every
  weight.

---

## Tooling: `scripts/inspect_video.py` (DONE — 2026-05-07)

Single-command triage report so we don't have to mentally stitch
together 5 separate diagnostics when a new video lands. Runs in ~10 s
on a cached bundle. Falls back to `scripts/run_pipeline.py` for fresh
videos. Supports `--no-gt` for demo videos without a `video_info` file.

Sections:
1. **Modality health** — per-modality stats (visual cuts/min, audio
   anomaly distribution, YAMNet music/speech distribution, transcript
   span count + speech coverage, brand_hits, lexicon_hits). Catches
   "X just isn't running" failures fast.
2. **Ground truth** — GT ad list, formatted with mm:ss timestamps.
3. **Auto-K decision trace** — full per-K table walking the actual
   algorithm. `step` column shows `accept` / `STOP -- ratio …` /
   `STOP -- interior …` / `(skipped after stop)` for every K. Then
   explicit lines: `>> rule walked to K=N`, `>> [soft floor
   explanation]`, `>> chosen K = M`. Makes the auto-K choice
   *auditable*, not black-box.
4. **Picks** — TP/FP labels + per-modality boundary signals
   (`edge_s/edge_e/pal_s/pal_e/lj_s/lj_e/int`). When GT exists, lets
   you immediately classify a wrong pick as "wrong K", "wrong
   location", or "right location, wrong boundaries".
5. **Score** — temporal P/R/F1 if GT exists, otherwise just segment
   counts.
6. **Diagnostic hints** — heuristic suggestions encoded from the
   intuitions built up over this session. Examples that fire on the
   cached set:
   - test_001: speech covers 89% → expect audio + YAMNet to drive picks
   - test_002: interior cleanly collapses K=2 → K=3 → K=2 well-supported
   - test_003: 62% shot-boundary density → scene_cut signal saturating
   - test_005: hit MAX_NUM_ADS=5 → may be a hard ceiling

Verified on test_001 (soft-floor raise K=1→K=3), test_002 (interior
collapse suppresses soft floor, K=2), test_003 (cut-saturation), and
test_005 (MAX-hit). Each report also writes to
`data/output/<name>_inspect_report.txt` for sharing.

---


- Synced `Songmao` with everything else on the remote: `main` (ASR rename) and `Leena` (fusion + evaluate + player_fusion + visual histogram bug fix).
- Installed every runtime dependency the full pipeline needs (Python deps + ffmpeg + faster-whisper `small` model).
- Wrote `scripts/run_pipeline.py` so the whole pipeline (visual + audio + speech + fusion) is one command.
- Player UI is verified to launch (`python -m player.player`).
- 20/20 tests passing.
- Branch is integrated with every teammate; no incoming work pending.

The next gating item is **getting a real video** under `videos_with_ad/` so we can produce numbers and tune.

---

## Where we started this session

| | Before |
|---|---|
| `Songmao` tip | `9be3e9a update` (audio module + speech wiring stub) |
| `audio_analyze/__main__.py` | Imported `Text_recognition.segment_text_analyzer` (broken on remote) |
| `requirements-player.txt` | did not exist |
| `scripts/run_pipeline.py` | did not exist |
| `SETUP.md` | did not exist |
| `Automatic_speech_recognition/` | did not exist locally |
| `fusion/` | did not exist locally |
| `scripts/evaluate.py` | did not exist locally |
| `player_fusion.py` | did not exist locally |
| `visual/analyze.py` | had the histogram size bug (24 vs 512 elements) |
| ffmpeg / PySide6 / faster-whisper | not installed |

## What changed on the remote since last session

| Branch | New commits | Notes |
|---|---|---|
| `origin/main` | 7 | Renamed `Text_recognition/` → `Automatic_speech_recognition/`, added `--model`, `--vad`, `--compute-type`, `--download-model` to ASR. New `build_speech_spans` signature: `(video, *, model_name='small', model_dir=None, compute_type='int8', language='en', vad=False)`. |
| `origin/Leena` | 3 | Added `fusion/` (`fuse.py`, `__main__.py`, `README.md`), `scripts/evaluate.py`, `player_fusion.py`. Fixed visual histogram bug. Tuned brand-name density detection for the 5 test videos. Already merged main into her branch. |
| `origin/Murali` | 0 | already in main via PR #1 |
| `origin/Dengyi` | 0 | identical to old main |

## Sequence of work performed today

1. **Confirmed merge plan** — dry-ran `git merge origin/main` and `git merge origin/Leena` against `Songmao`, both conflict-free.
2. **`git merge origin/main` into `Songmao`** → commit `7596945` Merge origin/main into Songmao.
3. **Patched `audio_analyze/__main__.py`** to import from `Automatic_speech_recognition` and call the new `build_speech_spans` signature (kwargs `model_name`, `model_dir`, `compute_type`, `language`, `vad`). Added CLI flags `--model {tiny|base|small|medium|large-v3}`, `--vad`, `--compute-type`, plus `--model-dir` alias. Updated docstring.
4. **User committed CLI fix** as `c5bd5a8 "Updated with main"` and pushed.
5. **`git merge origin/Leena` into `Songmao`** → commit `d68e731`. Confirmed `player/player.py` already imports `from player_fusion import run_video_segmentation` (came in via Leena's player.py update).
6. **Installed Python runtime deps:**
   - PySide6 6.11.0 (player UI)
   - scenedetect 0.6.7.1 (visual)
   - faster-whisper 1.2.1 + ctranslate2 4.7.1 + onnxruntime 1.25.1 + av 17.0.1 (speech)
   - huggingface_hub (model download)
7. **Installed ffmpeg via winget** (`Gyan.FFmpeg`, version 8.1). PATH refresh required in each new shell:
   ```powershell
   $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
   ```
8. **Downloaded faster-whisper small model** to `Automatic_speech_recognition/models/faster-whisper-small/` (~500 MB).
9. **Created `requirements-player.txt`** — PySide6 only, kept separate from the analysis core deps.
10. **Created `scripts/run_pipeline.py`** — one-shot end-to-end runner: visual → audio (with optional speech) → fusion → segments JSON.
11. **Created `SETUP.md`** at repo root — install + run guide.
12. **Smoke tests** — 20/20 pytest pass; player window opens cleanly; pipeline `--help` returns in 0.06 s.
13. **User committed** scripts + setup as `b78483e "Player test 1"` and pushed.
14. **Re-fetched all remotes** — confirmed `git log Songmao..origin/<branch>` is empty for `main`, `Leena`, `Murali`, `Dengyi`. Branch is fully integrated.

## Final repo state

```
Songmao (= origin/Songmao)  b78483e  Player test 1
                            d68e731  Merge origin/Leena
                            c5bd5a8  Updated with main
                            7596945  Merge origin/main
                            9be3e9a  update                            (prior)
                            8423889  text recognition and audio baseline (prior)
```

`git log Songmao..origin/main` empty.
`git log Songmao..origin/Leena` empty.

### Files added/changed this session

- `audio_analyze/__main__.py` — Automatic_speech_recognition wiring + new flags
- `requirements-player.txt` — PySide6
- `scripts/run_pipeline.py` — end-to-end runner
- `SETUP.md` — install + run guide
- (merged in) `fusion/`, `scripts/evaluate.py`, `player_fusion.py`, `Automatic_speech_recognition/`

### System changes on this machine

- Pip-installed: PySide6, scenedetect, faster-whisper, huggingface_hub (and their deps)
- Winget-installed: `Gyan.FFmpeg`
- Downloaded: faster-whisper `small` model to `Automatic_speech_recognition/models/faster-whisper-small/`

---

## How to actually run things

End-to-end on a video:

```powershell
$env:PYTHONPATH = "."
python scripts/run_pipeline.py videos_with_ad/test_001.mp4
python -m player.player
# In the GUI: Open Video -> select test_001.mp4 -> segments load from data/output/test_001_segments.json
```

Faster (no speech):

```powershell
python scripts/run_pipeline.py videos_with_ad/test_001.mp4 --skip-speech
```

Evaluate against ground truth:

```powershell
python scripts/evaluate.py            # all tests
python scripts/evaluate.py --test test_001
```

Tests:

```powershell
python -m pytest -q
```

If a new shell can't find ffmpeg, refresh PATH from registry (see step 7 above).

---

## Pending / next steps (in priority order)

1. **Drop a real video** under `videos_with_ad/` (`test_001.mp4` … `test_005.mp4`) to actually exercise the pipeline. Without one, the player only ever sees the demo fallback.
2. **Sanity-check vs Leena's published numbers** on `test_001` (her F1 = 0.772 with no audio) — make sure our audio integration doesn't *hurt* the cases that already work.
3. **Push `test_003` / `test_005` numbers up** — these are the audio-dominant failures (Leena's F1 = 0.000 / 0.397). Likely tuning targets:
   - `_anomaly_scores` percentile mapping in `audio/analyze.py` (currently 50→0, 95→1).
   - `_classify_audio_label` thresholds — set conservatively against synthetic WAVs, may need to be relaxed against real broadcast/film audio.
   - Possibly add a `loudness_delta_prev` field for boundary cues.
4. **Coordinate a one-line fusion change with Leena** — let `audio_label == "music"` in mid-content position trigger `Advertisement` (currently fusion only acts on `silence` and `anomaly_score > 0.75`). Would help recall on no-speech ads.
5. (Optional) Add `scripts/check_updates.ps1` that fetches and reports new commits on each remote in one go.

## Audio module contract (cheat sheet)

Each `AudioWindow` we emit carries (in `model_extra`):

| Field | Range | Fusion behavior |
|---|---|---|
| `audio_label` | `"silence"`/`"speech"`/`"music"`/`"mixed"` | `silence` → `Inactivity` |
| `energy_rms` | `[0, 1]` | `< 0.02` → `Inactivity` |
| `anomaly_score` | `[0, 1]` | `> 0.75` → `Advertisement` |

Plus auxiliary: `rms_db`, `zcr`, `zcr_var`, `spectral_centroid`, `spectral_rolloff`, `spectral_flatness`.

## Speech contract (handled by `Automatic_speech_recognition/`)

`SpeechSpan(t0, t1, text)` per segment. Fusion runs keyword + brand-density matching:
- Tier 1: direct overlap with `_SPONSORSHIP_PHRASES`, `_SELF_PROMO_PHRASES`, `_OUTRO_PHRASES`, `_INTRO_PHRASES`, `_RECAP_PHRASES`.
- Tier 2: ≥ 2 hits from `_AD_BRAND_NAMES` within ±15 s → `Advertisement`.

## Signal priority (set by fusion)

`Speech > Audio > Visual` — speech overrides audio overrides visual baseline.

---

## Useful commands to remember

```powershell
# Refresh PATH from registry (fixes "ffmpeg not found" in fresh shells)
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")

# What's new on each remote?
git fetch --all --prune
git log Songmao..origin/main   --oneline
git log Songmao..origin/Leena  --oneline
git log Songmao..origin/Murali --oneline
git log Songmao..origin/Dengyi --oneline

# Rebuild the audio bundle without re-running ffmpeg
python -m audio_analyze --audio-in data/intermediate/test_001.wav --bundle-in data/output/test_001_analysis_bundle.json --bundle-out data/output/test_001_analysis_bundle.json
```

---

# Session notes — 2026-05-05

Local working notes; **not committed** (see `.gitignore`).

---

## TL;DR

- Got real videos in `videos_with_ad/` (test_001..test_005, plus a new test_010 stitched on a different schema). Ran the full visual + audio + Whisper-small + fusion pipeline end-to-end on all six.
- Audited the upstream audio + Whisper output inside ground-truth ad windows. Quantified what fusion *actually sees* vs what its detector was designed for. Documented the structural mismatch ("sponsor-read style detector vs TV-style ad data").
- Created branch `tuning_audio` (typo-renamed from `tunring_audio`) and committed two focused commits:
  1. `2ad94b7` — extend `schemas/video_info.py` to handle the new `test_010` ground-truth format (4 ads + intro/outro + new key names).
  2. `990371a` — rework the fusion audio/speech module: word-boundary brand matching, ambiguous-brand co-signal requirement, TV-ad lexicon (imperative / deal / tagline / compliance / pricing), transcript-density-drop interior feature, loudness-jump boundary feature.
- **Mean F1 0.380 → 0.490 (+29%); mean IoU 0.305 → 0.456 (+50%) across all 6 videos.**
- Filled in Whisper coverage on test_001 / test_004 / test_010 (they had been skipped). Did not change overall F1 (-0.009) but exposed two concrete remaining gaps: brand list is missing real broadcast-TV brands actually present in the data (`frank's red hot`, `bosch`, `instacart`), and a few generic lexicon phrases (`the only`, `the new`) need to be co-signal-only.
- 22/22 → **34/34 tests passing**.
- Player UI verified to construct + run cleanly with the new module (`PlayerWindow` smoke test).
- Branch is pushed to `origin/tuning_audio`.

The **next gating item** is choosing one of the four follow-ups in §"Next steps" — the highest-leverage cheap wins are (A) brand-list expansion + lexicon demotion, and (B) speech down-weighting in confident-cut windows.

---

## Where we started this session

| | Before |
|---|---|
| Branch | `main` (= `Songmao` after merges) |
| `videos_with_ad/` | empty — no real videos to test against |
| `video_info/test_010.json` | did not exist |
| `schemas/video_info.py` | `Literal["video_content","ad"]` — failed to load test_010 |
| Fusion brand matching | substring match (`if b in combined`); `apple` fired on `snapple`, `discover` on `discovered` |
| Fusion speech-text scorer | sponsor-read style only: phrases like "use code", "brought to you by", "≥2 brand hits" |
| Fusion edge scorer | `0.40·vis + 0.25·scene_cut + 0.25·anomaly_delta + 0.10·speech_trans` |
| Fusion foreignness | `W_AUDIO=0.50, W_VISUAL_SEMANTIC=0.50` only |
| Tests | 22/22 |
| F1 (mean over 6 videos) | 0.380 |
| IoU (mean over 6 videos) | 0.305 |

---

## Sequence of work performed today

1. **Smoke-tested the updated player + fusion code** on all six videos with `--skip-speech`. test_001..test_005 took ~40 s of visual analysis each; test_010 (1.06 GB / 32:40) took 401 s of visual + 3 s audio + 1 s fusion. Confirmed `PlayerWindow` constructs cleanly and `player_fusion.run_video_segmentation` loads cached `data/output/<stem>_segments.json` for every test (`[fusion] Loaded 7 segments from test_00X_segments.json`).
2. **Found that `schemas/video_info.py` couldn't load `test_010.json`.** That JSON uses a newer schema: `inserted_segments` instead of `inserted_ads`, `segment_type` / `final_video_start_seconds` instead of `ad_index` / `final_video_ad_start_seconds`, `intro` / `outro` types in `timeline_segments`, and 4 ads instead of 3. The old pydantic model raised `ValidationError` on the `intro`/`outro` literal type and silently dropped the new key names.
3. **Patched `schemas/video_info.py`** — relaxed `TimelineSegment.type` to `str`, added `validation_alias` for the new key names, added `reference_non_content_segments_player_shape()` for intro+outro+ads. The existing `reference_ad_segments_player_shape()` still returns ads-only so evaluator precision/recall isn't polluted.
4. **Ran first full evaluation.** Mean F1 = 0.388 over 5 (test_010 not yet analyzed). With test_010, **F1 = 0.380, IoU = 0.305**. test_005 was 0.000 (predicted ads at 44–138, 790–882, 1266–1358; GT at 151–196, 677–707, 1054–1084 — zero overlap).
5. **Re-ran the audio-dominant cases (test_002 / test_003 / test_005) with Whisper-small.** Used `--skip-analysis` to keep cached visual bundles. Took ~4 min each. Mean F1 only moved 0.380 → 0.380. Per-video deltas were tiny because the existing speech-text scorer wasn't capturing what was actually in the transcripts.
6. **Did a per-ad audit** of audio + Whisper output inside ground-truth ad windows for test_002 / test_003 / test_005. Documented every transcript span, every brand hit, every phrase hit, audio label distribution, mean RMS_dB, and `anomaly_score`. Saved at `data/output/_audio_speech_audit.log` and `_ad_transcript_audit.log`.

   **Key findings:**
   - 0 of 9 GT ads had any sponsorship phrase (`brought to you by`, `use code`, etc.).
   - 0 of 9 had ≥2 brand mentions.
   - 6 of 9 had **no** brand mention at all.
   - Several "brand hits" were really common English words (`discover` as verb, `secret`, `max`, `wish`, `prime`, `coke` substring of "Jacoke").
   - Transcript chars/sec inside ads was 10–74 % of the global baseline (test_002: 3.2 vs 9.5; test_003: 0.4 vs 3.9). **Strongest single signal in the data.**
   - `rms_db` stepped 6–12 dB at every real ad boundary.
   - `anomaly_score` (the existing audio interior signal) was *not* discriminative on test_002 / test_003 / test_005 — sometimes higher *outside* ads.

7. **Created branch `tunring_audio` (typo); renamed to `tuning_audio` before push.**

8. **Committed schema fix + test_010 ground truth** (`2ad94b7`).

9. **Implemented the new audio/speech module** (`990371a`):
   - `fusion/ad_signals.json`: added `tv_ad_imperative`, `tv_ad_deal`, `tv_ad_tagline`, `tv_ad_compliance`, `tv_ad_pricing` phrase categories and an `ambiguous_brands` list.
   - `fusion/fuse.py`: word-boundary regex brand matching; ambiguous-brand co-signal requirement; new `_speech_text_ad_signal` (sponsorship 0.95 / lexicon 0.55–0.85 / safe brand 0.45–0.65 / ambiguous +0.10 only as bump); new `_loudness_jump_score` (median rms_db delta, 8 dB → 1.0); new `_transcript_density_score` (chars/sec drop vs global baseline).
   - Wired loudness_jump into `_compute_edge_scores` (weight 0.20, replacing some of the unreliable anomaly-delta weight). Wired density_drop into `_compute_foreignness_scores` (weight 0.20).
   - Added 12 unit tests; total 22/22 → 34/34 passing.

10. **Re-evaluated.** Mean F1 0.380 → 0.490 (+29 %); IoU 0.305 → 0.456 (+50 %). test_002 jumped from F1 0.147 to 0.520 (the case where the audit predicted density-drop would dominate). test_010 went 0.320 → 0.488 once loudness-jump replaced part of the anomaly-delta. Side-by-side report at `data/output/_eval_summary_before_after.txt`.

11. **Filled in Whisper coverage** on test_001 / test_004 / test_010 (they had been `--skip-speech` from the first run). Took ~13 min sequential. Result: 494, 314, 119 spans respectively. Mean F1 essentially unchanged (0.490 → 0.481). Per-video deltas exposed two specific remaining issues:
    - test_004 **gained** +0.032 F1 — Whisper found "welcome to McDonald's may I take your order" inside ad #1 (safe brand hit) and "Enterprise" inside ad #3.
    - test_010 **lost** −0.039 F1 — Whisper found Frank's RedHot ad #3 ("Chingy Frank's red hot the greatest of all time") and Bosch ad #4 ("Bosch appliances"), but neither brand is in `ad_signals.json`. Meanwhile a generic lexicon phrase (`the only` from "you're not the only one this can be heavy" inside ad #1) caused a false positive that shifted the prediction off a stronger visual cut.
    - test_001 **lost** −0.047 F1 — visual + audio were already nailing the boundaries without speech, and the new density signal nudged ad #1's end cut earlier (110–228 → 110–184), losing 44 s of recall.
12. **Pushed branch.** `origin/tuning_audio` exists with both commits.

---

## Final repo state

```
tuning_audio (= origin/tuning_audio)
  990371a  Rework fusion audio/speech module for TV-ad-style data
  2ad94b7  Extend video_info schema for new test_010 (4-ad / intro+outro) format
  6a75681  higher stats                                            (= main)
```

`git log main..tuning_audio` returns the two commits above. `data/output/` is fully populated with `*_analysis_bundle.json` (with Whisper spans for all 6) and `*_segments.json`.

### Files added/changed this session

Tracked (committed):
- `schemas/video_info.py` — accept both old and new ground-truth shapes
- `video_info/test_010.json` — new ground truth (4 ads + intro/outro)
- `fusion/ad_signals.json` — TV-ad lexicon + ambiguous brand list
- `fusion/fuse.py` — word-boundary matching, density-drop, loudness-jump
- `tests/test_fusion.py` — +12 tests (now 14 fusion tests, 34 total)

Untracked (gitignored):
- `data/output/test_00*_analysis_bundle.json` — six analysis bundles, all six now contain Whisper transcripts
- `data/output/test_00*_segments.json` — six fusion outputs
- `data/output/_audio_speech_audit.log`, `_ad_transcript_audit.log`, `_speech_audit_v2.log` — per-ad audits used for tuning decisions
- `data/output/_evaluate_*.log`, `_pipeline_*.log` — run logs
- `data/output/_eval_summary_before_after.txt` — three-column comparison (v0 baseline / v1 new module / v2 with full Whisper)

### System changes on this machine

None new. All Python deps and the faster-whisper `small` model were already installed from the 2026-04-30 session.

---

## Final numbers

```
                       v0       v1       v2     v2-v1
                     baseline  new mod.  +full speech
test_001 F1          0.890    0.812    0.765   -0.047
test_002 F1          0.147    0.520    0.520    0.000
test_003 F1          0.284    0.336    0.336    0.000
test_004 F1          0.613    0.762    0.794   +0.032
test_005 F1          0.025    0.023    0.023    0.000
test_010 F1          0.320    0.488    0.449   -0.039

MEAN     F1          0.380    0.490    0.481
         IoU         0.305    0.456    0.446
```

`v0` = main baseline (mixed Whisper coverage). `v1` = new fusion module, same Whisper coverage as v0. `v2` = new fusion module + Whisper run on all 6.

Mean F1 over **v0 → v2 = +27 %**; mean IoU **+46 %**. Speech filled in correctly across all six but speech alone wasn't where the win came from — the win was the loudness-jump and density-drop features in fusion.

---

## How the new module actually works (cheat sheet)

`_speech_text_ad_signal(t0, t1, speech_spans)` → float in `[0, 1]`. Combines four kinds of evidence in a ±20 s window:

| Tier | Trigger | Score |
|---|---|---|
| 1 | Any sponsorship phrase (`brought to you by`, `use code`, ...) | 0.95 |
| 2 | ≥2 distinct TV-ad lexicon categories (imperative + compliance, etc.) | 0.85 |
| 2 | Exactly 1 TV-ad lexicon category | 0.55 |
| 3 | ≥2 safe brand hits | 0.65 |
| 3 | 1 safe brand hit | 0.45 |
| coadj | Ambiguous brand mentions (`discover`, `apple`, `max`, ...) | only +0.10 bump per hit (max 3) and only when other evidence is present; never drives the score alone |

`_loudness_jump_score(t_boundary, audio_windows, half_sec=10.0)` → `[0, 1]`. Median `rms_db` in `[t-10, t-1]` vs `[t+1, t+10]`; an 8 dB step normalizes to 1.0.

`_transcript_density_score(t0, t1, speech_spans, baseline)` → `[0, 1]`. Chars/sec inside `[t0-8, t1+8]` vs the global baseline. 50 % drop = 0.5; full silence = 1.0.

`_compute_foreignness_scores`:
```
W_AUDIO            = 0.45
W_VISUAL_SEMANTIC  = 0.40
W_DENSITY_DROP     = 0.20
W_PALETTE          = 0.00
W_NOSPEECH         = 0.00
```

`_compute_edge_scores`:
```
0.35 * vis + 0.20 * scene_cut + 0.15 * aud_delta + 0.20 * loud_jump + 0.10 * speech_transition
```

---

## Current inefficiencies / known issues

### High-impact (would move numbers)

1. **Brand list is missing the broadcast-TV brand catalog actually present in the data.** Concrete misses confirmed in transcripts: `frank's red hot`, `bosch`, `instacart`, plus a long tail of fast-food / CPG brands. Adding 30–40 entries should recover most of the F1 lost on test_010.
2. **A few lexicon phrases are too generic to single-trigger.** `"the only"`, `"the new"`, `"trusted by"`, possibly `"introducing the"`. These collide with show language and false-fired on test_010 ad #1. Should be moved to a co-signal-only tier alongside ambiguous brands.
3. **Speech evidence over-rotates boundaries when visual+audio already agree.** test_001 ad #1 (118 s) had visual+loudness pinning the cut correctly, but the new density signal (1.0 inside the wordless ad) shifted the end cut earlier and lost 44 s of recall. Should down-weight speech contributions when palette_delta + loudness_jump both fire above threshold at a candidate boundary.
4. **Fusion is hardcoded to `NUM_ADS = 3`.** `test_010` has 4 ads and structurally caps recall at ~0.75. If the held-out test set varies in ad count this needs to become threshold-based (or N-best with a cutoff).
5. **`audio.analyze`'s `_anomaly_scores` is not discriminative on real broadcast audio.** Per-video MFCC normalization washes out the signal — `anomaly_score` is sometimes *higher* outside ads on test_002 / test_005 / test_010. Either recalibrate the percentile mapping (currently 50 → 0, 95 → 1) against the real videos or replace the rule-based classifier with a learned one (YAMNet).
6. **Substring brand-list collisions still exist for some single-token entries.** Word-boundary regexes fix the `apple`/`snapple` class of bug, but `target`, `apple`, `discover` etc. are real English words — they're now in `ambiguous_brands` but anyone editing the brand list should keep adding to that list, not just `brands`.

### Medium

7. **`scripts/run_pipeline.py` step counter is wrong with `--skip-analysis`.** Prints `[1/2] Skip visual analysis ...` then `[3/2] Fusion`. Cosmetic but ugly.
8. **`SESSION_NOTES.md` (this file) is gitignored.** Means each contributor only sees their own. If we want shared knowledge transfer we should land a `docs/` or `notes/` directory that *is* committed for the agreed-on facts.
9. **`pre-existing` linter warnings in `fusion/fuse.py`.** Sonarqube flags four pre-existing issues we haven't touched: `np.random.rand` (line 497), unused params `duration` (849), `min_segment_seconds` (953), `enforce_three_ads` (954).
10. **`test_010.json` claims `video_filename: test_007.mp4`** but the actual file in `videos_with_ad/` is `test_010.mp4`. Probably a bookkeeping artifact from whoever stitched it. Worth confirming with whoever produced it before we trust the GT times.
11. **Whisper-small VAD off by default.** test_001 took 5 min for 24 min of video. With `--vad` it would skip silence and likely halve runtime. Audit didn't surface accuracy issues, but enabling VAD is a free speedup.

### Low / future

12. **No held-out test set yet.** Everything we've measured is on the same 6 we tuned on. Once more videos arrive we should split.
13. **No CI on this branch.** `pytest` runs locally but no GitHub Actions checks.

---

## Next steps (in priority order)

1. **(A) Brand-list expansion + lexicon demotion** — 30 min, no new deps, expected ≈+0.05 F1 on test_010. Add `frank's red hot`, `bosch`, `instacart`, `dewalt`, `makita`, `tide`, `kelloggs` (already there?), `kraft` (already), plus the long tail. Move `the only`, `the new`, `trusted by`, `introducing the` into ambiguous-tier (require co-signal).
2. **(B) Down-weight speech in confident-cut windows** — 1 hour, no new deps, expected ≈+0.05 IoU on test_001. When `palette_delta > 0.5` AND `loudness_jump > 0.5` both fire at a candidate boundary, scale density and text contributions to 0.3×. Goal: stop speech from rotating already-correct cuts.
3. **(C) YAMNet for audio events** — half day, +17 MB dep. The only real path to fix `test_005` (currently F1 0.02). Categories `Television advertisement`, `Theme music`, `Jingle (music)` would be near-perfect ad indicators on the current data.
4. **(D) Sentence-embedding discontinuity** — half day, +90 MB dep (`all-MiniLM-L6-v2`). Cosine-sim drops between consecutive Whisper spans at ad boundaries. Content-agnostic — handles ads with no brand and no audio cliff. Best safety net for "Frank's RedHot"-style cases without expanding the brand list.
5. **Make `NUM_ADS` dynamic** — replace the hardcoded 3 with a score-threshold-based selection. Without this, recall on `test_010` is structurally capped.
6. **Fix `scripts/run_pipeline.py` step counter** — quick polish.
7. **Promote shared notes out of `SESSION_NOTES.md`** to a committed `docs/notes/` so the team isn't relying on one person's local file.
8. **Add CI** — minimum: `pytest -q` on push.

---

## Useful commands to remember

```powershell
# Refresh PATH from registry (fixes "ffmpeg not found" in fresh shells)
$env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")
$env:PYTHONPATH = "."

# Full pipeline on one video (visual + audio + speech + fusion)
python scripts/run_pipeline.py videos_with_ad/test_001.mp4

# Re-fuse without re-extracting visual or audio (fastest iteration loop on fusion)
python -c "
from pathlib import Path
from fusion.fuse import fuse_bundle_to_segments, load_bundle, write_segments_json
out_dir = Path('data/output')
for bundle_path in sorted(out_dir.glob('test_*_analysis_bundle.json')):
    name = bundle_path.stem.replace('_analysis_bundle','')
    bundle = load_bundle(bundle_path)
    segments = fuse_bundle_to_segments(bundle)
    write_segments_json(segments, out_dir / f'{name}_segments.json')
    print(name, len(segments))"

# Evaluate against ground truth
python scripts/evaluate.py
python scripts/evaluate.py --test test_001     # single video

# Audit speech inside GT ad windows for one or more videos
# (the audit scripts under data/output/_audio_speech_audit.log etc. were one-shot
#  scripts; if needed again, reconstruct from the patterns at the start of
#  fusion/fuse.py and schemas/video_info.py)

# Tests
python -m pytest -q

# Player UI
python -m player.player

# Whisper coverage check
python -c "
from pathlib import Path
from schemas.modality import AnalysisBundle
for n in ['test_001','test_002','test_003','test_004','test_005','test_010']:
    b = AnalysisBundle.model_validate_json(Path(f'data/output/{n}_analysis_bundle.json').read_text(encoding='utf-8'))
    chars = sum(len(s.text or '') for s in b.speech_spans)
    print(f'{n}: {len(b.speech_spans):>4} spans, {chars:>6} chars')"
```

---

# Session notes — 2026-05-06

Local working notes; **not committed** (see `.gitignore`).

---

## TL;DR

- **Removed the hardcoded `NUM_ADS = 3` cap** in `fusion/fuse.py`. The 3-stage unrolled DP is now a generic `_find_best_k_ads(edge, foreign, windows, k)` over arbitrary K, plus a marginal-gain auto-K selector with a floor of 3.
- **Default behaviour changed**: `fuse_bundle_to_segments(...)` and `python -m fusion ...` now run **auto-K** (`num_ads=None`). Pass `--num-ads 3` (or `num_ads=3`) to restore the old fixed-K behaviour.
- **Mean F1 0.481 → 0.510 across all 6 videos (+0.029, +6%).** Mean IoU 0.446 → 0.430 (slight drop, expected since adding ads widens the predicted region).
- **`test_010` specifically: F1 0.449 → 0.568 (+27%)** — auto-K finds the 4th ad that the K=3 cap was missing. Recall jumps 0.511 → 0.702. This is the answer to the "what do we get on test_010 if we lift the limit" question.
- Bonus: `test_003` lifts F1 0.336 → 0.390 (+16%). Auto picks K=5 on this one — 5 turns out to genuinely beat 3 even though there are only 3 reference ads, because the GT is bookkeeping-loose on it.
- No regressions on the other 4 cases — auto-K stays at K=3 on them.
- Added 4 new fusion tests (K=1, K=4, auto-K-on-4-ad-bundle, auto-K-stays-at-3-on-3-ad-bundle). **34/34 → 38/38 tests passing.**
- Branch is still `tuning_audio`; no commits yet — changes are sitting in the working tree.

---

## Why naive auto-K (no floor) didn't work, and what does

I first ran a `marginal_ratio` sweep across all 6 videos. The fundamental problem on real broadcast video is that the per-K interval scores are **very compressed**:

```
test         g1     g2/g1   g3/g1   g4/g1   g5/g1   g6/g1
test_001   5.652   0.806   0.762   0.646   0.530   0.480
test_002   5.095   0.915   0.886   0.877   0.806   0.778
test_003   5.224   0.977   0.973   0.939   0.910   0.881
test_004   5.516   0.963   0.936   0.747   0.659   0.657
test_005   5.203   0.928   0.911   0.898   0.878   0.877
test_010   5.333   0.999   0.946   0.924   0.897   0.891
```

So even the 6th-best "ad" is scored at 65-90 % of the best one. The DP score function isn't discriminative enough to find a clean knee. Any ratio threshold that's loose enough to catch test_010's true 4th ad (need r ≤ 0.92, since `g_4/g_1 = 0.924`) ends up picking K = 6 on test_002 / 003 / 005 (their `g_6/g_1 ≥ 0.85`).

The pragmatic fix: **floor the auto-K at 3** and let the rule only decide whether to add a 4th, 5th, 6th. That works because:
- The score-curve shape *can* discriminate "is there a 4th true ad" reasonably well (test_010's `g_4/g_1 = 0.924` is at the ratio threshold; test_001's drops to `0.646`).
- Below K=3 the rule under-detects (would pick K=1 or K=2 even on cases that clearly have 3 ads), so we just refuse to go below 3.

Final defaults in `fuse.py`:
```python
NUM_ADS              = None   # None -> auto. Pass 3 (or any int) to force.
MIN_NUM_ADS          = 3      # auto-K floor
MAX_NUM_ADS          = 6
MIN_MARGINAL_RATIO   = 0.90
```

Per-test K selected by auto with these defaults: 3, 3, **5**, 3, 3, **4** for test_001..005, test_010.

---

## Final numbers

```
                       v0       v1       v2       v3
                     baseline  new mod  +full speech  + auto-K
test_001 F1          0.890    0.812    0.765    0.765
test_002 F1          0.147    0.520    0.520    0.520
test_003 F1          0.284    0.336    0.336    0.390   (+0.054)
test_004 F1          0.613    0.762    0.794    0.794
test_005 F1          0.025    0.023    0.023    0.023
test_010 F1          0.320    0.488    0.449    0.568   (+0.119)

MEAN     F1          0.380    0.490    0.481    0.510
         IoU         0.305    0.456    0.446    0.430
```

Cumulative `v0 → v3`: **F1 +34 %, IoU +41 %**. The auto-K change alone delivered +6 % F1 over v2.

---

## Files changed this session (working tree, not yet committed)

- `fusion/fuse.py` — generic `_find_best_k_ads(k)` DP, `_select_num_ads_auto(...)` selector, new `num_ads` / `max_num_ads` / `min_num_ads` / `min_marginal_ratio` parameters on `fuse_bundle_to_segments`. Default flipped from `NUM_ADS=3` to `NUM_ADS=None` (auto).
- `fusion/__main__.py` — `--num-ads` and `--max-num-ads` CLI flags; default `--num-ads auto`.
- `tests/test_fusion.py` — 4 new tests (`test_fusion_recovers_four_ads_when_num_ads_is_4`, `test_fusion_recovers_one_ad_when_num_ads_is_1`, `test_fusion_auto_k_picks_four_on_four_ad_bundle`, `test_fusion_auto_k_stays_at_three_on_three_ad_bundle`).
- `scripts/sweep_num_ads.py` (new, gitignored helper) — re-fuses all 6 cached bundles under K=3 / K=4 / auto and runs the evaluator on each.
- `scripts/sweep_auto_ratio.py` (new, gitignored helper) — sweeps `min_marginal_ratio ∈ {0.55, 0.70, 0.80, 0.85, 0.90, 0.95}` on all 6 videos, dumps the per-K F1/IoU table that drove the threshold choice.
- All `data/output/test_00*_segments.json` re-generated with auto-K. The player UI and the evaluator both pick those up automatically.

---

## How to use the new knob

```powershell
# Auto-K (default) — what the evaluator just ran
python -m fusion --bundle data/output/test_010_analysis_bundle.json

# Force K=4 (matches test_010 ground truth)
python -m fusion --bundle data/output/test_010_analysis_bundle.json --num-ads 4

# Restore old fixed-3 behaviour
python -m fusion --bundle data/output/test_010_analysis_bundle.json --num-ads 3

# Programmatic
from fusion.fuse import fuse_bundle_to_segments, load_bundle
bundle = load_bundle(Path("data/output/test_010_analysis_bundle.json"))
segments = fuse_bundle_to_segments(bundle, num_ads=None)              # auto
segments = fuse_bundle_to_segments(bundle, num_ads=4)                 # forced
segments = fuse_bundle_to_segments(bundle, num_ads=None,
                                   min_marginal_ratio=0.85,
                                   max_num_ads=8)                     # tuned
```

---

## Next steps (revised)

These move up to the top now that auto-K is in place:

1. **(A) Brand-list expansion + lexicon demotion** — **DONE later this session, smaller-than-projected win.** See "Brand-list expansion" subsection below. Mean F1 0.510 → 0.511, test_010 F1 0.568 → 0.575. The right brands now get matched, but a structural false positive on test_010 at 798-858 caps the per-K=4 result; auto-K compensates by picking K=5.
2. **Make the DP score more discriminative.** The compressed `g_k / g_1` ratios (0.85+ even at K=6 on real video) are why auto-K needed an artificial floor. Adding an absolute interior-score threshold (e.g. require `mean(foreignness) > 0.4` for any interval to count as an ad) would let auto-K work without the floor and give a better answer on `test_005` too (currently a noise floor at F1 0.023 is hiding real ad regions).
3. **(B) Down-weight speech in confident-cut windows** — same as before, ~1 hr, expected ≈+0.05 IoU on test_001.
4. **(C) YAMNet for audio events** — only realistic path to fix `test_005`. Half day, +17 MB dep. Now the highest-leverage remaining item — see roadmap canvas (`canvases/score-improvement-roadmap.canvas.tsx`) for the full headroom analysis.
5. **(D) Sentence-embedding discontinuity** — content-agnostic safety net, half day, +90 MB dep. Helps test_002 (podcast-style sponsor reads) and test_003 (long wordless music ad).
6. Old item 5 ("make `NUM_ADS` dynamic") — **DONE this session.**

### Brand-list expansion — what landed and what didn't

Added to `fusion/ad_signals.json`:
- New `home_household_tools` category (32 entries): `bosch`, `dewalt`, `makita`, `ryobi`, `tide`, `clorox`, `lysol`, `weber grills`, etc.
- Long-tail food/beverage additions: `frank's red hot` (and 3 spelling variants), `tabasco`, `sriracha`, `cheez-it`, etc.
- `instacart`, `doordash`, `uber eats`, `grubhub`, `postmates` in `subscription_lifestyle`.
- Removed `"the only"` from `tv_ad_tagline` (confirmed false positive on test_010 ad #1 — it fired on `"you're not the only one this can be heavy"` from the show's dialogue).

Kept `"the new"`, `"trusted by"`, `"introducing the"` in their existing categories — no audit evidence of false-firing on the current videos.

Numerical impact:
- test_010 F1: 0.568 → 0.575 (+0.007). Recall 0.702 → 0.752 (caught Bosch ad #4 + better Frank's RedHot localisation).
- All other videos: unchanged.
- Mean F1: 0.510 → 0.511. Mean IoU: 0.430 → 0.419 (auto-K shifted from K=4 to K=5 on test_010, predictions widen).

Why the win was smaller than the projected +0.04 F1:

1. The audit had already shown only test_010 has brand-list-fixable issues. test_001/test_004 are wordless ads. test_002 is host-read sponsorships ("Hey, it's Kay Davis…") with no broadcast brands. test_003 is wordless ("Oh, Oh, Oh"). test_005 only has the ambiguous `coke`/`discover` mentions (still ambiguous). So the brand list helped *only* the one case it was theoretically supposed to.
2. On test_010 specifically, the new brand evidence successfully pulls auto-K toward catching 4 of 4 GT ads, but a structural false positive at 798-858s (between GT#2 and GT#3) still gets ranked alongside the real ads. Auto-K compensates by picking K=5 (4 real + 1 FP) instead of K=4 (which now drops GT#2 in favor of a too-narrow GT#4 match — lower total score). F1 of 0.575 reflects 4-of-4 GT ad recall but at +20% more predicted ad time than the K=4-correct world would give.

Verdict: leave the brand additions in (they're correct and fix real Whisper-visible misses), but the next leg of improvement on test_010 needs a structural fix to the foreignness scoring at the 798-858 false positive — likely YAMNet audio-event detection, since visual+audio in that band are simply ambiguous.

---

## Useful commands

```powershell
# Sweep K for a single bundle (or all 6) — reuses cached bundles
python scripts/sweep_num_ads.py             # K=3 / K=4 / auto across all 6
python scripts/sweep_auto_ratio.py          # marginal-ratio threshold sweep

# Quick re-fuse with a chosen K (no re-extraction)
python -c "
from pathlib import Path
from fusion.fuse import fuse_bundle_to_segments, load_bundle, write_segments_json
bundle = load_bundle(Path('data/output/test_010_analysis_bundle.json'))
segments = fuse_bundle_to_segments(bundle, num_ads=4)   # or num_ads=None for auto
write_segments_json(segments, Path('data/output/test_010_segments.json'))"

# Evaluate the new state
python scripts/evaluate.py
python scripts/evaluate.py --test test_010
```

---

# Session notes — 2026-05-06 (afternoon)

Local working notes; **not committed** (see `.gitignore`).

---

## TL;DR

- **Two new ad-interior features in fusion** beyond the existing audio /
  visual / density / loudness signals:
  1. **Spectral flatness deviation** — per-video median-subtracted
     `spectral_flatness` (already extracted by `audio/analyze.py`,
     just not previously read by fusion). Free win, no new deps.
  2. **YAMNet music & non-speech probabilities** — Google's 521-class
     audio-event classifier from AudioSet, ONNX export from
     `zeropointnine/yamnet-onnx` (16 MB). Runs in ~2 s per 30-min
     video on CPU via the `onnxruntime` we already have installed
     for faster-whisper's Silero VAD.
- **Mean F1 0.511 → 0.552 across the 6 videos** (+0.041, +8.0 %).
- **`test_005` F1 0.023 → 0.508** — the case the prior session notes
  flagged as needing YAMNet specifically. Single biggest per-video
  jump in the project history.
- **`test_010` F1 0.575 → 0.429 (-0.146)** — auto-K mis-selection on
  the new stronger foreignness signal. K=3 forced gives 0.449 (back
  to baseline). Tuning would recover this but at the cost of
  test_003/test_005 wins; current defaults pick the better
  cross-the-board trade-off.
- 38/38 → **52/52 tests passing** (added 14 helper tests).
- Two commits on `tuning_audio`:
  1. `7dc3994` — spectral-flatness feature + cap MAX_NUM_ADS at 5.
  2. `396f121` — YAMNet music/speech features + per-video baseline
     subtraction + W_AUDIO/W_VISUAL_SEMANTIC rebalance.

---

## Numbers (per-video, auto-K, MAX=5)

```
                  v3 (auto)  + spectral   + YAMNet     Δ vs v3
                  (0.510)    (0.541)      (0.552)      (+0.042)
test_001 F1       0.765      0.765        0.765         0.000
test_002 F1       0.520      0.520        0.408        -0.112    auto-K K=4
test_003 F1       0.390      0.325        0.411        +0.021
test_004 F1       0.794      0.794        0.794         0.000
test_005 F1       0.023      0.265        0.508        +0.485    !
test_010 F1       0.575      0.575        0.429        -0.146    auto-K K=5 worse intervals

MEAN     F1       0.510      0.541        0.552
         IoU      0.430      0.439        0.424

K=3 forced mean   0.470      0.470        0.541   ← cleaner picture
                                                    of YAMNet quality
```

**Cumulative v0 → here: F1 0.380 → 0.552, +45 %; IoU 0.305 → 0.424, +39 %.**

---

## Why YAMNet helps where the existing audio signal didn't

The MFCC-based `anomaly_score` (computed in `audio/analyze.py`) ranks
windows by distance from the per-video MFCC median. On real broadcast
TV that score is essentially uninformative — sometimes *higher*
outside ads — because the show's own audio mixing dominates the MFCC
distribution. YAMNet learned semantic categories (Music, Speech,
Theme music, Jingle) on AudioSet's massive collection, so it
generalises to the broadcast-vs-show distinction directly.

Inside-vs-outside-ad gap on the cached 6-video set
(`scripts/yamnet_diagnostic.py`):

```
test         music gap   speech gap
test_001     +0.747     -0.915    very strong
test_002     +0.446     -0.558    strong
test_003     -0.025     -0.231    weak music, decent speech
test_004     +0.584     -0.764    very strong
test_005     +0.284     -0.258    moderate
test_010     +0.142     +0.065    weak (drama with continuous music)
```

The +0.14..+0.75 music gap dwarfs the spectral-flatness gap of
+0.05..+0.25 from the previous commit. Speech inverse adds
complementary evidence on dialog-dense shows where ads go wordless.

Per-video baseline subtraction is essential: test_010 sits at music
~0.30 throughout (it's a film with continuous underscore), so the
raw probability mis-fires; the deviation captures the *additional*
musical boost ad blocks impose on top of the show. Same trick the
spectral-flatness signal already uses; same `np.median` baseline.

---

## Why the auto-K regressions on test_002 and test_010

The DP scores K candidate ads jointly. Adding a powerful interior
signal (YAMNet music) compresses the marginal-gain ratios further:

```
                           g_2/g_1  g_3/g_1  g_4/g_1  g_5/g_1
test_002 spectral-only      0.915    0.886    0.877    0.806
test_002 +YAMNet            0.994    0.989    0.912    0.832
```

So the K=3→K=4 transition on test_002 goes from "0.886 → 0.877" (no
clear plateau) to "0.989 → 0.912" — auto-K with `MIN_MARGINAL_RATIO=0.90`
now sees K=4 as still high-quality and adds an extra interval that
turns out to fragment GT ad #2.

Tightening to `MIN_MARGINAL_RATIO=0.95` recovers test_002 (lands on
K=3, F1=0.520) but craters test_003 (lands on K=4 instead of optimal
K=5; K=4 has F1=0.059 because its picks fragment the wordless
montage). Net: -0.04 mean F1. Worse than current.

The cleanest fix would be a per-video score-threshold rule rather
than a marginal-ratio rule, but that's "Make the DP score more
discriminative" which is its own project.

---

## How to use

```powershell
# Get the YAMNet model (one-time, ~16 MB)
python -c "from huggingface_hub import hf_hub_download; import shutil, os
os.makedirs('audio/models', exist_ok=True)
for fn in ('yamnet.onnx', 'yamnet_class_map.csv'):
    src = hf_hub_download(repo_id='zeropointnine/yamnet-onnx', filename=fn)
    shutil.copyfile(src, os.path.join('audio/models', fn))"

# Backfill YAMNet on existing analysis bundles (no re-running visual /
# audio / Whisper)
$env:PYTHONPATH = "."
python scripts/add_yamnet_to_bundles.py            # all 6
python scripts/add_yamnet_to_bundles.py --tests test_005

# Or run the full pipeline — audio/analyze.py picks up YAMNet
# automatically when audio/models/yamnet.onnx is present, no flag needed
python scripts/run_pipeline.py videos_with_ad/test_005.mp4

# Sanity check the inside-vs-outside-ad gap
python scripts/yamnet_diagnostic.py
```

---

## Files added/changed this session

Tracked (committed in `7dc3994` + `396f121`):

- `fusion/fuse.py` — spectral_flatness + YAMNet helpers, weight
  rebalance, `MAX_NUM_ADS = 5`.
- `audio/analyze.py` — optional YAMNet pass on the same 16 kHz
  waveform, gated on `audio/models/yamnet.onnx` existing.
- `audio/yamnet_features.py` (new) — singleton-cached ONNXRuntime
  session, raw-waveform inference, per-window aggregation of
  Music / Background music / Theme music / Jingle / Soundtrack /
  Speech to four scalar extras.
- `scripts/add_yamnet_to_bundles.py` (new) — one-shot script to
  graft YAMNet scores onto cached bundles in place.
- `scripts/yamnet_diagnostic.py` (new) — inside-vs-outside-ad gap
  table.
- `scripts/sweep_num_ads.py` — drop hardcoded `max_num_ads=6` so it
  uses the module default (`MAX_NUM_ADS=5`).
- `tests/test_fusion.py` — +14 tests (6 spectral-helper +
  8 YAMNet-helper).
- `SETUP.md` — section 3b: YAMNet model download recipe.
- `.gitignore` — exempt `/audio/models/`.

Untracked (gitignored):

- `audio/models/yamnet.onnx`, `audio/models/yamnet_class_map.csv` — the
  16 MB model + 14 KB class map, downloaded from HuggingFace.
- `data/output/test_*_analysis_bundle.json` — re-stamped with
  `yamnet_*` extras on every AudioWindow.
- `data/output/test_*_segments.json` — re-fused with the new signals.

---

## Boundary extension along foreignness (DONE — 2026-05-06)

**Problem.** The DP scoring `2.5 * (edge[s] + edge[e]) + interior_mean(s, e)`
is *scale-invariant in duration* — a 20 s interval with mean=0.88 beats a
118 s interval with mean=0.70 even when the longer one is the actual ad.
This was costing real-ad recall on every video where the GT ad was
longer than ~30 s. Boundary diagnostic on test_001 K=3:

```
GT ad #1 = 106-224 s  (118 s long)
DP pick   = 110-184 s  ( 74 s, interior_mean = 0.885)
```

The DP locked onto the high-density first ~70 s of ad #1 and stopped at
edge_e = 0.66 because extending to 224 dropped the interior mean.

**Fix.** After the DP returns `(s, e)`, walk each boundary outward up to
`EXTEND_SEARCH_SEC=30 s` while the *raw* per-window foreignness stays
above `EXTEND_KEEP_RATIO=0.70` × `interior_mean(s, e)`. This is a
post-processing step that uses a different optimisation criterion
(local foreignness threshold rather than per-interval mean), so it
naturally captures the rest of a long ad when the DP under-extends.
Adjacent extended intervals are then pulled inward to preserve
`GAP_MIN_SEC` so two extensions can't merge.

| Video    | F1 before | F1 after | Δ       |
|----------|-----------|----------|---------|
| test_001 |    0.765  |   0.935  | +0.170  |
| test_002 |    0.598  |   0.728  | +0.130  |
| test_003 |    0.411  |   0.421  | +0.010  |
| test_004 |    0.794  |   0.949  | +0.155  |
| test_005 |    0.508  |   0.426  | -0.082  |
| test_010 |    0.429  |   0.431  | +0.002  |
| **mean** |  **0.584**| **0.648**|**+0.064**|

The kw setting was tuned with `scripts/sweep_extension.py`; 30 s/0.70
was the best of {15, 20, 25, 30, 40} × {0.50, 0.60, 0.70, 0.80}.

test_005 regressed because it picks K=5, three of which are FPs that
extend into surrounding content; lower keep_ratio extends them
further. K=4 oracle on test_005 is now 0.476 — the loss vs the
previous baseline (K=5, 0.508) shows up because boundary-extending
the FPs hurts precision more than extending the 2 TPs helps recall.
Tightening keep_ratio further (0.80) recovers test_005 partially but
costs more on test_001 / test_004 — net negative.

Mean F1 now **0.648** (raw baseline 0.380 → +0.268, +70% relative).

---

## Auto-K interior-mean floor (DONE — 2026-05-06)

**Problem.** YAMNet compressed the marginal-ratio series to the
point that auto-K added a 4th ad on test_002 (no real ad #4 exists)
and auto-K failed to *reduce below* MIN_NUM_ADS=3 even though K=2
was clearly the right answer.

**Fix.** Added a second auto-K rule alongside the marginal ratio:
the *minimum* normalised foreignness mean across the K chosen
intervals must stay >= `MIN_INTERIOR_MEAN_FLOOR=0.40`. The
diagnostic at `scripts/interior_diagnostic.py` shows this transition
is clean across every cached video — fakes drop from ~0.6-0.9 to
~0.1-0.3:

```
test_002 K=2 worst=0.881   K=3 worst=0.314   ← collapses, accept K=2
test_004 K=3 worst=0.630   K=4 worst=0.111   ← collapses, accept K=3
test_001 K=4 worst=0.773   K=5 worst=0.152   ← collapses, accept K=4
```

The MIN_NUM_ADS=3 floor was downgraded from a hard clamp to a
*soft* floor: only raise to it when no interior-collapse was
observed below 3. On test_002 (which has only 2 strong ads), auto-K
now correctly returns K=2.

| Video    | F1 before | F1 after | Δ     | Auto-K  |
|----------|-----------|----------|-------|---------|
| test_001 |    0.765  |   0.765  |   0   | K=3     |
| test_002 |    0.408  |   0.598  | +0.190| K=2     |
| test_003 |    0.411  |   0.411  |   0   | K=5     |
| test_004 |    0.794  |   0.794  |   0   | K=3     |
| test_005 |    0.508  |   0.508  |   0   | K=5     |
| test_010 |    0.429  |   0.429  |   0   | K=5     |
| **mean** |  **0.552**| **0.584**|**+0.032**| —    |

Mean F1 now 0.584 (was 0.552, raw baseline 0.380 — a **+54%** total
improvement from the start of session).

The two K=5 videos (test_005, test_010) are still over-detecting:
their 5th interval has interior_mean ~0.59 / ~0.54 (above the
floor), so the rule can't tell it's spurious.  Fundamental signal-
quality limit; needs a different feature, not a different rule.

---

## Next leverage points (in order)

1. **test_003 precision is the biggest remaining headroom.**
   F1=0.411 at K=5 with P=0.326, R=0.554. Recall is OK but only
   ~3 of the 5 picks are real ads. test_003's content is a
   wordless / minimal-dialogue montage, so brand-name and lexicon
   signals are useless and YAMNet music doesn't help (the show
   itself is music-heavy). Most likely lift: a *boundary-based*
   discriminator that fires only on broadcast hard cuts —
   palette_delta + loudness_jump at the same window. Would let
   the DP edge term separate true cut-to-ad transitions from
   internal music-section transitions.
2. **Down-weight foreignness signals at confident cuts** (the old
   "speech down-weight" idea from the previous session, generalised).
   When palette_delta + loudness_jump both fire at a candidate
   boundary, the cut is already certain — interior signals shouldn't
   re-rotate it. Would help test_010's interval-shift regression.
3. **Sentence-embedding discontinuity** (`all-MiniLM-L6-v2`,
   ~90 MB). Cosine drop between consecutive Whisper spans at ad
   boundaries. Content-agnostic — covers ads with no brand and no
   audio cliff. Most likely lift on test_002 (sponsor reads with no
   broadcast brand).
4. **Train a tiny model on the existing features.** With YAMNet now
   in the bundle, every window has ~10 informative scalars
   (anomaly, energy, rms_db, zcr, zcr_var, centroid, rolloff,
   flatness, music, speech). A 1-hidden-layer MLP or an
   xgboost over those should outperform the hand-tuned weighted
   sum, especially given that the right per-video relative weights
   differ across videos.
5. **Add the `--skip-yamnet` flag** to `audio_analyze`/`run_pipeline`
   so users on minimal-deps installs don't surprise-load the model.
   (Currently silently skipped when `audio/models/yamnet.onnx`
   isn't present, but explicit-is-better-than-implicit.)
6. **Promote shared notes to a committed `docs/` tree** — same
   open item as last session.
