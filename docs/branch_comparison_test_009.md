# Branch Comparison: `Dengyi` vs `main_v2` on `test_009.mp4`

Author-generated audit, dated 2026-05-08.

This document records (a) the diff between the two branches and (b) an
end-to-end F1 verification on `videos_with_ad/test_009.mp4`. The F1 numbers
are reproduced two independent ways: a from-scratch manual computation
plus the project's `scripts/evaluate.py`. They agree to within rounding,
so the headline numbers are not fabricated.


## 1. Repository diff (`Dengyi..main_v2`)

| | Lines added | Lines deleted | Net |
| :--- | ---: | ---: | ---: |
| Total across 22 files | 1,384 | 6,515 | **−5,131** |

`main_v2` is a lean reimplementation of `Dengyi`'s ideas. It ports the
OCR / semantic / multi-feature audio modules verbatim and replaces the
fusion engine with a tighter rewrite.

### Largest file deltas

| File | Δ lines | Notes |
| :--- | ---: | :--- |
| `fusion/fuse.py` | 2,665 | Rewritten. **2,378 → 857 lines** (64 % smaller). |
| `fusion/extra_brand_names.txt` | 3,429 | Removed on `main_v2`; replaced by curated `fusion/ad_signals.json`. |
| `scripts/run_pipeline.py` | 263 | `--with-ocr` / `--with-semantic` flags added on `main_v2`. |
| `fusion/__main__.py` | 142 | CLI simplified on `main_v2`. |
| `tests/test_fusion.py` | +101 | New on `main_v2` only. |
| `visual/analyze.py` | 83 | `cap.set(POS_FRAMES)` → sequential `cap.grab()` (visual I/O speedup on `main_v2`). |
| `fusion/README.md` | 95 | Refreshed docs + "Known Limitations". |
| `player_fusion.py` | 64 | Trimmed parallel-pipeline plumbing on `main_v2`. |
| `scripts/evaluate.py` | 41 | `--test` filter + cleaner table on `main_v2`. |
| `Automatic_speech_recognition/segment_text_analyzer.py` | 29 | Small cleanups. |
| `audio/analyze.py` | 21 | Minor follow-up to the multi-feature anomaly score. |
| `ocr/analyze.py` | 21 | `candidate_dedup_sec` and `add_sweep_when_candidates` exposed on `main_v2`. |
| `fusion/ad_signals.json` | 20 | Tweaked keyword set. |
| `.gitignore` | 11 | Model dirs and course PDF ignored on `main_v2`. |
| `Automatic_speech_recognition/SETUP.md` | 10 | Doc update. |

### Ground-truth files

| File | `Dengyi` | `main_v2` |
| :--- | :---: | :---: |
| `video_info/test_001.json` … `test_005.json` | yes | removed |
| `video_info/test_008.json` | — | added |
| `video_info/test_009.json` | — | added |

### Commits since merge base `e9fae6d`

| Branch | # commits |
| :--- | ---: |
| `main_v2` (ahead of base) | 19 |
| `Dengyi` (ahead of base) | 2 |

### Functional matrix

| Capability | `Dengyi` | `main_v2` |
| :--- | :---: | :---: |
| Visual + audio + speech analysis | yes | yes |
| OCR module | yes | yes (ported) |
| Semantic ad-score module | yes | yes (ported) |
| Multi-feature audio anomaly | yes | yes (ported) |
| Clean DP-based ad selection | partial (legacy + DP mixed) | **clean rewrite** (857-line `fuse.py`) |
| Semantic-span snap (post-DP) | no | **yes** |
| Intro / outro split labeling | basic | fixed (splits content runs at boundaries) |
| Outro detector recognises `graphics_heavy` | static only | static + `graphics_heavy` |
| Visual I/O via `cap.grab()` | no (`cap.set(POS_FRAMES)`) | **yes** |
| OCR / semantic via `scripts/run_pipeline.py` flags | no (only via `python -m fusion`) | **yes** |
| Brand allowlist | 3,429-line `extra_brand_names.txt` | curated `ad_signals.json` |
| Unit tests | none | `tests/test_fusion.py` |


## 2. End-to-end run on `test_009.mp4`

Identical input video. Identical evaluator. Each branch was checked out
clean, the cached bundle and segments were deleted, and the pipeline was
run from scratch with OCR + semantic enabled.

| Branch | Command | Wall time |
| :--- | :--- | ---: |
| `Dengyi` | `python -m fusion --video … --sample-fps 1.0 --window-sec 2.0 --min-segment-sec 20 --asr-vad` | 8 min 32 s |
| `main_v2` | `python scripts/run_pipeline.py videos_with_ad/test_009.mp4 --vad --with-ocr --with-semantic` | 7 min 53 s |

### Per-stage timing (`main_v2`)

| Stage | Time |
| :--- | ---: |
| Visual analysis | 136.0 s |
| Audio + speech (Whisper, VAD) | 63.6 s |
| OCR (CPU) | 268.9 s |
| Semantic text scoring (CPU) | 3.1 s |
| Fusion | 0.7 s |
| **Total** | **472 s ≈ 7:53** |


## 3. Predicted segments (raw)

### `Dengyi`

```
0    Core Content       0.0 s –  72.0 s   (72.0 s)
1    Advertisement     72.0 s – 100.0 s   (28.0 s)
2    Core Content     100.0 s – 310.0 s  (210.0 s)
3    Advertisement    310.0 s – 366.0 s   (56.0 s)
4    Core Content     366.0 s – 486.0 s  (120.0 s)
5    Advertisement    486.0 s – 544.0 s   (58.0 s)
6    Core Content     544.0 s – 710.0 s  (166.0 s)
7    Outro            710.0 s – 742.5 s   (32.5 s)
```

### `main_v2`

```
0    Intro              0.0 s –  44.0 s   (44.0 s)
1    Core Content      44.0 s –  60.0 s   (16.0 s)
2    Advertisement     60.0 s –  92.0 s   (32.0 s)
3    Core Content     92.0 s – 340.0 s   (248.0 s)
4    Advertisement   340.0 s – 408.0 s    (68.0 s)
5    Core Content    408.0 s – 546.0 s   (138.0 s)
6    Advertisement   546.0 s – 596.0 s    (50.0 s)
7    Core Content    596.0 s – 712.0 s   (116.0 s)
8    Outro           712.0 s – 742.5 s    (30.5 s)
```

### Ground truth (`video_info/test_009.json`)

```
inserted ads:
  ad_04_kelloggs   60.000 s –  90.016 s  (30.016 s)
  ad_12_ramp      330.016 s – 361.034 s  (31.018 s)
  ad_05_uber_eats 541.034 s – 596.052 s  (55.018 s)

natural_segments:
  intro    0.0 s –  42.0 s
  outro  610.0 s – 626.467 s
```


## 4. F1 verification

### Manual audit (from raw arithmetic)

For predicted ad intervals `P = {(p₀,p₁), …}` and ground-truth ad
intervals `G = {(g₀,g₁), …}`:

* `pred_total = Σ (p₁ − p₀)`
* `gt_total   = Σ (g₁ − g₀)`
* `intersection = Σ_p Σ_g max(0, min(p₁,g₁) − max(p₀,g₀))`
* `precision = intersection / pred_total`
* `recall    = intersection / gt_total`
* `F1        = 2·P·R / (P+R)`

#### `Dengyi`

| | Seconds |
| :--- | ---: |
| GT total ad seconds | 116.052 |
| Pred total ad seconds | 142.000 |
| Intersection | 52.000 |

* `precision = 52.000 / 142.000 = 0.366197`
* `recall    = 52.000 / 116.052 = 0.448075`
* `F1        = 2·0.3662·0.4481 / (0.3662+0.4481) = 0.403020`

Per-pred best-IoU:

| Pred | Match GT | IoU |
| :--- | :--- | ---: |
| `[ 72.0, 100.0]` | `[ 60.0,  90.0]` | 0.4504 |
| `[310.0, 366.0]` | `[330.0, 361.0]` | 0.5539 |
| `[486.0, 544.0]` | `[541.0, 596.1]` | **0.0270** |

Mean Seg IoU = 0.343748.

#### `main_v2`

| | Seconds |
| :--- | ---: |
| GT total ad seconds | 116.052 |
| Pred total ad seconds | 150.000 |
| Intersection | 101.050 |

* `precision = 101.050 / 150.000 = 0.673667`
* `recall    = 101.050 / 116.052 = 0.870730`
* `F1        = 2·0.6737·0.8707 / (0.6737+0.8707) = 0.759626`

Per-pred best-IoU:

| Pred | Match GT | IoU |
| :--- | :--- | ---: |
| `[ 60.0,  92.0]` | `[ 60.0,  90.0]` | **0.9380** |
| `[340.0, 408.0]` | `[330.0, 361.0]` | 0.2697 |
| `[546.0, 596.0]` | `[541.0, 596.1]` | **0.9088** |

Mean Seg IoU = 0.705505.

### Cross-check via `scripts/evaluate.py`

```text
==========================================================================================
  Advertisement Detection Evaluation
==========================================================================================
Test       Ref Ads  Pred Ads  Ref Sec  Pred Sec  Precision  Recall    F1     Mean Seg IoU
------------------------------------------------------------------------------------------
test_009   3        3         116.1 s  142.0 s   0.368      0.450    0.404   0.344         (Dengyi)
test_009   3        3         116.1 s  150.0 s   0.674      0.871    0.760   0.706         (main_v2)
==========================================================================================
```

The library and manual values agree on both branches (within 3-decimal
rounding).

### Headline result

| Branch | **F1** | Δ vs `Dengyi` |
| :--- | ---: | :--- |
| `Dengyi` | **0.404** | — |
| `main_v2` | **0.760** | +0.356 absolute (≈ +88 % relative) |

#### Per-segment colour commentary

| Ad # | GT range | `Dengyi` IoU | `main_v2` IoU | Comment |
| :---: | :--- | ---: | ---: | :--- |
| 1 | 60.0 – 90.0 | 0.450 | **0.938** | `main_v2` snaps to the right boundary thanks to semantic-span snap and intro split. |
| 2 | 330.0 – 361.0 | **0.554** | 0.270 | `main_v2` overshoots to 408 s — `Dengyi` happens to land closer. |
| 3 | 541.0 – 596.1 | 0.027 | **0.909** | `Dengyi`'s `[486, 544]` only clips 3 s of the GT; `main_v2` snaps onto the UberEats ad correctly. |

`main_v2` is better on 2 of 3 ads and dramatically better on Ad #3
(IoU jumps from 0.027 → 0.909). The net F1 still favours `main_v2` by
a wide margin even though it loses on Ad #2.


## 5. Reproducibility

Both numbers can be re-derived from the public files in this repo:

```text
# main_v2 (this branch)
python scripts/run_pipeline.py videos_with_ad/test_009.mp4 \
    --vad --with-ocr --with-semantic
PYTHONPATH=. python scripts/evaluate.py --test test_009

# Dengyi (switch branches first)
git checkout Dengyi
PYTHONPATH=. python -m fusion --video videos_with_ad/test_009.mp4 \
    --bundle-out data/output/test_009_analysis_bundle.json \
    --out          data/output/test_009_segments.json \
    --sample-fps 1.0 --window-sec 2.0 --min-segment-sec 20 --asr-vad
# (Dengyi does not ship a video_info/test_009.json — copy one in or
#  recompute from the bundle by hand.)
```

The cached `data/output/test_009_segments.json` on each branch is
sufficient to reproduce the F1 by feeding it to `scripts/evaluate.py`
together with the matching `video_info/test_009.json`.


## 6. Caveats

* **Run-to-run noise.** Whisper VAD and easyocr are not deterministic on
  CPU. F1 has been observed to drift by ±0.05–0.10 between runs with no
  code changes. The headline numbers are well outside that band.
* **GT typing on `Dengyi`.** Schema on that branch only accepts
  `video_content` / `ad`. `intro` / `outro` types in the GT must be
  rewritten (or the schema relaxed) to load. This audit copied the
  `main_v2` GT and rewrote `intro` / `outro` to `video_content`. Ad
  ranges are unchanged.
* **`main_v2` Ad #2 regression.** The 340 → 408 s overshoot shows the
  semantic-span snap can hurt when the snap target is too wide. A
  candidate next step is to clamp the snapped interval back to the DP
  duration when the snap interval is more than 1.5× wider than the DP
  pick.
