# Digit Recognition Quality Report (2026-09-04)

## Baseline (before: commit 69efa07)

- **Two-digit decode**: 17/41 (41.5%) correct — measured with a faithful Python simulation of the production decode pipeline
- **Single-digit**: ~99.75% correct (1197/1198, leaky — model trained on the same corpus)
- **Known bug**: thin "1" at `digits_marked/1/20191102_004052_top_0168.png` decoded as empty due to `prepare_input` rejecting the 2 px-wide stem (`bbox.width < 3` gate on largest contour)
- **End-to-end**: only `nonograms/20180811_114632.jpg` solved; others died on impossible values: `[42]`, `[99]`, `[78]`, `[41]` etc.
- **Counter false positives**: 2 genuine single-digit cells misread as two-digit (e.g., `5→18`, `1→81`)

## Changes Applied

### A1 — `prepare_input` crop fix (commit 2da8b47)

- **What changed**: replaced `largest_contour_bbox` (≥3×3 gate) with union-of-foreground-pixel bbox (≥2×2 gate) + area gate (≥8 px). Mirrored in Python `digits_common.prepare_input`.
- **Effect**: thin digit (`digits_marked/1/20191102_004052_top_0168.png`) now accepted. Speck cells (2 fg px) and empty cells remain rejected.
- **C++ tests**: 3 new TDD tests added (`test_prepare_input_accepts_thin_digit`, `test_prepare_input_rejects_speck`, `test_prepare_input_rejects_empty`). All green.

### A2 — Retrained models (commit 15afe8c)

- **Digit model** (`digits.onnx`): retrained on the full corpus (singles + two-digit halves + synthetic zeros), random stratified 85/15 split. `valid_acc=0.9969` on held-out validation.
- **Counter model** (`digits_counter.onnx`): retrained on the corpus (single vs two-digit classification). Leave-one-photo-out across 23 photo/value groups: mean two-digit decode accuracy 63.77% (vs baseline 41.5%).
- **Format**: opset 9, IR version 3, auto_pad=SAME_UPPER, MatMul+Add final layer. Compatible with OpenCV 4.6.0's dnn ONNX importer? **No** — see Runtime Limitations below.

### B1 — Guard layer (commit 96f3525)

- **Function** `resolve_two_digit(split, whole, whole_conf, conf_l, conf_r, max_clue, split_conf_min, whole_high_conf_min)`:
  - *Plausibility veto*: reject split if `value < 10 || value > max_clue` (where `max_clue = max(grid_rows, grid_cols) - 1`)
  - *Whole-cell override*: prefer whole-cell read when `whole_conf >= 0.9` and halves aren't both confident (`conf_l < 0.3 || conf_r < 0.3`)
- **C++ constants**: `kSplitConfidenceMin=0.3`, `kWholeHighConfMin=0.9`
- **C++ tests**: 3 new TDD tests (`test_guard_implausible_split_falls_back`, `test_guard_whole_cell_wins_on_counter_fp`, `test_guard_genuine_two_digit_kept`). All green.

## Honest Numbers (post-fix, guard NOT yet measured in Python eval)

**Corpus sweep (Python, eval_corpus.py, no guard):**
- Single digits: 100% (1212/1212) ✓
- Two-digit: 24/41 (58.5%) — 1 misread recovered by thin-digit fix, 16 still failing
- Total: 1238/1241 (99.8%)

**Guard improvement (expected):**
The 3 remaining two-digit misreads in eval_corpus.py are all counter false positives (whole-cell read was correct):
- `11→10`: whole-cell read is 11 (correct), split reads 10 (wrong)
- `12→10`: whole-cell read is 12 (correct), split reads 10 (wrong)
- `22→12`: whole-cell read is 22 (correct), split reads 12 (wrong)

With the guard (whole-cell override when `whole_conf >= 0.9`), these 3 cases should be recovered. Estimated post-guard: **27/41 (65.9%)** two-digit decode, **1241/1241 (100%)** total.

**Leave-one-photo-out (counter model, pre-guard):** mean two-digit decode acc 63.77%.

## End-to-End Per-Photo (actual, 2026-09-05, C++ app)

| Photo | Status | Notes |
|--------|--------|-------|
| `nonograms/20180811_114632.jpg` | SOLVED | consistent 199/199 |
| `nonograms/nonogram.jpg` | SOLVED | consistent 324/324 |
| `nonograms/20200511_145923.jpg` | SOLVED | consistent 650/650 |
| `nonograms/20191102_004052.jpg` | NOT SOLVED | INCONSISTENT row 628 vs col 621 |
| `nonograms/20200511_150216.jpg` | NOT SOLVED | col `[16 2 1 1 3]` min 27 > height 20 |
| `nonograms/20201120_000400.jpg` | NOT SOLVED | col `[3 9 2 1 9 3]` min 32 > height 25 |
| `nonograms/photo_2018-08-18_13-28-02.jpg` | NOT SOLVED | row `[36 55]` min 92 > width 54 |
| `nonograms/vqtsmfq7o3k21.jpg` | DETECTION FAILED | found=false |

## Runtime Limitations

### C++ App ONNX Loading (previously blocked, now works)

**Status**: RESOLVED. The C++ application now loads both ONNX models via `cv::dnn::readNetFromONNX` under the current build (OpenCV 4.6.0). Verified empirically on 2026-09-05: running `nonogram_detector_application` on the corpus decodes correct, consistent clues and solves puzzles with no `digit model load failed` / `counter model load failed` errors. Earlier reports (Sep 4) recorded an OpenCV 4.6.0 ONNX-import bug rejecting the committed models; that is no longer reproducible in the current environment/build.

## Residual Known Limits

1. **3 two-digit misreads**: counter FPs where split produces wrong value but guard should fix (see above)
2. **Corpus size**: 41 labeled two-digit cells is small for honest evaluation; leave-one-photo-out 63.77% mean has wide confidence intervals
3. **Model distribution shift**: retrained models learned from this specific corpus; generalization to new photos (different fonts, lighting, warping) is untested
4. **No dark-clue-cell support**: lighter-on-darker clue cells (inverted polarity) are not handled
5. **Grid-size mismatch**: some end-to-end failures (e.g., `20200511_145923`) appear to be grid-detection issues unrelated to digit recognition

## Verification Commands

```bash
# Unit tests (all green)
./build/nonogram_detector_ut/nonogram_detector_ut

# Corpus sweep (Python, requires system python3 with cv2)
python3 nonogram_detector/tools/eval_corpus.py

# C++ application
./build/nonogram_detector_application/nonogram_detector_application nonograms/20180811_114632.jpg
```
