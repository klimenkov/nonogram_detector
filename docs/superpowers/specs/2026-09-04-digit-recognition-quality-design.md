# Digit Recognition Quality — Single and Two-Digit Clue Cells (Design)

**Date:** 2026-09-04
**Status:** Approved (brainstorm) — ready for implementation planning
**Scope:** Improve the quality of (a) selecting clue cells and (b) recognizing the digits in them, for both one-digit and two-digit numbers. Measured, not assumed.

---

## 1. Context

### 1.1 The pipeline today
1. `CrossLocsDetector.detect` finds the grid and produces `main` / `top` / `left` cross-location matrices (subpixel, CV_32FC2, `(-1,-1)` sentinel). The top/left are the clue strips.
2. `get_cell_warped_images_vector` (in `image_operations.cpp`) perspective-warps every clue cell to **20×20**.
3. `decode_region` (in `decode.cpp`) runs each cell through `DigitRecognizer`:
   - `digit_count()` (counter model, `digits_counter.onnx`) decides 1 vs 2 digits.
   - If 2: `recognize_two_digits()` splits the cell into left/right halves, upscales ×3, reads each half, composes `left*10+right`; on failure falls back to whole-cell read.
   - Else: whole-cell `recognize()`.
4. The clue strips then feed `clue_corrector` + solver (in the application).

`prepare_input` (`digit_recognizer.cpp`) normalizes a 20×20 cell to the model input: BGR→gray, Otsu `THRESH_BINARY_INV`, crop a 4 px margin ring (20% of the cell) to strip the grid frame, take the **largest contour** bounding box, require it `≥ 3×3 px`, pad 2 px, aspect-preserving resize to a 28×28 canvas, whiten `(x/255 − 0.1307)/0.3081`.

### 1.2 What was measured (2026-09-04, corpus sweep)
A faithful Python reproduction of production recognition was run over every labeled 20×20 warped cell in `digits_marked/` (the ground-truth corpus: folders `1..9`, `10..22`, `-1` empty, `e` excluded). Same `prepare_input` math, same ONNX models, same `decode_region` decision rule. `e/` is a byte-identical duplicate of `-1/` (both are empty spacer cells in the clue strips) and is not a misread source.

**Single digits — nearly solved (but leaky).** 99.75% (3/1198 wrong):
- `digits_marked/1/20191102_004052_top_0168.png`: a genuine `1` decoded as **empty** (`dc=-1, pred=-1`). Root cause (verified): the 4 px margin crop splits the thin `1` (2 px-wide stem) into fragments; the largest contour's bbox is `2 px wide < 3` → `prepare_input` rejects the cell as "no reliable foreground" and the model never runs.
- `digits_marked/1/nonogram_left_0082.png`: counter false-positive (dc=2) → split composed `81`.
- `digits_marked/5/20191102_004052_left_0148.png`: counter false-positive (dc=2, p2=0.86) → split composed `18`.

Caveat: the committed `digits.onnx` was trained on these same images, so 99%+ is optimistic; the 3 failures are the survivors that even memorization could not fix.

**Two digits — the real problem.** 17/41 (41.5%) decoded correctly:

| Failure mode | Examples | Count |
|---|---|---|
| Counter **misses** a 2-digit cell (dc=1) → whole-cell read collapses it | `10→1`, `11→1`, `12→2`, `14→4`, `15→5`, `18→1` | 14 |
| Split halves **misread** | `10→13`(×4, `0→3`), `16→15`(`6→5`), `18→15`(`8→5`), `21→41`, `22→42` (`2→4`) | 8 |
| Counter **false-positives** single digits → split composes garbage | `5→18`, `1→81` (single-digit cells) | 2 |

**End-to-end effect.** Running the current app on every `nonograms/*.jpg`: only `20180811_114632` solves; every other photo dies on an impossible clue produced by the two-digit misreads (`[42]`, `[99]`, `[98]`, `[78]`, `[41]`).

**Couldn't-catch:** `0` alone barely exists in the corpus (only inside two-digit halves); the old model had no real standalone `0` samples.

### 1.3 Image-inherent boundaries
Some ambiguity is inherent to 20×20 warped photos: low resolution, perspective-warp interpolation, handwriting. The retrained model + guards improve the *systematic* failures; a few residual ambiguous cells may remain and are judged honestly by the held-out numbers, not by a hardcoded override.

---

## 2. Goals / Non-goals

**Goals**
1. Fix the confirmed preprocessing bug: thin single digits must not be discarded as empty.
2. Retrain `digits.onnx` (10 classes, now with real `0` halves) and `digits_counter.onnx` (single/empty vs two-digit) on the full labeled corpus, using the **fixed** `prepare_input` so train == inference.
3. Measure honestly: random stratified split for single digits; **leave-one-photo-out** for counter + two-digit decode behavior. Report before/after numbers.
4. Add a validated guard layer in the decode decision so the common failure modes lose.
5. Keep the whole thing offline, tiny, and reproducible — training/eval tooling committed under `nonogram_detector/tools/` (never again `/tmp`-only).

**Non-goals**
- No change to grid detection / clue-cell *selection* geometry (cross-locs, warp size stays 20×20 — the corpus and models are 20×20-distributed).
- No `NG_CLUE_FIXES`-style hardcoded overrides for these failures; correctness comes from the models + guards, not per-cell exceptions.
- No new heavy dependency in the runtime pipeline (still OpenCV `dnn` + ONNX; PyTorch is a **build/train-time** tool only).
- No architectural rewrite of the recognizer classes.

---

## 3. Design

### 3.1 A1 — `prepare_input` crop fix (`digit_recognizer.cpp`)
Replace the "largest single contour, bbox `≥ 3×3`" step with:

- Compute the bounding box of the **union of all contours** in the margin-cropped inner region (equivalently: the min/max of foreground pixel coordinates), not the bbox of one contour.
- Accept the cell when the union bbox `width ≥ 2` **and** `height ≥ 2` **and** foreground **area ≥ 8 px²** (replaces the meaningless strict `≥3×3`);
  a thin 2×11 stem (22 px²) passes; a noise speck (a few px) still fails.
- All other steps in `prepare_input` (Otsu, margin, pad, aspect-preserving 28×28, whitening) are **unchanged**, so retrained models see the same distribution at inference.

Why this is safe: the 4 px margin ring already removed the grid frame; post-margin ink is the digit, so the area filter alone is sufficient anti-noise; thin strokes that were previously lost now survive.

### 3.2 A2 — Retrain both ONNX models (PyTorch, build-time only)
Both models keep the existing architecture exactly (verified from the ONNX graphs):

```
Conv2d(1→8, k5, pad2) → ReLU → MaxPool2(k2,s2)
  → Conv2d(8→16, k5, pad2) → ReLU → MaxPool3(k3,s3)
  → Flatten(16·4·4=256) → Linear(256→10)            [digits; 256→2 for counter]
Input 1×1×28×28, output 1×10 / 1×2 logits. Export ONNX at **opset 13** (the existing models are opset 9; the runtime requirement is only the same input/output contract — gate 5 verifies that OpenCV dnn loads the exported file before it is committed). 
```

**Training data** (all cells prepped through the *fixed* `prepare_input`, mirrored exactly in Python):

| Model | Class | Cells |
|---|---|---|
| `digits.onnx` | digit 1–9 | folders `1..9` whole cells |
| | digit 0–9 (halves) | split every two-digit cell (`10..22`) → left `value//10`, right `value%10` (this is the real-`0` source: right halves of `10`,`20`) |
| | (supplement) | synthetic `0` renders only if the real-`0` count stays far below the class floor; primary source is real halves |
| `digits_counter.onnx` | 0 = single / empty | folders `1..9` whole + `-1` whole |
| | 1 = two-digit | folders `10..22` whole |

- **Class imbalance:** counter sees ~2000 class-0 vs 41 class-1 cells → weighted sampling (class weight) plus a few **strong augmentation** copies (affine/scale/blur/threshold-jitter) of the scarce two-digit cells.
- **Augmentation** for all classes is mild (consistent with real photo warp artifacts) and produces the same prepared 28×28 input.
- **Export:** `torch.onnx.export` with a fixed opset compatible with the runtime OpenCV `dnn` backend; outputs stay raw logits (softmax remains on the C++ side), so `recognize_ex`/`digit_count_ex` contracts are unchanged.

### 3.3 A3 — Honest evaluation protocol
- **Single digits:** random stratified train/valid split (e.g. 85/15) over `1..9` cells + halves; report per-class accuracy and the full confusion matrix.
- **Counter + two-digit decode behavior:** **leave-one-photo-out** across every photo that contains two-digit cells (train on all other photos' cells, evaluate the *entire* decode decision — counter → split/whole → composed value — on that photo's cells; rotate; average). This is the number that answers "will a new photo work?"
- Record both the leaky in-corpus number (old pipelines) and the honest held-out numbers in the final report.

### 3.4 A4 — Committed tooling (reproducibility)
New `nonogram_detector/tools/`:
- `train_digits.py` — builds the datasets above, trains the counter and digit models, exports both ONNX files, runs the eval protocol, writes the report into `docs/superpowers/reports/`.
- `eval_corpus.py` — reproducer for the corpus sweep + held-out metrics (the counterpart of this spec's measurements).
- The Python `prepare_input` mirror and the C++ `prepare_input` are kept visibly in sync (documented in both places; unit test the C++ side to match expectations).

### 3.5 B1 — Decode guard layer (validated subset)
Add a parameterized guard layer in `decode_region` / `recognize_two_digits`. Thresholds are **fixed by the leave-one-photo-out measurements**, and a guard is kept only if it reduces held-out errors (no cargo-culting; the chosen values and the confusions they fix are recorded in the report):

1. **Plausibility veto** — accept a split result only if `10 ≤ value ≤ max(grid_height, grid_width)`. Otherwise fall back to the whole-cell read. (A two-digit clue cannot be `< 10` or exceed the puzzle size.)
2. **Whole-cell consistency check** — when the counter says 2 but the *whole-cell single read* is a **high-confidence** digit and the split halves are not both confident, prefer the whole-cell read. Targets counter false-positives (`5→18`, `1→81`) — the whole-cell read of those is the correct digit at high confidence; genuine two-digit cells' whole-cell read is the collapsed low-confidence garbage.
3. **Split-half confidence floor** — keep and retune the existing `0.3` floor against held-out halves so confident-but-wrong halves (`10→13`: both halves confident) are caught without forcing benign fallbacks.

### 3.6 B2 — End-to-end gates
- `nonograms/20180811_114632.jpg` must still decode consistently (199/199) and solve.
- Each other `nonograms/*.jpg`: re-run the app; the prior impossible-clue failures (`[42]`, `[99]`, `[98]`, `[78]`, `[41]`) must be gone or materially reduced; record per-photo consistency + solve status before/after, cross-checked against corpus ground truth where the cells overlap.

### 3.7 B3 — Unit tests (`nonogram_detector_ut`)
- `prepare_input` thin-digit regression: a synthetic thin `1` (≈2 px-wide stem) must produce a valid input (not rejected); a noise-speck cell must still be rejected as empty.
- Guard layer: implausible split value veto and whole-cell-vs-split preference, exercised with synthetic cells.
- All existing tests remain green.

---

## 4. Data flow summary

```
corpus (digits_marked/)  ──train tool──►  digits.onnx (10 classes)
                ├──────────────────────►  digits_counter.onnx (2 classes)
                └──eval tool──►  report: random-split digits acc + leave-one-photo-out
                                 two-digit decode acc (before/after)

runtime: warped cell → prepare_input (fixed crop) → counter → single/split → guard → digit/empty
```

---

## 5. Files touched

| Path | Change |
|---|---|
| `nonogram_detector/src/digit_recognizer.cpp` | `prepare_input` crop fix (§3.1); guard-support helpers if needed (§3.5) |
| `nonogram_detector/src/decode.cpp` | guard layer in `decode_region` + pass grid dims (§3.5) |
| `nonogram_detector/include/digit_recognizer.hpp` | any new guard helper signatures (minimal) |
| `nonogram_detector/models/digits.onnx` | replaced by retrained model |
| `nonogram_detector/models/digits_counter.onnx` | replaced by retrained model |
| `nonogram_detector/tools/train_digits.py` | **new** — dataset build, train, export, eval, report |
| `nonogram_detector/tools/eval_corpus.py` | **new** — corpus sweep reproducer + held-out metrics |
| `nonogram_detector_ut/main.cpp` | thin-digit regression + guard tests |
| `docs/superpowers/reports/2026-09-04-digit-recognition-quality.md` | **new** — honest before/after report |

No changes to: warp size (`cell_warped_side_length=20`), `CrossLocsDetector`, `image_operations.cpp`, `clue_corrector`, the application's solve path.

---

## 6. Verification gates

1. **Unit tests** — `cmake --build build && ./build/nonogram_detector_ut/nonogram_detector_ut` → "all tests passed".
2. **Retraining metrics** — `tools/train_digits.py` completes and writes the report; single-digit held-out accuracy and two-digit leave-one-photo-out accuracy both reported (target: two-digit decode errors reduced vs the 41.5% baseline and no single-digit regression beyond the measured baseline).
3. **Preprocessing regression** — the thin `1` (`20191102_004052_top_0168` cell) no longer decodes as empty.
4. **End-to-end** — `20180811_114632` still consistent + solved; the previously failing photos no longer emit the flagged impossible values.
5. **Model compat** — the app loads the retrained ONNX through the existing `DigitRecognizer` path with no API break.

---

## 7. Risks / mitigations

| Risk | Mitigation |
|---|---|
| Tiny two-digit set (41 cells) → noisy held-out numbers | Leave-one-photo-out + strong augmentation; report variance; single-digit uses the big random split |
| Guard thresholds overfit the small held-out set | Thresholds chosen on leave-one-photo-out aggregates, not on a single photo; guards kept only on measured benefit |
| High-confidence whole-cell read vetoing a *real* two-digit cell | Only veto when split halves are NOT both confident AND whole-cell conf is high; empirically checked on all 41 two-digit cells |
| ONNX export op mismatch with OpenCV dnn | Export at a pinned opset (13) and verify by loading both models through `DigitRecognizer` (gate 5) before committing |
| Retrain "memorizes" corpus (leakage) → misleading success | The honest per-photo hold-out is the reported metric; leakage affects only the random-split digit number, which we disclose as optimistic |
| Augmentation drifts too far from real warps | Keep augmentation mild (scale/translate/blur only); validate no accuracy loss on the held-out photo set |