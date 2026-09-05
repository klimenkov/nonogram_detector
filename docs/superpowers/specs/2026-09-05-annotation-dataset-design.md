# Labeled-Corpus Annotation Workflow — Browser Review, Model-Prefilled Labels (Design)

**Date:** 2026-09-05
**Status:** Approved (brainstorm) — ready for implementation planning
**Scope:** A repeatable workflow to grow the labeled `digits_marked/` corpus from any input photo: the C++ app exports every warped clue cell with its model prediction, a browser page lets the user visually review and correct the labels, and the corrected labels feed the existing retraining pipeline.

---

## 1. Context

### 1.1 The pipeline today
1. `CrossLocsDetector.detect` finds the grid and produces `main` / `top` / `left` cross-location matrices (subpixel, CV_32FC2, `(-1,-1)` sentinel). The top/left are the clue strips.
2. `get_cell_warped_images_vector` (image_operations.cpp) perspective-warps every clue cell to **20×20**.
3. `decode_region` (decode.cpp) runs each cell through `DigitRecognizer`: counter model (`digit_count`) decides 1 vs 2 digits; two-digit cells are split into halves, ×3-upscaled, read per half, composed, then run through `resolve_two_digit` (guard) + `sanitize_clue_digit`; single cells are whole-cell recognized.
4. The clue strips feed `clue_corrector` + solver in the application.

### 1.2 The labeled corpus
`digits_marked/` (untracked data dir) is a hand-labeled corpus of 20×20 warped cells: folders `1..9` (~1200 cells), `10..22` (41 cells), `-1` empty (~830). `e/` is a byte-identical duplicate of `-1` and is not a training source.
- `train_digits.py` builds datasets from the folders, trains `digits.onnx` (10 classes) and `digits_counter.onnx` (2 classes), exports ONNX, runs the eval protocols (random stratified split for single digits; leave-one-photo-out for counter/two-digit), and writes the report.
- `eval_corpus.py` sweeps the corpus with the committed models and prints per-class accuracy + misreads.

Labeling today means manually creating/curating those folders. The user wants a faster, visual, repeatable way to mark cells for any photo.

### 1.3 Chosen approach (brainstorm, 2026-09-05)
- **Browser review page** with active learning: the model's prediction is pre-filled; existing `digits_marked` labels are shown as hints.
- **Full re-review with old labels**: for every cell the reviewer sees prediction + old label (when the old label is alignable — see §3.2 notes); re-review can also correct old mislabels.
- **Any photo**: the tool takes an image path like the app, runs detection automatically.
- **C++ export / Python review boundary**: the app dumps the exact production warped cells + predictions; Python (stdlib only) serves the page and writes labels into `digits_marked/`.

---

## 2. Goals / Non-goals

**Goals**
1. Export every clue-strip cell (top + left) of any photo to disk, with the model's production prediction and confidence, using the exact production code path (warp → recognize).
2. A local browser page to review cells arranged in the real strip layout, correct labels with minimal keystrokes, and save incrementally.
3. Materialize the reviewed labels into `digits_marked/<label>/` with deterministic names; re-review corrects old mislabels.
4. Keep `train_digits.py` unchanged; make the eval protocol honest about labels that came from the model itself.
5. Zero new Python runtime dependencies (stdlib only annotator).

**Non-goals**
- No grid-geometry correction in the browser (approach C deferred; misaligned grids stay out of scope).
- No separate `e/` writing (`e` is a byte duplicate of `-1`).
- No standalone `0` class (a stray whole-cell `0` is treated as empty, matching production).
- No change to the runtime recognizer architecture, models, or warp size.

---

## 3. Design

### 3.1 C++ — shared per-cell decoder + export

**Refactor `decode_region`** so the per-cell decision lives in one shared helper used by both the existing `decode_clues` and the new exporter. Per-cell info surfaced via a new struct:

```cpp
// decode.hpp
struct ClueCellInfo {
    cv::Mat cell;          // warped 20x20 (the exact pixels training will see)
    int    digit;          // final production label (-1 empty) — mirrors decode_clues
    int    count;          // counter digit-count (1/2, 0 for empty)
    double whole_conf;     // softmax confidence of the whole-cell read
    double conf_l, conf_r; // split-half confidences (0 when not two-digit)
};

bool decode_clues_ex(cv::Mat const& image, Detection const& detection,
                     DigitRecognizer const& recognizer,
                     ClueGrid& out,
                     std::vector<std::vector<ClueCellInfo>>& top_info,
                     std::vector<std::vector<ClueCellInfo>>& left_info);
```

- `decode_clues` becomes a thin wrapper (fills `out` only) so the solve path is untouched.
- `ClueCellInfo::digit` must equal `ClueGrid::top/left` at the same position (enforced by a unit test).

**Env-gated exporter** in `main.cpp`, activated with `NG_EXPORT_CELLS=<dir>`:
- Photo id = image path basename without extension.
- After detection + recognizer load, calls `decode_clues_ex` and writes:
  - `cells/<photo>_top_<rrrr>_<cccc>.png`, `cells/<photo>_left_<rrrr>_<cccc>.png` (4-digit zero-padded grid row/col) for every cell.
  - `index.json`: photo id; strip dims; per cell `{pos, row, col, png, predicted, count, whole_conf, conf_l, conf_r}` with `pos` = reading-order index (row-major) used for old-label matching.
- Sentinel cells (cross_locs `(-1,-1)`) produce empty warps; they are exported as cells the reviewer can mark empty.

**Unit tests** (nonogram_detector_ut):
- `decode_clues_ex` digit grid equals `decode_clues` grid exactly on a synthetic warp.
- Per-cell `ClueCellInfo` fields populated for a single-digit and a two-digit synthetic cell.

### 3.2 Python annotator (`nonogram_detector/tools/annotate.py`, stdlib only)

**`annotate.py serve <cells_dir>`**
- Serves a page at `http://127.0.0.1:8899` (bind `127.0.0.1`, stdlib `http.server`).
- Renders the top/left strips as two grids in the true cell layout; each cell = upscaled warped PNG with a border color:
  - **green** = prediction and old label agree / stable,
  - **amber** = low whole-cell confidence (< 0.6),
  - **red** = old label and prediction conflict.
- Interaction: click a cell → bottom bar shows it enlarged with a text input. Type `1`..`9`, `10+` (two-digit), `-`/`e` (empty); Enter saves and advances to the next cell; `Space` confirms the shown label as human-verified; Esc skips (unmarked for now).
- A "keep remaining as prediction" button sets the untouched cells to accepted-prediction (so the reviewer folds in everything without clicking through; the flagged amber/red cells are reviewed first). Cells that already carry an old `digits_marked` label are left alone and still resolve as carried-over (rule 3). This is **not** the same as per-cell Confirm: it marks cells `verified=false`, never `human`.
- Old-label hints from `digits_marked/` matched by (photo, strip, `pos`). Deterministic `<photo>_<strip>_<rrrr>_<cccc>.png` names always align. Legacy sequential names only align when a strip's indices form a complete unique `0..N-1` set (they do not for the current corpus — its indices are per-dump-run counters that overlap across runs); for those photos the first review pass is predictions-only and old labels become visible after the first deterministic export.
- State saved incrementally to `labels.json` in the cells dir (quit/restart safe).

**Label semantics** per cell (resolved at export time):
1. Explicitly typed value → **human**, `verified=true`.
2. Confirmed via `Space` / confirm button → **human**, `verified=true`.
3. No action but an old corpus label exists → **carried-over**, `verified=true`.
4. Otherwise, when "keep remaining as prediction" was used → **accepted-prediction** (model label), `verified=false`.

Cells with no action, no old label, and no confirm-all are **excluded from the manifest** — the reviewer deliberately skipped them (or the review is a partial pass); they stay unlabeled. The "keep remaining" button is the explicit opt-in for folding in the rest.

**`annotate.py export <cells_dir>`**
- Materializes into `digits_marked/<value>/` (`1`..`9`, two-digit `10+`, empty → `-1`).
- Deterministic filenames `<photo>_top_<rrrr>_<cccc>.png` so re-runs overwrite rather than duplicate.
- Removes that photo's previously-labeled cells from other label folders (re-review corrects old errors).
- Writes `manifest.csv` (photo, strip, pos, value, source, verified) in the cells dir for eval tooling.

### 3.3 Retraining + honest evaluation (unchanged pipeline)

- `train_digits.py` reads the folder corpus as today — no code change.
- `eval_corpus.py` gains `--verified-only <manifest.csv>`: counts only human/carried-over labels as ground truth, treating accepted-prediction cells as unlabeled (avoids self-agreement bias for the model that produced the predictions). Without the flag, behaves exactly as today.

### 3.4 Output location / git hygiene

- Annotator working files live under a per-run output dir (e.g. passed to `NG_EXPORT_CELLS`), gitignored.
- `digits_marked/` and `nonograms/` are already untracked data dirs; the tool only writes there.

---

## 4. Data flow summary

```
photo ──C++ app (NG_EXPORT_CELLS)──► cells/*.png + index.json
                                          │
                      annotate.py serve ◄──┘  (browser review, labels.json)
                                          │
                       annotate.py export ─┘
                                          ▼
                          digits_marked/{1..9,10..22,-1}/
                                          │
                          train_digits.py ▼        eval_corpus.py --verified-only
                              digits.onnx          report (before/after)
                              digits_counter.onnx
```

---

## 5. Files touched

| Path | Change |
|---|---|
| `nonogram_detector/src/decode.cpp` | shared per-cell decision helper; `decode_clues_ex`; `decode_clues` wrapper |
| `nonogram_detector/include/decode.hpp` | `ClueCellInfo` + `decode_clues_ex` |
| `nonogram_detector_application/main.cpp` | `NG_EXPORT_CELLS` export block |
| `nonogram_detector/tools/annotate.py` | **new** — serve/export (stdlib only) |
| `nonogram_detector/tools/eval_corpus.py` | `--verified-only` flag |
| `nonogram_detector_ut/main.cpp` | decode_clues_ex parity + ClueCellInfo tests |
| `.gitignore` | annotator output dir |
| `docs/superpowers/reports/` | updated numbers after first retraining on grown corpus |

No changes to: `CrossLocsDetector`, `image_operations.cpp`, `digit_recognizer.*`, `train_digits.py`, `clue_corrector`, solver.

---

## 6. Verification gates

1. C++ unit tests green (`decode_clues_ex` parity with `decode_clues`; `ClueCellInfo` populated).
2. Export all 8 corpus photos → `index.json` + PNG counts match the warped-cell counts of the strips; no crash on vqtsmfq (currently found=true via INTER_AREA fallback).
3. Auto-pass over one photo with no edits: every label equals either the old corpus label or the model prediction (checks label-resolution logic).
4. Manual: serve one photo, fix a couple of cells, export; corrected files land in the right `digits_marked/` folder and the old wrong entries are removed.
5. Full loop: `train_digits.py` reruns after a materialization; report updates; the app loads the retrained ONNX with no API break.

---

## 7. Risks / mitigations

| Risk | Mitigation |
|---|---|
| Old-label alignment off when a strip's dimensions changed since the corpus dump | Hints dropped (with a notice) on per-strip dimension mismatch; reviewer sees cell image + prediction side-by-side so displacement is caught by eye; re-marking fixes it |
| Self-agreement bias: model predictions masquerading as ground truth | Source/verified flags in manifest + `eval_corpus.py --verified-only`; report distinguishes human vs accepted-prediction counts |
| Corrupting old corpus with a bad map | Re-review is user-controlled; export removes only the target photo's cells from other label folders; `git`/filesystem keeps `digits_marked` out of the repo, so a clone is a safe backup point |
| Browser page not available (headless box) | Annotator is stdlib; reviewer can also drop cells into folders manually as a fallback (documented) |
| Huge cell counts (top+left may be ~2× (rows+cols)) | Pre-filled predictions + "confirm all remaining" make review proportional to corrections, not cells |