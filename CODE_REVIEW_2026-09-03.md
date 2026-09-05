# Code Review — 2026-09-03

Scope: `master...opencode/big-pickle` diff (plus uncommitted working-tree changes), high-effort recall-biased review.

## Correctness

### 1. Uncaught model-load exception crashes the app
**[nonogram_detector_application/main.cpp:312](nonogram_detector_application/main.cpp#L312)**

`DigitRecognizer`'s constructor throws on a missing/corrupt ONNX model and `main()` has no try/catch anywhere, so the whole app crashes instead of exiting cleanly.

Run the built binary from a directory where `digits.onnx` isn't found beside the exe or under `models/`, and `NONOGRAM_MODEL` is unset: `model_path` falls back to a relative path (line 310) that only exists if cwd happens to be the repo root. `cv::dnn::readNetFromONNX` fails, `DigitRecognizer`'s constructor throws `std::runtime_error` (digit_recognizer.cpp:44-48), and since `main()` never wraps this in try/catch, the process aborts with an unhandled exception instead of printing an error and returning a status code the way every other failure path in `main()` (image not read, grid not found) does.

### 2. `decode_clues` OR-check mismatches solver's AND-need
**[nonogram_detector/src/decode.cpp:65](nonogram_detector/src/decode.cpp#L65)**

`decode_clues` reports success (OR of top/left) even though the downstream solver path needs both strips (AND), producing a confusing failure instead of the intended "no clue regions decoded" message.

If only the left strip is detected (top strip missed due to glare/occlusion/perspective), `decode_clues` (line 65: `!detection.top.empty() || !detection.left.empty()`) still returns true. `main.cpp` then computes `W = clues.top.empty() ? 0 : clues.top[0].size()` = 0 while `H > 0` (main.cpp:386-387), builds a 0-column `constraints.cols`, and feeds it to `solve_nonogram`, which reports an opaque "invalid constraints" message instead of the accurate "no clue regions decoded" branch that exists specifically for total detection failure (main.cpp:457-460).

### 3. Column `digit_count` skips the row path's clamp
**[nonogram_detector_application/main.cpp:411](nonogram_detector_application/main.cpp#L411)**

Column `ClueCell.digit_count` is assigned raw from `clues.top_count` with no fallback, while the row path clamps the same kind of value, so rows and columns built from the same decoder can disagree on `digit_count`.

When no counter model is loaded (`digits_counter.onnx` missing, or `NONOGRAM_COUNTER_MODEL` misconfigured), `DigitRecognizer::digit_count` returns -1 even for a successfully-read cell, and decode.cpp:49 (`out_count.push_back(digit < 0 ? 0 : count)`) stores that -1 verbatim since `digit` is still >= 0. `main.cpp`'s row-building `clueline` lambda (393-406) clamps any out-of-[1,2] count to 1, but the column-building loop (411-415) assigns `clues.top_count[r][c]` straight into `ClueCell.digit_count` with no such fallback, so every column `ClueCell` ends up with `digit_count == -1` while the equivalent row cell gets 1 from identical source data.

### 4. Two-digit split reads skip confidence gating
**[nonogram_detector/src/digit_recognizer.cpp:216](nonogram_detector/src/digit_recognizer.cpp#L216)** — PLAUSIBLE

`recognize_two_digits` calls `recognize()` with the default `confidence_min=0.0`, so it never actually rejects a low-confidence half despite its header comment claiming it returns -1 when a half "cannot be read reliably".

`digit_recognizer.hpp:59` documents `recognize_two_digits` as returning -1 when "either half cannot be read reliably," but `recognize(left_up)`/`recognize(right_up)` (digit_recognizer.cpp:216-217) use the default `confidence_min=0.0`, and `recognize()`'s only -1 path besides that gate is an empty/no-foreground cell. A blurry or ambiguous glyph in one half of a two-digit clue cell (e.g. near-uniform softmax output, ~11% confidence but still the argmax) is accepted as a confident read and composed into the two-digit value fed straight to the solver.

## Efficiency

### 5. `recognize_two_digits` reruns `digit_count` needlessly
**[nonogram_detector/src/digit_recognizer.cpp:201](nonogram_detector/src/digit_recognizer.cpp#L201)**

`recognize_two_digits` re-invokes `digit_count()` on the same cell even though its only caller already computed that value and only calls it when it was 2, doubling the counter-model ONNX forward pass per two-digit cell.

`decode_region` (decode.cpp:33) already calls `recognizer.digit_count(cell)` and only invokes `recognize_two_digits` when `count==2` (decode.cpp:40). `recognize_two_digits` (digit_recognizer.cpp:201) calls `digit_count(cell)` again internally, re-running `prepare_input`'s normalization and the counter model's ONNX forward pass a second time for every genuine two-digit clue cell, with no way to pass in the already-known count.

## Reuse

### 6. Softmax logic duplicated across two functions
**[nonogram_detector/src/digit_recognizer.cpp:152](nonogram_detector/src/digit_recognizer.cpp#L152)**

The softmax normalization (minMaxLoc, subtract, exp, sum, divide) is hand-rolled identically in both `recognize_ex` and `digit_count_ex` instead of being factored into one shared helper.

digit_recognizer.cpp:152-159 (`recognize_ex`) and :187-194 (`digit_count_ex`) both implement the same 6-line minMaxLoc/exp/sum softmax block. A future numerical-stability fix or confidence-calculation change applied to one copy and not the other would silently make the digit-class confidence gate and the two-digit-probability gate diverge in behavior.

## Simplification

### 7. Cross-loc mat builders triplicate pad+augment
**[nonogram_detector/src/cross_locs_detector.cpp:483](nonogram_detector/src/cross_locs_detector.cpp#L483)**

`get_cross_locs_main_mat`, `get_cross_locs_top_mat` and `get_cross_locs_left_mat` each repeat the same "search, convert, pad with sentinel, copy at offset, augment" sequence, differing only in deltas and padding.

`get_cross_locs_main_mat` (483-539), `get_cross_locs_top_mat` (542-616) and `get_cross_locs_left_mat` (619-693) each independently call `get_cross_locs_map`, `convert_to_mat`, bail on empty, allocate a padded (-1,-1) canvas, `copyTo` at an offset, then call `augment`. The next fix to the pad-then-augment logic (e.g. an off-by-one in perimeter padding) must be hand-applied in three places, and a slip in one copy silently produces a differently-shaped cross_locs matrix for only one of main/top/left.

### 8. Solver duplicates solution-extraction per status
**[nonogram_solver/src/solver.cpp:76](nonogram_solver/src/solver.cpp#L76)**

The `Status::OK` and `Status::NOT_LINE_SOLVABLE` branches contain an identical solution-extraction block (resize `result.solution`, copy every `grid.get_tile(...)`) duplicated instead of factored into a shared helper.

solver.cpp:76-90 (`Status::OK`) and :92-108 (`Status::NOT_LINE_SOLVABLE`) each resize `result.solution` and loop over `grid.get_tile(x, y)` to fill it, differing only in the `line_solvable` value assigned. A bug fix to the tile-copy loop (e.g. a swapped x/y index) applied to one branch and not the other would leave the partial-grid "not fully line solvable" path silently wrong while the OK path looks correct in testing.

## Test coverage

### 9. Test reimplements `.non` format instead of calling it
**[nonogram_detector_ut/non_file_test.cpp:109](nonogram_detector_ut/non_file_test.cpp#L109)**

`write_non_file` is a static function only reachable from the application executable target, so the unit test hand-writes the same `.non` custom-block format itself instead of exercising the real function.

`write_non_file` lives in `nonogram_detector_application/main.cpp`'s anonymous namespace; the app's `CMakeLists.txt` builds only an executable with no library target exposing it, so `nonogram_detector_ut` (which links only `nonogram_detector`/`nonogram_solver`) can't call it. `non_file_test.cpp:109`'s comment admits it "mirrors write_non_file logic" and reimplements `parse_non_custom_block` by hand. If `write_non_file`'s field order or delimiters ever change, this test's hand-copied writer/reader won't be touched and keeps "passing" against stale format knowledge, no longer verifying the real code path.

## Altitude

### 10. `NG_CLUE_FIXES` bandaids one misread in `main()`
**[nonogram_detector_application/main.cpp:329](nonogram_detector_application/main.cpp#L329)**

A bespoke ground-truth-override text format (`NG_CLUE_FIXES`) is parsed inline in `main()` to patch one specific confirmed misread, instead of generalizing the two-digit disambiguation inside `DigitRecognizer`/`decode_region`.

main.cpp:329-374 adds an ad-hoc override mechanism (env `NG_CLUE_FIXES`, lines like `T421=5`) built specifically to patch one hand-drawn digit's misread, rather than fixing the underlying whole-cell-vs-split-read arbitration in `DigitRecognizer`. Any future puzzle with a similarly ambiguous glyph requires its own hand-authored fix file outside the detection pipeline instead of being resolved automatically, and the override machinery now permanently lives in shared application code for what was a one-off case.

---

## Note: candidate ruled out during review

An initial finder pass flagged `CrossLocsDetector::augment()`'s `while` loop ([cross_locs_detector.cpp:399](nonogram_detector/src/cross_locs_detector.cpp#L399)) as a possible infinite loop. Tracing `get_cross_locs_map`/`convert_to_mat` showed the matrix passed in is always a dense rectangular grid seeded with at least one resolved cell, so the loop is a standard bounded multi-source flood-fill over a 4-connected grid — it always terminates within the grid's diameter. Not included above.
