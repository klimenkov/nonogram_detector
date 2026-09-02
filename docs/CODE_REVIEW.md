# Code Review — nonogram_detector

**Reviewer:** OpenCode (C++ / OpenCV)
**Branch:** opencode/big-pickle
**Commit reviewed:** 9c36355 ("Cells structure") + subsequent doc commit
**Date:** 2026-09-02

This is a technical review of the `nonogram_detector` codebase from a C++ and
OpenCV engineering perspective. It covers correctness, performance, code quality,
and robustness, and ends with prioritized improvement recommendations.

---

## 1. Summary

The overall design (template-matching the grid via convolution with hand-built
cross/square kernels, then flood-filling) is reasonable and works for the target
use case. However the code is at the *research/prototype* stage: it contains live
debug instrumentation, is heavily Windows-path-hardcoded, duplicates a class,
and has several performance bottlenecks that will matter as soon as the input
grid gets large or runs unattended.

**Verdict:** Keep the architecture; harden and optimize the hotspots; remove the
debug scaffolding.

---

## 2. Correctness & robustness issues

### 2.1 Live `imshow`/`waitKey` blocks `detect()` (Critical)
`cross_locs_detector.cpp:55-64` — `detect()` calls `cv::imshow("image_thresholded",
...)` followed by `cv::waitKey(0)`. This **blocks forever** waiting for a keypress
on every run and requires an interactive display. The detector cannot be used
headless, in a test harness, or in a server pipeline. It must be gated behind a
debug flag / `#ifdef`, or removed.

### 2.2 Same class defined twice (Critical design smell)
`grid_detector.hpp/.cpp` defines the *same* class `CrossLocsDetector` (a slightly
older variant whose `detect()` returns `std::tuple<cv::Mat, cv::Mat, cv::Mat>`
with no found-flag and no empty checks). It is NOT in the library's CMake
`SOURCES`, so it is never compiled or linked. This is confusing and risks ODR /
ambiguity if someone adds it to the build. It should be deleted or renamed
(e.g. `GridDetector`) and only one kept.

### 2.3 `assert()` used for runtime validation
`masks.cpp:18,34` — `assert(length % 2 == 1)` only fires in debug builds (and only
if NDEBUG is not defined); in release the constructor proceeds silently with an
even mask, producing wrong results. Prefer explicit validation with a real error
message, or document the precondition.

### 2.4 Division by zero risk with `scale`
`detect()` divides by `scale`. `resize()` computes `scale =
max_dest / max(rows, cols)`. If the image is empty (0 rows/cols) this is a div-by-0;
callers call `imread` but don't guard before `detect()`. Guard at the entry point.

### 2.5 Unchecked ROI / empty matrix paths
`find_kernel_loc` correctly checks `is_inside`, but `get_roi(center, size)` can
produce a rect that extends past the image bounds (negative origin) when the
center is near the border. `find_kernel_loc` returns `false` in that case, but the
semantic is easy to get wrong. `get_cross_locs_main_mat` returns an empty `Mat` in
one path but not in others; callers that consume the result must check `empty()`.

### 2.6 `augment` ignores its `image_resized` argument
`augment(cv::Mat image_resized, ...)` receives an image that is never used (all
call sites pass `cv::Mat()`). Dead parameter — remove it.

---

## 3. Performance bottlenecks

### 3.1 Naive per-node full `filter2D` (Dominant cost — High)
`get_cross_locs_map` (cross_locs_detector.cpp:156-235) runs a BFS. For every
located cross it calls `find_kernel_loc`, which executes a full
`cv::filter2D` over a ROI the size of `(2*cell_side_length)²` with a kernel of
size `~1.5*cell_side_length` (main) or `cell_side_length` (clue regions).

Cost per node: `O(ROI_area · kernel_area)`. For a grid of `N×N` cells with side
`S`, the whole main grid is `O(N² · (2S)² · (1.5S)²)` = `O(N² · S⁴)`. For large
photos this is very slow. Each node re-filters overlapping regions that could be
reused.

**Improvements (in increasing order of impact):**
1. **Shrink the search region.** Each cross is predicted from its neighbor and
   only needs to be refined slightly. Use a small ROI (e.g. `1.5*S`) centered on
   the prediction instead of `2*S`.
2. **Use a separable / smaller kernel.** The cross mask is separable; decomposing
   it into two 1-D passes turns `O(k²)` into `O(k)` per pixel.
3. **Match-template via `matchTemplate` (TM_CCORR_NORMED or SQDIFF_NORMED)** on a
   small window instead of raw `filter2D` + manual normalization. OpenCV's
   implementation is SIMD-optimized.
4. **Compute the grid analytically after the first few anchors.** Once 2 rows and
   columns are reliably located, the grid is near-uniform; you can extrapolate the
   remaining `cross_locs` positions with the known `cell_side_length` and only
   verify a sparse subset. This changes the runtime from `O(N²)` convolutions to
   `O(N)`.
5. **Parallelize the BFS frontier** with `std::execution::parallel` or TBB, since
   each node's search is independent.

### 3.2 Cell-size scan is wasteful (Medium)
`find_cell_side_length_cell_loc` (cross_locs_detector.cpp:123-153) loops every
candidate side length `L ∈ [min,max]` and does `filter2D` with an `L×L` square
mask over a 150×150 ROI. Better: do a **single** FFT/correlation with a known
small probe, or derive the periodicity from a 1-D profile (row/column projection
separately) — the grid is a near-uniform texture so the dominant spatial
frequency gives `cell_side_length` directly. This reduces a loop of ~45
`filter2D`s to a couple of 1-D autocorrelations.

### 3.3 Redundant rescaling / integer math
Results are computed in integer pixel space then divided by `scale` at the end.
Fine, but `cv::Mat / scale` on a `CV_32SC2` matrix does per-element float
division — negligible compared to §3.1.

### 3.4 Repeated mask construction
Masks are rebuilt inside loops (e.g. `get_mask_square` per candidate length, and
per region). These are tiny — negligible — but could be precomputed/cached if this
ever becomes hot.

---

## 4. Code quality

### 4.1 Formatting / style
- Mixed tabs and spacing; long lines.
- `std::cout` debug output scattered through `detect()` and helpers
  (e.g. `line_width:`, `cell_side_length:`, `cross_locs.size()` in
  image_operations.cpp:124).
- Many commented-out debug blocks (imshow/draw/print) that should be deleted.

### 4.2 Return types
`detect()` returns a 4-tuple `std::tuple<bool, cv::Mat, cv::Mat, cv::Mat>`.
Replace with a small struct, e.g.:
```cpp
struct Detection {
    bool found = false;
    cv::Mat main, top, left;   // cross_locs matrices
};
```
This is self-documenting and avoids `std::tie` ordering mistakes.

### 4.3 Const-correctness & ownership
Reasonably const-correct already. `augment` takes `cv::Mat image_resized` (not
const) that it never uses — remove (see 2.6).

### 4.4 Naming / organization
- `INDICES_DELTA_UP/RIGHT/...` and `cross_loc_deltas` are clearer if grouped as
  structs/direction tables rather than parallel `std::vector<cv::Point>`s.
- `print()` is dead debug code.

### 4.5 Build system
- `cmake_minimum_required(VERSION 2.8)` is ancient; bump to 3.10+.
- Top-level `CMakeLists.txt` hardcodes subdirectories; fine for now.
- `find_package(OpenCV REQUIRED)` — consider `OpenCV_DEPENDENCIES` / link
  specific modules (`opencv_core`, `opencv_imgproc`, `opencv_highgui`) instead of
  all `OpenCV_LIBS` to trim binary size.
- The `application` and `test` targets depend on both `cross_locs_detector.hpp`
  and debug behavior; `cxx_std_17` (or 14) should be declared explicitly.

### 4.6 The `test` target is not a test suite
`nonogram_detector_test` is an interactive trackbar tool (`WindowTrackbarDetector`)
for parameter tuning, not an automated unit test. That's fine as an experiment
harness, but there are **no automated tests**. Recommended to add a real
`nonogram_detector_ut` with a handful of synthetic grids, assert on
`cross_locs_main` dimensions/positions, and cover the "not found" branch.

---

## 5. Windows / portability

- Hardcoded `C:\Users\klimenkov\...` absolute paths in the application and test
  `main.cpp`. Accept the image path as `argv[1]`, or use a data directory relative
  to the repo.
- `.gitignore` mentions `.vs` and `CMakeSettings.json` (Windows/VS artifacts).
- This repo was cloned to a Linux box; a `data/` dir with sample images would make
  it runnable cross-platform.

---

## 6. Recommendations (prioritized)

### P0 — Do now (correctness/blockers)
1. Remove/gate the blocking `imshow`/`waitKey(0)` inside `detect()`.
2. Delete `grid_detector.hpp/.cpp` (duplicate `CrossLocsDetector`) or rename and
   wire in properly.
3. Guard against empty input image and `scale == 0`.
4. Remove the unused `augment` `image_resized` parameter.

### P1 — Do soon (quality/clarity)
5. Replace the 4-tuple with a `Detection` struct.
6. Bump CMake minimum and link only needed OpenCV modules; declare C++ standard.
7. Accept input path from `argv` instead of hardcoded Windows paths.
8. Delete dead debug code and commented-out blocks.

### P2 — Do when performance matters (speed)
9. Shrink per-node search ROI and switch to separable kernels or `matchTemplate`.
10. Replace the cell-size scan with a 1-D projection/periodicity estimate.
11. Extrapolate the bulk of the grid analytically and verify a sparse subset
    (reduces `O(N²)` → `O(N)` convolutions).
12. Parallelize the BFS frontier when grid sizes grow.

### P3 — Future
13. Add a real automated unit test target with synthetic grids.
14. Add a `Detection`/`Grid` abstraction so the pipeline is testable in isolation.
15. Decode cells (empty/filled/digit) from the warped 20×20 patches, which is the
    natural next stage after `get_cell_warped_images_vector`.

---

## 7. Runtime verification (build + run, 2026-09-02)

Toolchain: Ubuntu 24.04, g++ 13.3, CMake 3.28, OpenCV 4.6.0 (Ubuntu `libopencv-dev`).

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # configured cleanly
cmake --build build                              # all 3 targets built, no errors
```

All three targets (`libnonogram_detector.a`, `nonogram_detector_application`,
`nonogram_detector_test`) build and link. This also confirms **`grid_detector.cpp`
is not compiled** (absent from the library build), backing the §2.2 finding.

A headless harness was built against the library and run on synthetic nonogram
grid images. `DISPLAY` exists but the active `imshow`/`waitKey(0)` in `detect()`
blocks forever, so the review's P0-1 gate (below) was required for automation.

### Confirmed bug A — crash on empty grid (Critical)
Reproduced via gdb; backtrace:

```
cv::operator/(cv::Mat const&, double)
CrossLocsDetector::detect(cv::Mat const&)
```

When `find_cell_side_length_cell_loc` succeeds but `get_cross_locs_main_mat`
returns an **empty matrix** (flood-fill finds no crosses), `detect` executes
`cross_locs_main_mat / scale` on the empty matrix → OpenCV throws
`(-5:Bad argument) Matrix operand is an empty matrix` → `terminate`/abort.
This is the §2.5 concern, confirmed as a hard crash. A guard was added: if
`cross_locs_main_mat.empty()` return the "not-found" tuple; the same guard was
added for `top`/`left`. After the fix the path degrades gracefully (`found=0`)
instead of aborting.

### Confirmed bug B — cell-size search window vs. resize default (High)
`find_cell_side_length_cell_loc` only probes side lengths `5..50` (the default
params). `detect` resizes the image so its longest side is `resize_width_height_max`
(default **1200** in `application/main.cpp`). On a typical photo this *upscales*
cells well past 50 px, so detection can never succeed:

- Clean grid, 40 px cells, resized to `max=1200` (scale≈2.3 → cells ≈ 92 px) →
  `found=0` (out of range).
- Same image at `max=400`/`600` (cells land ≈ 21/37 px, in range) → `Cell is
  detected`, `cell_side_length: 21/37`.

So the shipped default parameters are internally inconsistent for common input
scales. Lower `resize_width_height_max` or widen the search window for larger
photos.

### Confirmed — strict similarity + sub-pixel sensitivity (Medium)
The 0.9 similarity threshold on the raw `filter2D` correlation is very strict.
Reproducing the masks in Python and scoring true grid intersections on clean
synthetic grids gave correlations around **-2** (far below 0.9) whenever line
pixels landed in the `-1` corners of the masks (e.g. after `INTER_LINEAR`
downscale anti-aliasing). The pipeline is tuned to specific real-photo morphology
and is fragile to thinning/blurring of grid lines; this motivates the P2-9/10/11
recommendations (smaller ROI, `matchTemplate`, analytic extrapolation).

### Timing
Full `detect()` calls on 512–800 px synthetic inputs in this environment:

- Failed/empty path: **~40 ms**
- Successful full main-grid detection (cells in range): **~60 ms**
  (e.g. `nonogram_40_2.png`, `max=600` → `found=1`, main grid 12×14)

A representative (or larger, `N×N` with big `N`) real nonogram photo is needed for
a production-scale benchmark; none is present in the repo. Add sample images under
`data/` to enable reproducible benchmarks and regression tests.

### Changes applied during verification (P0)
1. `cross_locs_detector.cpp`: gated the debug `imshow`/`waitKey(0)` behind
   `if (std::getenv("NG_DEBUG_IMSHOW"))` (added `#include <cstdlib>`) — enables
   headless/CI use.
2. `cross_locs_detector.cpp`: guards in `detect()` to avoid dividing empty
   matrices and aborting.

These are minimal, verified fixes for confirmed P0 blockers; everything else in
§6 remains as recommendations.
