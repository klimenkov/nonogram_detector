# Nonogram Detector — Architecture & Design

## 1. Overview

`nonogram_detector` is a C++ application built on OpenCV that identifies the grid
structure of a nonogram (picross) puzzle from a photo of the puzzle. Given an input
image it locates:

- the **main cells region** (the grid where the puzzle is solved),
- the **top clues region** (column clues),
- the **left clues region** (row clues),

and returns, for each of these regions, the pixel position of every grid
intersection (a "cross location", abbreviated `cross_loc`).

The project is early-stage and research-oriented: much of it is a working prototype
with debugging instrumentation still present.

## 2. Build system

CMake with four subprojects:

| Subproject | Type | Purpose |
|---|---|---|
| `nonogram_detector` | static library | Core algorithm (`ng` namespace) |
| `nonogram_solver` | static library | Adapter wrapping the third-party picross solver + dedicated clue-correction step |
| `nonogram_detector_application` | executable | Driver program (argument-driven, headless) |
| `nonogram_detector_ut` | executable | Automated synthetic-grid + solver unit tests |

- The library links OpenCV (`find_package(OpenCV REQUIRED)`,
  `core`/`imgproc`/`imgcodecs`/`dnn`) and exposes its `include/` dir publicly.
- The solver module links the third-party **picross** library, pulled in at
  configure time via `FetchContent` (pinned commit, `GIT_SHALLOW`; its bundled
  app/CLI/tests/examples are disabled via `PICROSS_BUILD_* OFF`, and `-Werror`
  is stripped from the picross target to survive modern GCC). The fetch can be
  disabled with `-DNG_ENABLE_SOLVER=OFF`, which also removes `nonogram_solver`
  and the solver wiring from the application/tests.
- There is no interactive/display component: the library and application are
  headless and never open a window or wait for user input.

## 3. Module layout & dependencies

```
nonogram_detector/
  include/
    image_operations.hpp      Free functions for resize/threshold/matching/warping
    masks.hpp                 Kernel-template generators (square, cross)
    point_compare.hpp         cv::Point strict-weak-ordering comparator
    cross_locs_detector.hpp   Main algorithm class (CrossLocsDetector)
    detection.hpp             ng::Detection result struct (found/main/top/left)
    digit_recognizer.hpp      ONNX-digit classifier (cv::dnn) for clue cells
    decode.hpp                Decodes clue strips into per-cell digit grids
    grid_smooth_fit.hpp       Polynomial surface grid-fitting models
  src/
    image_operations.cpp
    masks.cpp
    point_compare.cpp
    cross_locs_detector.cpp
    digit_recognizer.cpp
    decode.cpp
    grid_smooth_fit.cpp
  models/
    digits.onnx               Bundled MNIST-digit CNN (ONNX Model Zoo)
nonogram_solver/
  include/solver.hpp          Declares ng::ClueConstraints / ng::SolveResult / solve_nonogram
  src/solver.cpp              Adapter over picross (the only TU linking picross)
  include/clue_corrector.hpp  Declares ng::ClueCell/DecodedCells + corrector/report API
  src/clue_corrector.cpp      Consistency analysis + correction layer (no picross dep)
nonogram_detector_application/main.cpp
```

Dependency direction is one-way: `application` links the detector library and
the solver library; library modules depend only on OpenCV
(core/imgproc/imgcodecs/dnn), each other, and (for `nonogram_solver`) the
vendored `picross`.

### 3.0 `DigitRecognizer` & `decode`

`ng::DigitRecognizer` (`digit_recognizer.hpp`) loads the bundled
`models/digits.onnx` (a ~26 KB MNIST convolutional network) through
`cv::dnn::readNetFromONNX` and classifies the digit inside a warped clue cell:

- `prepare_input` geometrically normalizes a cell: grayscale, Otsu plus-polarity
  inversion, an inward crop past the surrounding grid frame, crop to the largest
  foreground contour, aspect-preserving rescale onto a 28×28 white-on-black
  canvas, then MNIST whitening `(x/255 − 0.1307)/0.3081` producing a
  `1×1×28×28` blob.
- `recognize` runs the forward pass and returns the argmax digit (the softmax
  probability is exposed via `recognize_ex` for confidence gating).
- `decode.hpp::decode_clues` reuses `get_cell_warped_images_vector` on the
  `top` and `left` regions of a `Detection` and fills an `ng::ClueGrid`
  (row-major `vector<vector<int>>`, `-1` for empty cells).



### 3.0a `solve_nonogram` adapter (`nonogram_solver`)

`ng::solve_nonogram` (`nonogram_solver`) is a small stateless adapter over the
third-party **picross** solver (`pierre-dejoue/picross-solver`, vendored via
`FetchContent`). It is the only translation unit that links picross; the rest of
the codebase never does, keeping the dependency isolated behind a stable, minimal
API.

- `ng::ClueConstraints` holds decoded clues: `rows` (one clue per grid row, from
  the `left` strip) and `cols` (one clue per grid column, from the `top` strip
  transposed). Values `<= 0` inside a line are treated as absent.
- `solve_nonogram` builds a `picross::InputGrid`, **validates it with
  `picross::check_input_grid`** (rejecting zero dimensions, a clue whose
  `min_line_size` exceeds the grid axis, or mismatched row/column filled-tile
  totals) and returns `ng::SolveResult { solved, line_solvable, solution_count,
  solution, message }` rather than letting the library throw on bad input. The
  first solution (or partial grid when not line-solvable) is surfaced as a
  `vector<vector<int>>` of 0/1.
- The whole solver unit sits behind the `NG_ENABLE_SOLVER` build flag, so the
  network fetch and picross dependency can be turned off entirely.

`ng::PointCompare` provides a strict weak ordering over `cv::Point`
(`p1.x < p2.x`, ties broken by `y`). It is used so `cv::Point` can be a key in
`std::map` / `std::set`.

Note: `cv::Mat` of type `CV_32FC2` stores each element as a 2×float tuple, which
`cv::Mat::at<cv::Point2f>` treats as a `cv::Point2f`. The code relies on this to
store subpixel grid cross locations in a dense matrix.

### 3.0b `ClueCorrector` consistency layer (`nonogram_solver`)

`ng::clue_corrector` (`nonogram_solver/include/clue_corrector.hpp`,
`src/clue_corrector.cpp`) is a dependency-light module (no picross) that sits
between `decode` and `solve`. It exists because the generic MNIST digit
classifier misreads printed-font glyphs confidently and inconsistently, so the
decoded clue strips are not reliable enough to hand straight to the solver.

- `ng::DecodedCells` wraps the raw clue-cell grids (`rows` = left strip, `cols`
  = top strip transposed) as `ng::ClueCell { digit, confidence }`.
- `ng::group_clue_line` turns a line of cells into clue numbers: consecutive
  non-empty cells concatenate into multi-digit clues, and an empty cell
  (`digit < 0`) separates clues. A resulting value of 0 (e.g. a lone `0`) is
  dropped.
- `ng::analyze_consistency` produces an `ng::ConsistencyReport`: the row vs
  column filled-tile totals and which lines overflow their grid axis. This is
  what converts the solver's opaque `invalid constraints` status into an
  actionable, printable diagnostic.
- `ng::correct_clues` applies two bounded, deterministic repairs before solving:
  1. **Sanitizer** — drops standalone `0` cells (border/stray reads) while
     preserving zeros inside multi-digit clues (`10`, `20`).
  2. **Grid-fit repair** — when a run of adjacent non-empty cells concatenates
     into a clue number larger than the grid axis (impossible for a real clue,
     caused by the decoder not emitting gap cells between distinct clues) and
     every digit is 1–9, splits the run into single-digit clues by inserting gap
     cells — kept only if that makes the line fit.

The corrected cells still may not equal the *true* puzzle: the repairs are purely
structural (they restore *a* consistent puzzle, not necessarily the original
one). A row like the photo's row-14 misread `9 2 7 7 1 1 2` cannot be recovered
by structure alone because the misread digits are high-confidence and inflate the
line past the grid even after splitting — the corrector therefore reports it as
overflow rather than guess. This ceiling is accepted; the report is the contract.


### 3.2 `masks`

Small integer kernels used as convolution templates. Each returns the mask plus a
"perimeter" value used to normalize the correlation score.

- `get_mask_square(side)` — border = 1, interior = -1. Used to detect a single
  cell when the side length is unknown (scanning candidate sizes).
- `get_mask_cross(length, margin)` — a cross (plus-shape): 1 on the central
  cross arm, 0 within a margin band, -1 in the corners. Used to detect cell
  intersections ("crosses") while tolerating the line width.
- `get_mask_cross(length)` — degenerate variant: 1 on the cross arms, 0
  elsewhere (no -1 corners). Used for the top/left clue regions.

### 3.3 `image_operations`

Free functions:

- `resize(image, max, interp)` — scales so the longest side equals `max`;
  returns the new image and the scale factor (used later to rescale results back).
- `threshold(image_gray, block, c)` — `cv::adaptiveThreshold`
  (`ADAPTIVE_THRESH_MEAN_C`, `THRESH_BINARY_INV`), producing a `CV_8U` image of
  0/1 values.
- `get_roi(center, size)` — a `cv::Rect` centered on a point.
- `is_inside(rect, sub_rect)` — containment check.
- `refine_peak_loc(image_filtered, peak)` — fits a 1-D parabola along x and y
  through the integer peak and its neighbors in the filtered response,
  returning a subpixel `cv::Point2f`; falls back to the integer peak when the
  peak sits at the response border or the response is flat.
- `refine_cross_locs_ink(image_gray, cross_locs, window_radius)` — second
  refinement stage: snaps every cross location to the geometric center of the
  drawn line intersection. Per axis it builds the darkness-weighted ink
  profile over a band around the location (so contaminating ink further out,
  e.g. a clue digit, cannot pull the centroid), subtracts the profile's
  baseline (the crossing line's uniform contribution), and takes the centroid,
  which anti-aliased line edges make subpixel-accurate. The step iterates: each
  pass re-centres the band on the refined location, so a bold line that the
  band truncated by landing near its edge on the first pass is fully visible on
  the next and the centroid converges to the true center (max 4 passes, stops
  when both axes move < 0.05 px). Locations are kept when the window clips the
  image border or the band holds too little ink (empty paper, extrapolated
  padding).
- `find_kernel_loc` (x2) — convolves a 0/1 image with a kernel via
  `cv::filter2D`, normalizes by the mask perimeter, and reports the peak via
  `minMaxLoc`, refined to subpixel precision with `refine_peak_loc`. A match is
  successful when the normalized peak exceeds `similarity_ratio_min`. The ROI
  overload clips to the image and returns coordinates offset back into the
  full image.
- `get_cell_warped_images_vector(image, cross_locs)` — from a `cross_locs`
  matrix, builds each cell's four corner points, computes a perspective
  transform, and warps every cell to a fixed 20×20 patch. Returned as a 2D
  vector of `cv::Mat`.

### 3.4 `CrossLocsDetector` (active algorithm)

The class encapsulates the detection pipeline. Configuration is injected through
the constructor and stored as constants:

- `resize_width_height_max`, `threshold_block_size`, `threshold_c`
- `find_cell_side_length_min/max`, `similarity_ratio_min`

Public API:

- `detect(image)` → `std::tuple<bool, cv::Mat, cv::Mat, cv::Mat>` where the bool
  is a found-flag and the mats are `cross_locs_main`, `cross_locs_top`,
  `cross_locs_left`.
- `static draw(image, cross_locs_mat, radius, color)` — overlays the cross
  locations as filled circles on a clone.

Private helpers are largely `static` and operate on the shared intermediate
representations (see §5).

## 4. Detection pipeline (data flow)

The core is `CrossLocsDetector::detect`:

1. **Preprocess** — `resize` to `resize_width_height_max`, convert to gray
   (`COLOR_BGR2GRAY`), `threshold` to a 0/1 image (`image_thresholded`).
   (A debug `imshow`/`waitKey` block is left active here.)
2. **Find cell geometry** — around the image center take a 150×150 ROI and call
   `find_cell_side_length_cell_loc`: for each side length in `[min, max]`,
   build `get_mask_square` and test `find_kernel_loc`; the first match yields the
   cell side length and the pixel location of the first cell corner.
3. **Grow the main grid** — starting from the detected first cross, flood-fill
   (`get_cross_locs_map`) outward. For each grid index a predicted cross location
   is computed from its already-found neighbor (`cross_loc + delta`), then
   re-located precisely with `find_kernel_loc` using the cross `mask_cross`
   within a sized ROI (each match refined to subpixel precision). The result
   is a sparse `std::map<cv::Point, cv::Point2f>` keyed by grid indices. The
   main map uses
   `mask_cross(cell_side_length*1.5 odd, line_width/2)` (margin version) and
   4-directional deltas.
4. **Derive clue regions** — `get_cross_locs_top_mat` / `get_cross_locs_left_mat`
   seed from the top row / left column of the main grid, then run the same
   flood-fill `get_cross_locs_map` with the plain `get_mask_cross` and
   3-directional deltas, growing the clue strips.
5. **To dense matrix** — `convert_to_mat` maps the sparse index→point map onto a
   dense `CV_32FC2` matrix (size = bounding box + 1), filling gaps with
   `cv::Point2f(-1, -1)`.
6. **Pad & augment** — each region is padded by one row/column (so the perimeter
   crossing exists) and `augment` fills any `cv::Point2f(-1, -1)` `cross_loc` by
   linear extrapolation from already-known neighbors (`cross_loc +
   cell_side_length * direction`, averaging when multiple). This fills
   bases/missing cells.
7. **Rescale** — `scale_cross_locs_mat` divides each location by `scale` to
   translate back to the original full-resolution image coordinate space,
   preserving `cv::Point2f(-1, -1)` sentinels; the float coordinates survive
   the mapping without truncation.
8. **Return** — the three `cross_locs` matrices plus the found-flag.

## 5. Data representation: `cross_locs`

The central concept. Each `cross_locs_*` is a `CV_32FC2` `cv::Mat` where element
`(x, y)` stores the subpixel position (`cv::Point2f`) of the grid intersection
at column `x`, row `y` of that region, located in two refinement stages: a
paraboloid fit on the `filter2D` peak inside `find_kernel_loc`
(`refine_peak_loc`), then the ink centroid of the grayscale line intersection
(`refine_cross_locs_ink`, run in `detect()` before scaling and before the
top/left searches so their seeds inherit the refined positions). A value of
`cv::Point2f(-1, -1)` means *not located* (before augmentation) / *empty
padding* (after).

The three matrices share an identical element type and differ only by region:

- `cross_locs_main` — the puzzle grid intersections.
- `cross_locs_top` — the column-clues region.
- `cross_locs_left` — the row-clues region.

The main region is grown first and seeds the clue regions' initial positions.

## 6. Application driver (`nonogram_detector_application/main.cpp`)

A procedural, headless driver:

1. Reads the image path from the first argument (`argv[1]`); optional second
   argument sets the resize max (default 1200).
2. Constructs `ng::CrossLocsDetector(resize_max, 15, 10.0, 5, 50, 0.9)`.
3. Runs `detect`, prints the found-flag, and draws the main/top/left results as
   blue/green/red circles.
 4. Decodes the clue strips via `ng::decode_clues` and prints `top clues:` /
    `left clues:` headlessly.
 5. When `NG_ENABLE_SOLVER` is defined, converts the decoded strips into
    `ng::DecodedCells`, runs `ng::correct_clues` and prints the resulting
    `ConsistencyReport` (tile totals + overflowing lines), re-groups the
    corrected cells into `ng::ClueConstraints` (rows = `left` strip, cols = `top`
    strip transposed), calls `ng::solve_nonogram`, prints the solver message /
    solution count, and — when solved — renders the first solution grid as ASCII
    (`#` filled, `.` empty).
 6. When the `NG_SAVE_OUTPUT` environment variable is set, saves the overlay to
    `grid.png`. No window is ever opened; it never waits for user input.

## 4b. Solve phase (after detection & decode)

The detect → decode → solve chain is split across two libraries: `detect`
(locating the grid) and `decode_clues` (reading digits) live in
`nonogram_detector`; the final solve step lives in `nonogram_solver`. The
application is the only place that composes them. The consistency layer of
`correct_clues` runs between decode and solve (see §3.0b), folding gross
decode artifacts (concatenated over-wide runs, stray `0`s) into a consistent-enough
constraint set, and printing a report that names *why* a puzzle still cannot be
solved. A malformed decode (spurious digits from the generic MNIST model) is then
rejected with a clear "invalid constraints" status by `check_input_grid` rather
than aborting, so the pipeline degrades gracefully on noisy real photos.

## 7. Experiment driver (removed)

An earlier interactive trackbar experiment tool (`nonogram_detector_test`, with
`WindowTrackbarDetector`) was removed from the build because it existed purely
for interactive parameter tuning via `imshow`/`waitKey`/`createTrackbar` and was
not an automated test. Parameter behavior is now validated by the synthetic
`nonogram_detector_ut` suite.

## 8. Dead / legacy code

- The legacy duplicate class `grid_detector.hpp` and `src/grid_detector.cpp` has been removed.
- Numerous commented-out debug blocks (imshow/draw/print) throughout
  `cross_locs_detector.cpp`.

## 9. Design observations & risks

- **Headless** — `detect` no longer opens any window or waits for user input;
  the previously blocking `imshow`/`waitKey(0)` block was removed. The detector
  runs unattended and is testable in a harness.
- **Hardcoded environment** — application/test use absolute Windows paths and no
  CLI/argument input; the top-level `cmake_minimum_required` is 2.8 (very old).
- **Tuple return instead of a result type** — `detect` returns a 4-tuple; a small
  struct would be self-documenting and less error-prone.
- **Representation coupling** — correctness relies on `CV_32FC2` ↔
  `cv::Point2f` aliasing and on the invariant that `indices_init[i]`
  corresponds to `cross_locs_init[i]` (documented in the code comment but not
  enforced).
- **BFS termination** — `get_cross_locs_map` can only expand to grid indices
  reachable via known deltas; if a neighbor has no valid ROI or no mask match the
  branch stops (which is why `augment` later fills gaps by extrapolation).
- **Augmentation logic** — `augment` currently uses only the first neighbor for
  extrapolation; a second neighbor (`indices_neighbor_2`) exists in the code but is
  commented out.
- **Unused resize parameter in `augment`** — `augment(image_resized, ...)`
  receives an image that is never used (its single use passes `cv::Mat()`); the
  parameter is effectively dead.
