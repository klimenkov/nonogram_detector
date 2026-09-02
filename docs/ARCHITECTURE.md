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

CMake with three subprojects:

| Subproject | Type | Purpose |
|---|---|---|
| `nonogram_detector` | static library | Core algorithm (`ng` namespace) |
| `nonogram_detector_application` | executable | Driver program (single hardcoded image) |
| `nonogram_detector_test` | executable | Interactive trackbar experiment tool |

- `cmake_minimum_required(VERSION 2.8)` and `.gitignore` entries (`.vs`,
  `CMakeSettings.json`) indicate a Windows/Visual Studio origin.
- The library links OpenCV (`find_package(OpenCV REQUIRED)`) and exposes its
  `include/` dir publicly.

## 3. Module layout & dependencies

```
nonogram_detector/
  include/
    image_operations.hpp      Free functions for resize/threshold/matching/warping
    masks.hpp                 Kernel-template generators (square, cross)
    point_compare.hpp         cv::Point strict-weak-ordering comparator
    cross_locs_detector.hpp   Main algorithm class (CrossLocsDetector)
    grid_detector.hpp         LEGACY duplicate of cross_locs_detector (see §8)
  src/
    image_operations.cpp
    masks.cpp
    point_compare.cpp
    cross_locs_detector.cpp
    grid_detector.cpp         LEGACY, not compiled
nonogram_detector_application/main.cpp
nonogram_detector_test/main.cpp
```

Dependency direction is one-way: `application`/`test` link the library; library
modules depend only on OpenCV and each other.

### 3.1 `point_compare`

`ng::PointCompare` provides a strict weak ordering over `cv::Point`
(`p1.x < p2.x`, ties broken by `y`). It is used so `cv::Point` can be a key in
`std::map` / `std::set`.

Note: `cv::Mat` of type `CV_32SC2` stores each element as a 2×int32 tuple, which
`cv::Mat::at<cv::Point>` treats as a `cv::Point`. The code relies on this to store
grid cross locations in a dense matrix.

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
- `find_kernel_loc` (x2) — convolves a 0/1 image with a kernel via
  `cv::filter2D`, normalizes by the mask perimeter, and reports the peak via
  `minMaxLoc`. A match is successful when the normalized peak exceeds
  `similarity_ratio_min`. The ROI overload clips to the image and returns
  coordinates offset back into the full image.
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
   within a sized ROI. The result is a sparse `std::map<cv::Point, cv::Point>`
   keyed by grid indices. The main map uses
   `mask_cross(cell_side_length*1.5 odd, line_width/2)` (margin version) and
   4-directional deltas.
4. **Derive clue regions** — `get_cross_locs_top_mat` / `get_cross_locs_left_mat`
   seed from the top row / left column of the main grid, then run the same
   flood-fill `get_cross_locs_map` with the plain `get_mask_cross` and
   3-directional deltas, growing the clue strips.
5. **To dense matrix** — `convert_to_mat` maps the sparse index→point map onto a
   dense `CV_32SC2` matrix (size = bounding box + 1), filling gaps with
   `(-1, -1)`.
6. **Pad & augment** — each region is padded by one row/column (so the perimeter
   crossing exists) and `augment` fills any `(-1, -1)` `cross_loc` by linear
   extrapolation from already-known neighbors (`cross_loc + cell_side_length *
   direction`, averaging when multiple). This fills bases/missing cells.
7. **Rescale** — results are divided by `scale` to translate back to the
   original full-resolution image coordinate space.
8. **Return** — the three `cross_locs` matrices plus the found-flag.

## 5. Data representation: `cross_locs`

The central concept. Each `cross_locs_*` is a `CV_32SC2` `cv::Mat` where element
`(x, y)` stores the pixel position of the grid intersection at column `x`, row `y`
of that region. A value of `cv::Point(-1, -1)` means *not located* (before
augmentation) / *empty padding* (after).

The three matrices share an identical element type and differ only by region:

- `cross_locs_main` — the puzzle grid intersections.
- `cross_locs_top` — the column-clues region.
- `cross_locs_left` — the row-clues region.

The main region is grown first and seeds the clue regions' initial positions.

## 6. Application driver (`nonogram_detector_application/main.cpp`)

A procedural driver:

1. Reads a hardcoded image path (`C:\Users\klimenkov\Desktop\nonograms\nonogram.jpg`).
2. Constructs `ng::CrossLocsDetector(1200, 15, 10.0, 5, 50, 0.9)`.
3. Runs `detect`, draws the main/top/left results as blue/green/red circles.
4. Resizes the overlay for display and shows it (`imshow`/`waitKey`).
5. Also demonstrates `get_cell_warped_images_vector` on `cross_locs_left`
   (perspective warp each clue cell to 20×20).

A commented-out `save_images` / `get_cell_rois` path exists for exporting cell
images.

## 7. Experiment driver (`nonogram_detector_test/main.cpp`)

`WindowTrackbarDetector` wraps the detector behind OpenCV trackbars so the
preprocessing parameters can be tuned live:

- Resize max (500–1500), threshold block size (3–203), and the adaptive `C`
  offset (-50–50).

Moving a trackbar reconstructs the detector and re-runs `detect`, redrawing the
overlay. This is the primary parameter-tuning playground and is *not* an
automated unit test suite.

## 8. Dead / legacy code

- `nonogram_detector/include/grid_detector.hpp` and `src/grid_detector.cpp` define
  a second class **also named `CrossLocsDetector`** (an earlier variant with a
  slightly different `detect` signature that omits the bool and has no empty-matrix
  guards). It is **not** listed in the library's CMake `SOURCES`, so it is never
  compiled or linked. Treat it as legacy/dead code; it can only ever be referred to
  via a mismatched translation unit and is not part of the active build.
- Numerous commented-out debug blocks (imshow/draw/print) throughout
  `cross_locs_detector.cpp`.

## 9. Design observations & risks

- **Active debug instrumentation** — `detect` contains a live
  `imshow`/`waitKey(0)` that blocks until a key is pressed, plus several
  `std::cout` diagnostics. These must be gated (e.g. behind a debug flag) before
  the detector can run unattended.
- **Hardcoded environment** — application/test use absolute Windows paths and no
  CLI/argument input; the top-level `cmake_minimum_required` is 2.8 (very old).
- **Tuple return instead of a result type** — `detect` returns a 4-tuple; a small
  struct would be self-documenting and less error-prone.
- **Representation coupling** — correctness relies on `CV_32SC2` ↔ `cv::Point`
  aliasing and on the invariant that `indices_init[i]` corresponds to
  `cross_locs_init[i]` (documented in the code comment but not enforced).
- **BFS termination** — `get_cross_locs_map` can only expand to grid indices
  reachable via known deltas; if a neighbor has no valid ROI or no mask match the
  branch stops (which is why `augment` later fills gaps by extrapolation).
- **Augmentation logic** — `augment` currently uses only the first neighbor for
  extrapolation; a second neighbor (`indices_neighbor_2`) exists in the code but is
  commented out.
- **Unused resize parameter in `augment`** — `augment(image_resized, ...)`
  receives an image that is never used (its single use passes `cv::Mat()`); the
  parameter is effectively dead.
