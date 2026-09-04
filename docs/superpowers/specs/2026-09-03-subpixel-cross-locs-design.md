# Subpixel Cross Location Refinement — Design

## Problem

Grid intersections ("cross locations") are detected by convolving the
thresholded binary image with a cross-shaped kernel and taking the integer
pixel of the peak response via `cv::minMaxLoc`. This integer peak is biased
toward a particular pixel of the intersection rather than the true geometric
center. The refined center is needed: accurate intersection coordinates
directly improve the quality of warped clue cells (hence digit recognition),
the grid geometry recorded in the `.non` file, and solution overlay rendering.

An additional source of error: positions are stored as integer `cv::Point` in
`CV_32SC2` matrices, so the final `/ scale` step (mapping resized-image coords
back to original-image coords) truncates fractional pixels.

## Goals / Requirements

- Produce subpixel (float) cross location coordinates for the main grid, top
  clue strip, and left clue strip.
- Refine each intersection to the best estimate of its true center using a
  **paraboloid (quadratic) fit** on the existing convolution response — cheap,
  reuses the `filter2D` output already computed by `find_kernel_loc`.
- Store positions as `cv::Point2f` in `CV_32FC2` matrices so fractional-pixel
  coordinates survive the `/ scale` mapping and downstream consumption.
- Preserve the existing detection pipeline, dimensions, and success
  behavior — only the coordinate precision changes.
- Keep all consumers (cell warping, overlay rendering, `.non` export, drawing)
  working with the new float representation, exporting float coordinates.

## Non-Goals

- No geometric line-centerline intersection refinement. The paraboloid fit on
  the convolution response is chosen instead (cheap, reuses `filter2D` output).
- No change to how missing intersections are augmented/interpolated beyond the
  type change.
- No change to the detector's ROI sizes or BFS expansion logic.

## Architecture

### Data structure (`detection.hpp`)

`Detection.main`, `.top`, `.left` change from `CV_32SC2` (holding `cv::Point`)
to `CV_32FC2` (holding `cv::Point2f`). The `found` flag is unchanged.

### Subpixel peak refinement (`image_operations.cpp`)

`find_kernel_loc` currently returns `std::pair<bool, cv::Point>`. After
`cv::minMaxLoc` finds the integer peak, sample the (already normalized)
filtered response at the peak and its four orthogonal neighbors and fit a
1-D parabola along each axis:

```
x_offset = (f(x-1) - f(x+1)) / (2 * (f(x-1) - 2*f(x) + f(x+1)))
y_offset = (f(y-1) - f(y+1)) / (2 * (f(y-1) - 2*f(y) + f(y+1)))
```

The refined location is `(peak.x + x_offset, peak.y + y_offset)`. Guard against
degenerate cases:

- If any neighbor is out of bounds (peak at ROI border), fall back to the
  integer peak.
- If the second difference is ~0 (flat response), fall back to the integer peak.

Both `find_kernel_loc` overloads change return type to `std::pair<bool,
cv::Point2f>`.

### Internal maps (`cross_locs_detector.cpp`)

The BFS uses `std::map<cv::Point, cv::Point, PointCompare>` keyed by grid-index
(second `cv::Point` is the pixel coordinate). The grid-index key stays
`cv::Point`; the value (pixel coordinate) becomes `cv::Point2f`:

- `cross_locs_init_map`, `cross_locs_map` → `std::map<cv::Point, cv::Point2f,
  PointCompare>`.
- Predicted neighbor inits (`cross_loc + cross_loc_deltas[i]`) become
  `cv::Point2f`.
- `get_roi(cross_loc_init, roi_size)` still needs an integer `cv::Point`
  center — round the `cv::Point2f` init to `cv::Point` for ROI selection only
  (the stored value stays the float refined location).

- `convert_to_mat` writes `cv::Point2f` into a `CV_32FC2` mat, default
  `cv::Point2f(-1, -1)` (with `.empty()` unchanged, and the sentinel value
  checks updated to `cv::Point2f(-1, -1)`).

- `augment` interpolates `cv::Point2f` values (same averaging logic).

### Seed / matrix helpers

- `get_cross_locs_main_mat`, `get_cross_locs_top_mat`,
  `get_cross_locs_left_mat` accept `cv::Point2f` seeds and produce
  `CV_32FC2` matrices.
- All `find_kernel_loc` call sites (square-mask seed, BFS, top/left) now
  receive and return `cv::Point2f`. The initial square-mask seed's returned
  `cell_loc` in `find_cell_side_length_cell_loc` becomes `cv::Point2f` too
  (tuple type updated accordingly).

### `/ scale` mapping (`detect()`)

`detection.main = cross_locs_main_mat / scale` now operates on `CV_32FC2`
float values — no truncation. The same applies to `.top` and `.left`.

## Consumers

### Cell warping (`image_operations.cpp`)

`get_cell_warped_images_vector` currently reads `cross_locs.at<cv::Point>` and
casts to `cv::Point2f`. Change to `cross_locs.at<cv::Point2f>` directly. The
`cv::getPerspectiveTransform` already takes `cv::Point2f` vectors; no other
change.

### Overlay rendering (`main.cpp`)

`render_nonogram_overlay`:
- `cell_poly` becomes `std::vector<cv::Point2f>` (from `.at<cv::Point2f>`).
- `cv::fillConvexPoly` accepts `cv::Point2f` when a scoped `cv::Rect` output is
  not required — use `cv::InputArray` of `Point2f`.
- Grid lines: `cv::line` accepts `cv::Point2f` (integer-rounded internally).

### `.non` export (`main.cpp`, `write_non_file`)

`#cells:` section serializes float coordinates, e.g. `123.45,67.89`, with
moderate precision (default `std::ostream` formatting is fine; use enough
decimals to preserve subpixel — e.g. two decimal places).

### Drawing (`CrossLocsDetector::draw`)

`cv::circle` accepts `cv::Point2f` center. Change the iteration
`cross_locs_mat.begin<cv::Point2f>()`.

### Tests

- `main.cpp` grid cases verify dimensions only — expected to pass unchanged.
- `non_file_test.cpp`: `parse_non_custom_block` parses float `x,y` tokens into
  `cv::Point2f`; comparisons updated. Empty-detection case unchanged.
- New unit test for the paraboloid fit: synthetic thresholded image with a
  cross whose true center sits at a half-pixel offset; verify the refined
  `Point2f` recovers it within a small tolerance (e.g. 0.15 px), and that the
  coarse integer peak is displaced by ~0.5 px from the true center.

## Error Handling

- Degenerate paraboloid (flat or out-of-bounds) falls back to the integer
  peak without affecting detection success.
- `.empty()` guards and sentinel `cv::Point2f(-1, -1)` checks are preserved
  throughout.

## Testing Strategy

- Unit test for paraboloid fit accuracy (see Tests).
- Full existing UI test-suite run (`nonogram_detector_ut`) must pass.
- Manual verification on a real photo: exported `.non` shows non-integer
  coordinates; overlay grid aligns with line intersections.

## Open Questions

- None.

---

## Addendum (2026-09-04): ink-centroid second-stage refinement

After shipping the paraboloid fit, residual bias remained on real photos
(plateau "first-max" corner placement, up to ~1 px on clue-strip and bold
boundary lines). A second refinement stage was added:

- `refine_cross_locs_ink(gray, cross_locs, window_radius)` (public, in
  `image_operations`): per cross, per axis, build the darkness-weighted ink
  profile over a 3 px band around the current location (ink outside the band —
  e.g. a clue digit — cannot contaminate it), subtract the profile's baseline
  (min over the window — removes the crossing line's uniform contribution),
  and take the centroid; anti-aliased line edges make it subpixel-accurate.
- Guards: `(-1,-1)` sentinels and border-clipped windows are skipped; a band
  with too little ink (empty paper, extrapolated padding) keeps the location.
  The band width itself bounds the maximum move (3 px per axis).
- Iteration (post-shipping improvement): the band centroid is computed
  iteratively, re-centring the band on the refined location each pass until
  both axes move < 0.05 px (max 4 passes). A bold line whose location lands
  near the band edge is truncated and under-corrected by one pass (~half a line
  width short); 2-3 passes recover the exact center (unit test: 6 px cross
  2.5 px off center → converges to 30.0 vs 29.5 single-pass).
- Integration: in `detect()`, applied to the main mat right after the BFS
  (before `/scale` and before the top/left searches, so their seeds inherit
  refined positions), then to the top and left mats.
- Verified: unit tests (analytic anti-aliased cross recovered within 0.05 px;
  empty-window and non-cross-ink guards), grid dims unchanged, and ASCII ink
  dumps on real photos showing the detected positions on the actual line
  centers (bold 3 px boundary line centered to 0.01 px).

