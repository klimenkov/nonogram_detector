# Subpixel Cross Location Refinement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add subpixel (float, `cv::Point2f`) cross-location coordinates to the grid/clue-strip detection pipeline via a paraboloid fit on the convolution peak, switching all cross-location storage from integer `CV_32SC2`/`cv::Point` to float `CV_32FC2`/`cv::Point2f`, and update all consumers (cell warping, overlay rendering, `.non` export, drawing, tests).

**Architecture:** `find_kernel_loc` already computes a normalized `filter2D` response and finds its integer peak via `cv::minMaxLoc`. We extract a new helper `refine_peak_loc` that fits a 1-D parabola along x and y through the peak's neighbors to recover a subpixel `cv::Point2f` peak. All cross-location values flow as `cv::Point2f` through the BFS map, matrix conversion, augmentation, and `/ scale` mapping; the `Detection` matrices become `CV_32FC2`. Consumers read `cv::Point2f`.

**Tech Stack:** C++17, OpenCV (`cv::Mat`, `cv::filter2D`, `cv::minMaxLoc`, `cv::getPerspectiveTransform`, `cv::warpPerspective`, `cv::fillConvexPoly`, `cv::line`, `cv::circle`), CMake.

Reference spec: `docs/superpowers/specs/2026-09-03-subpixel-cross-locs-design.md`

---

## File Map

- **Modify:** `nonogram_detector/include/image_operations.hpp` — add `refine_peak_loc`, change `find_kernel_loc` return types to `cv::Point2f`.
- **Modify:** `nonogram_detector/src/image_operations.cpp` — implement `refine_peak_loc`; update both `find_kernel_loc`; update `get_cell_warped_images_vector` to read `cv::Point2f`.
- **Modify:** `nonogram_detector/include/detection.hpp` — `CV_32SC2` → `CV_32FC2`, comment update.
- **Modify:** `nonogram_detector/include/cross_locs_detector.hpp` — `Point2f` in map/tuple signatures.
- **Modify:** `nonogram_detector/src/cross_locs_detector.cpp` — `Point2f` everywhere (BFS, convert_to_mat, augment, get_cross_locs_*, detect), sentinel-preserving `/ scale`.
- **Modify:** `nonogram_detector_application/main.cpp` — overlay render + `.non` export read/write `cv::Point2f`.
- **Modify:** `nonogram_detector_ut/main.cpp` — add paraboloid-fit unit tests.
- **Modify:** `nonogram_detector_ut/non_file_test.cpp` — parse float coordinates.

Build/test commands (run from `build/`):
- Build: `cmake --build . -j$(nproc)`
- Tests: `./nonogram_detector_ut/nonogram_detector_ut`

---

### Task 1: Add standalone `refine_peak_loc` helper (public + unit-tested)

Adds the subpixel parabola-fit helper as a new public free function. `find_kernel_loc` is **not** changed yet — it still returns `cv::Point`, so the pipeline keeps compiling and passing. This task is independently green.

**Files:**
- Modify: `nonogram_detector/include/image_operations.hpp:33-50`
- Modify: `nonogram_detector/src/image_operations.cpp` (add function near line 56)
- Test: `nonogram_detector_ut/main.cpp` (add tests)

- [ ] **Step 1: Write the failing unit test for `refine_peak_loc`**

Add to `nonogram_detector_ut/main.cpp`, inside the anonymous namespace (before `int main()`), these two test functions:

```cpp
namespace
{

// A pure 2-D quadratic response peaked at a known subpixel center. Sampling it
// on the integer grid and feeding the integer peak into refine_peak_loc must
// recover the true subpixel center of each axis (a parabola fit is exact for a
// quadratic).
void test_refine_peak_loc_x()
{
    std::cout << "case: refine_peak_loc recovers subpixel center (x)\n";
    // Peak of the parabola, x0 at integer+0.4, y centered on integer row 10.
    float const x0 = 15.6f, y0 = 10.0f;
    float const a = 1.0f;
    cv::Mat resp(21, 31, CV_32F);
    for (int y = 0; y < resp.rows; ++y)
        for (int x = 0; x < resp.cols; ++x)
            resp.at<float>(y, x) = -a * ((x - x0) * (x - x0) + (y - y0) * (y - y0));

    cv::Point const int_peak(cvRound(x0), cvRound(y0));  // (16, 10)
    cv::Point2f const refined = ng::refine_peak_loc(resp, int_peak);
    if (std::fabs(refined.x - x0) > 1e-3f || std::fabs(refined.y - y0) > 1e-3f)
    {
        std::cerr << "  [FAIL] refined=" << refined << " expected ~(" << x0
                  << "," << y0 << ")\n";
        return;
    }
    std::cout << "  [ok] refined=" << refined << "\n";
}

void test_refine_peak_loc_fallback_on_boundary()
{
    std::cout << "case: refine_peak_loc falls back to int peak at border\n";
    // Peak at (0, 20) — no left neighbor; must return the integer peak.
    cv::Mat resp(21, 21, CV_32F, cv::Scalar(0));
    resp.at<float>(20, 0) = 1.0f;
    cv::Point const int_peak(0, 20);
    cv::Point2f const refined = ng::refine_peak_loc(resp, int_peak);
    if (refined != cv::Point2f(0.0f, 20.0f))
    {
        std::cerr << "  [FAIL] refined=" << refined << " expected (0,20)\n";
        return;
    }
    std::cout << "  [ok] refined=" << refined << "\n";
}

}
```

The two new test functions must be called from `int main()` in the same file. Add this call block right after the existing `estimate_cell_side_length` case (before `failures += run_digit_recognizer_tests();`):

```cpp
    {
        std::cout << "case: subpixel peak refinement\n";
        test_refine_peak_loc_x();
        test_refine_peak_loc_fallback_on_boundary();
    }
```

- [ ] **Step 2: Run the test to verify it fails to compile**

Run: `cmake --build . -j$(nproc)`
Expected: compile error — `ng::refine_peak_loc` is not declared.

- [ ] **Step 3: Declare `refine_peak_loc` in the header**

In `nonogram_detector/include/image_operations.hpp`, insert this declaration just above the first `find_kernel_loc` overload (line 32-33):

```cpp
// Fits a 1-D parabola along x and through the y-neighbors of the integer peak
// in the (single-channel, floating-point) filtered response, returning a
// subpixel peak location. Falls back to the integer peak when the peak sits at
// the response border or the response is flat.
cv::Point2f refine_peak_loc(cv::Mat const& image_filtered, cv::Point const& peak);
```

(Do not touch the `find_kernel_loc` signatures in this task.)

- [ ] **Step 4: Implement `refine_peak_loc` in the .cpp**

In `nonogram_detector/src/image_operations.cpp`, insert this function before `find_kernel_loc` (before line 56):

```cpp
cv::Point2f refine_peak_loc(cv::Mat const& image_filtered, cv::Point const& peak)
{
    auto const x = peak.x;
    auto const y = peak.y;

    // Need the four orthogonal neighbors present.
    if (x - 1 < 0 || x + 1 >= image_filtered.cols ||
        y - 1 < 0 || y + 1 >= image_filtered.rows)
    {
        return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
    }

    auto const f = [&image_filtered](int px, int py) {
        return image_filtered.at<float>(py, px);
    };

    // 1-D parabola fit along x: offset where the quadratic peaks.
    float x_offset = 0.0f;
    {
        auto const f_m1 = f(x - 1, y);
        auto const f_0  = f(x,     y);
        auto const f_p1 = f(x + 1, y);
        auto const denom = f_m1 - 2.0f * f_0 + f_p1;
        if (std::fabs(denom) > 1e-6f)
            x_offset = (f_m1 - f_p1) / (2.0f * denom);
    }

    float y_offset = 0.0f;
    {
        auto const f_m1 = f(x, y - 1);
        auto const f_0  = f(x, y);
        auto const f_p1 = f(x, y + 1);
        auto const denom = f_m1 - 2.0f * f_0 + f_p1;
        if (std::fabs(denom) > 1e-6f)
            y_offset = (f_m1 - f_p1) / (2.0f * denom);
    }

    return cv::Point2f(static_cast<float>(x) + x_offset,
                       static_cast<float>(y) + y_offset);
}
```

- [ ] **Step 5: Run the full test suite to verify the refinement tests pass and nothing broke**

Run: `cmake --build . -j$(nproc)` then `./nonogram_detector_ut/nonogram_detector_ut`
Expected: `case: subpixel peak refinement` prints `[ok]` for both sub-cases; all other cases still pass (grid dims unchanged, `find_kernel_loc` still returns `cv::Point`).

- [ ] **Step 6: Commit**

```bash
git add nonogram_detector/include/image_operations.hpp nonogram_detector/src/image_operations.cpp nonogram_detector_ut/main.cpp
git commit -m "feat: add subpixel paraboloid peak-refinement helper"
```

---

### Task 2: Make `find_kernel_loc` return `cv::Point2f` and switch detector storage to `CV_32FC2`

This is the coupled type change: `find_kernel_loc` returns subpixel coordinates, and every consumer in the detector stores them as `cv::Point2f`. Everything is updated in one task so the commit stays green.

**Files:**
- Modify: `nonogram_detector/include/image_operations.hpp:33-50` (`find_kernel_loc` signatures)
- Modify: `nonogram_detector/src/image_operations.cpp` (both `find_kernel_loc`, `get_cell_warped_images_vector`)
- Modify: `nonogram_detector/include/detection.hpp:8-18`
- Modify: `nonogram_detector/include/cross_locs_detector.hpp` (several signatures)
- Modify: `nonogram_detector/src/cross_locs_detector.cpp` (whole file)

- [ ] **Step 1: Update `find_kernel_loc` signatures in the header**

In `nonogram_detector/include/image_operations.hpp`, change both `find_kernel_loc` overloads' return type to `std::pair<bool, cv::Point2f>`:

```cpp
// The boolean flag in the return value shows if the search was successful.
// The returned location is subpixel (cv::Point2f).
std::pair<bool, cv::Point2f> find_kernel_loc(
    cv::Mat const& image_thresholded,
    cv::Mat const& kernel,
    double const max,
    double const similarity_ratio_min,
    cv::Point const& anchor);


bool is_inside(cv::Rect const& rect, cv::Rect const& sub_rect);


std::pair<bool, cv::Point2f> find_kernel_loc(
    cv::Mat const& image,
    cv::Rect const& roi,
    cv::Mat const& kernel,
    double const max,
    double const similarity_ratio_min,
    cv::Point const& anchor = cv::Point(-1, -1));
```

- [ ] **Step 2: Update `find_kernel_loc` implementations in `image_operations.cpp`**

Update the first overload (lines 56-84) to return `cv::Point2f` and call `refine_peak_loc`:

```cpp
std::pair<bool, cv::Point2f> find_kernel_loc(
    cv::Mat const& image_thresholded,
    cv::Mat const& kernel,
    double const max,
    double const similarity_ratio_min,
    cv::Point const& anchor)
{
    // Convolve image with a <kernel> to get locations of the <kernel>
    cv::Mat image_filtered;
    cv::filter2D(
        image_thresholded,
        image_filtered,
        CV_32F,
        kernel,
        anchor,
        0.0,
        cv::BORDER_ISOLATED);

    // Normalize the image with a known max value
    image_filtered /= max;

    double peak_max;
    cv::Point peak_max_loc;
    cv::minMaxLoc(image_filtered, nullptr, &peak_max, nullptr, &peak_max_loc);

    if (peak_max > similarity_ratio_min)
    {
        auto const refined = refine_peak_loc(image_filtered, peak_max_loc);
        return std::make_pair(true, refined);
    }

    return std::make_pair(false, cv::Point2f(-1.0f, -1.0f));
}
```

Update the second overload (lines 93-119) to return `cv::Point2f`:

```cpp
std::pair<bool, cv::Point2f> find_kernel_loc(
    cv::Mat const& image_thresholded,
    cv::Rect const& roi,
    cv::Mat const& kernel,
    double const max,
    double const similarity_ratio_min,
    cv::Point const& anchor)
{
    cv::Rect const image_thresholded_roi(cv::Point(0, 0), image_thresholded.size());
    if (!is_inside(image_thresholded_roi, roi))
    {
        return std::make_pair(false, cv::Point2f(-1.0f, -1.0f));
    }

    bool kernel_loc_found;
    cv::Point2f kernel_loc;
    std::tie(kernel_loc_found, kernel_loc) = find_kernel_loc(
        image_thresholded(roi),
        kernel,
        max,
        similarity_ratio_min,
        anchor);

    return kernel_loc_found ?
        std::make_pair(true, kernel_loc + cv::Point2f(roi.tl())) :
        std::make_pair(false, cv::Point2f(-1.0f, -1.0f));
}
```

Note: `cv::Point2f` has a converting constructor from `cv::Point`, so `cv::Point2f(roi.tl())` works.

- [ ] **Step 3: Update the `Detection` struct type and comment**

In `nonogram_detector/include/detection.hpp`, replace the comment block:

```cpp
// Result of a grid detection: a boolean found-flag plus the located grid
// intersection ("cross_loc") matrices for the main cells region and the
// top / left clue regions. Each matrix is CV_32FC2; element (x, y) holds the
// (subpixel) pixel position of the grid intersection at that index of the
// region. (-1, -1) marks a missing location.
struct Detection
{
    bool found = false;
    cv::Mat main;
    cv::Mat top;
    cv::Mat left;
};
```

- [ ] **Step 4: Update the header signatures in `cross_locs_detector.hpp`**

Replace the map/tuple/matrix signatures that mention `cv::Point` (the value side) with `cv::Point2f`. The grid-index keys remain `cv::Point`. Specifically:

- `find_cell_side_length_cell_loc` return: `std::tuple<bool, int, cv::Point>` → `std::tuple<bool, int, cv::Point2f>`
- `get_cross_locs_map`:
  - `cross_locs_init`: `std::vector<cv::Point>` → `std::vector<cv::Point2f>`
  - `cross_loc_deltas`: `std::vector<cv::Point>` → `std::vector<cv::Point2f>`
  - return: `std::map<cv::Point, cv::Point, PointCompare>` → `std::map<cv::Point, cv::Point2f, PointCompare>`
- `get_bounding_rectangle`: unchanged (keys only).
- `convert_to_mat`: `std::map<cv::Point, cv::Point, PointCompare>` → `std::map<cv::Point, cv::Point2f, PointCompare>`
- `augment`: `cv::Mat const&` unchanged (operates on the float matrix).
- `get_cross_locs_main_mat`: `cv::Point const& cross_loc_init` → `cv::Point2f const& cross_loc_init`
- `get_cross_locs_top_mat`, `get_cross_locs_left_mat`: parameter `cv::Mat const&` unchanged.

The static `INDICES_DELTA_*` / `INDICES_DELTAS` remain `cv::Point` (they are index deltas, not pixel coordinates).

- [ ] **Step 5: Update `detect()` and the seed flow in `cross_locs_detector.cpp`**

In `detect()` (lines 109-193):

- `cv::Point cell_loc(0, 0)` → `cv::Point2f cell_loc(0.0f, 0.0f)`.
- `get_cross_locs_main_mat` is called with `cell_loc` (now `Point2f`).
- The `/ scale` steps must preserve sentinels. Replace:

```cpp
detection.main = cross_locs_main_mat / scale;
```

with a sentinel-preserving scale. Add a private static helper (defined in Step 7; declared in the header) and call it:

```cpp
detection.main = scale_cross_locs_mat(cross_locs_main_mat, scale);
```

Do the same for `.top` and `.left`.

The BFS seeds: `cross_locs_init` passed to `get_cross_locs_map` must be `std::vector<cv::Point2f>`. The seed for the main grid is `{ cross_loc_init }` (a `Point2f`). For top/left, the seeds come from `cross_locs_main_mat.at<cv::Point2f>(...)`.

- [ ] **Step 6: Update `get_cross_locs_map` internals to `cv::Point2f`**

In `get_cross_locs_map` (lines 242-309):

- `cross_locs_init_map`: `std::map<cv::Point, cv::Point, PointCompare>` → `std::map<cv::Point, cv::Point2f, PointCompare>`.
- `cross_locs_map`: same change.
- `cross_loc_neighbor_init = cross_loc + cross_loc_deltas[i]` stays valid (`Point2f + Point2f` once deltas are `Point2f`).
- `get_roi(cross_loc_init, roi_size)` needs an integer center — round:

```cpp
cv::Point const cross_loc_init_rounded(
    cvRound(cross_loc_init.x), cvRound(cross_loc_init.y));
```

and use `get_roi(cross_loc_init_rounded, roi_size)` for the search ROI, while the stored `cross_loc` (from `find_kernel_loc`) is the `Point2f` refined value.

- `find_kernel_loc(...)` returns `std::pair<bool, cv::Point2f>` — update the `std::tie(cross_loc_found, cross_loc)` accordingly (already `Point2f`).

- [ ] **Step 7: Update `convert_to_mat`, `augment`, and `get_cross_locs_*_mat`**

`convert_to_mat` (lines 345-374):
- Matrix type `CV_32SC2` → `CV_32FC2`.
- Initial value `cv::Scalar(-1, -1)` → `cv::Scalar(-1.0, -1.0)`.
- Write side: `cross_locs_mat.at<cv::Point2f>(indices_mat) = indices_cross_loc_it->second;`.
- The map value type is now `cv::Point2f`.

`augment` (lines 377-480):
- Sentinel check: `== cv::Point(-1, -1)` → `== cv::Point2f(-1.0f, -1.0f)`.
- `cross_locs_mat_augmented.at<cv::Point>(indices)` reads → `at<cv::Point2f>`.
- `cross_loc_interpolated` accumulation: `cv::Point` → `cv::Point2f`; `cell_side_length * direction` where `direction` is a unit `cv::Point` → cast to float: `cv::Point2f(cell_side_length * direction.x, cell_side_length * direction.y)`. Round the average when writing? No — keep float. The `neighbors[0]` is `Point2f`, so `neighbors[0] + cell_side_length * direction` needs `direction` as `Point2f`. Provide:
  ```cpp
  cv::Point2f const dir_f(static_cast<float>(direction.x), static_cast<float>(direction.y));
  auto const cross_loc_interpolated = neighbors[0] + cell_side_length * dir_f;
  ```
- Sum: `std::accumulate(..., cv::Point2f())`, and divide by `n` as `(float)n`.

`get_cross_locs_main_mat` / `get_cross_locs_top_mat` / `get_cross_locs_left_mat`:
- Matrix type for the resized mats: use `cross_locs_main_mat.type()` (already float) — no hardcoded `CV_32SC2`.
- `cv::Scalar(-1, -1)` → `cv::Scalar(-1.0, -1.0)`.
- Top/left seed extraction: `cross_locs_main_mat.at<cv::Point>(indices)` → `at<cv::Point2f>`; only enqueue if `!= cv::Point2f(-1,-1)`. The local seed vectors `cross_locs_neighbors_init` in `get_cross_locs_top_mat` / `get_cross_locs_left_mat` change from `std::vector<cv::Point>` to `std::vector<cv::Point2f>` (matching `get_cross_locs_map`'s parameter).
- Top/left: `get_mask_cross(cell_side_length_odd)` and ROI unchanged.
- The `cross_loc_deltas` vectors passed to `get_cross_locs_map` must become `std::vector<cv::Point2f>`. For example in `get_cross_locs_main_mat` (lines 500-504):
  ```cpp
  std::vector<cv::Point2f> const cross_loc_deltas = {
      cv::Point2f(0.0f, -cell_side_length),
      cv::Point2f(cell_side_length, 0.0f),
      cv::Point2f(0.0f, cell_side_length),
      cv::Point2f(-cell_side_length, 0.0f) };
  ```
  Do the same for `get_cross_locs_top_mat` and `get_cross_locs_left_mat`.

Add the sentinel-preserving scale helper as a private static in the header and define it:

```cpp
static cv::Mat scale_cross_locs_mat(cv::Mat const& cross_locs_mat, float const scale);
```

```cpp
cv::Mat CrossLocsDetector::scale_cross_locs_mat(cv::Mat const& cross_locs_mat, float const scale)
{
    if (cross_locs_mat.empty())
        return cv::Mat();
    cv::Mat out = cross_locs_mat.clone();
    for (int y = 0; y < out.rows; ++y)
    {
        for (int x = 0; x < out.cols; ++x)
        {
            auto& p = out.at<cv::Point2f>(y, x);
            if (p != cv::Point2f(-1.0f, -1.0f))
                p = cv::Point2f(p.x / scale, p.y / scale);
        }
    }
    return out;
}
```

- [ ] **Step 8: Update `draw()`**

In `draw()` (lines 696-718), change the iteration to `cv::Point2f`:

```cpp
std::for_each(
    cross_locs_mat.begin<cv::Point2f>(),
    cross_locs_mat.end<cv::Point2f>(),
    [&](cv::Point2f const& cross_loc)
    {
        cv::circle(image_copy, cross_loc, radius, color, -1);
    });
```

- [ ] **Step 9: Update `get_cell_warped_images_vector` in `image_operations.cpp`**

Remove the manual `cv::Point2f` casts (read directly):

```cpp
cv::Point2f const cell_tl = cross_locs.at<cv::Point2f>(tl);
cv::Point2f const cell_tr = cross_locs.at<cv::Point2f>(tr);
cv::Point2f const cell_br = cross_locs.at<cv::Point2f>(br);
cv::Point2f const cell_bl = cross_locs.at<cv::Point2f>(bl);
```

(Here `tl`, `tr`, `br`, `bl` are `cv::Point` grid indices; `cross_locs.at<cv::Point2f>(index)` is valid.)

- [ ] **Step 10: Build and run the full test suite**

Run: `cmake --build . -j$(nproc)`
Expected: `nonogram_detector_ut` compiles. `main.cpp` (the app) will likely fail until Task 3 — this task does not build the application target fully. Build the `nonogram_detector_ut` target only:

`cmake --build . --target nonogram_detector_ut -j$(nproc)`

Run: `./nonogram_detector_ut/nonogram_detector_ut`
Expected: all cases pass (grid dims unchanged; refinement tests pass). The `non_file_test` cases remain green because they build their own synthetic matrix and do not depend on `Detection.main`'s type.

- [ ] **Step 11: Commit**

```bash
git add nonogram_detector
git commit -m "feat: store cross locations as subpixel Point2f / CV_32FC2"
```

---

### Task 3: Update the application (`main.cpp`) consumers

**Files:**
- Modify: `nonogram_detector_application/main.cpp` (write_non_file `#cells`, render_nonogram_overlay)

- [ ] **Step 1: Update `write_non_file` `#cells` export to float**

In `write_non_file` (around line 159), replace the integer point write:

```cpp
cv::Point const pt = detection_main.at<cv::Point>(r, c);
out << pt.x << "," << pt.y;
```

with float output preserving subpixel precision:

```cpp
cv::Point2f const pt = detection_main.at<cv::Point2f>(r, c);
out << std::fixed << std::setprecision(2) << pt.x << "," << pt.y;
```

Include `<iomanip>` at the top of `nonogram_detector_application/main.cpp` if not already present.

- [ ] **Step 2: Update `render_nonogram_overlay` to `cv::Point2f`**

Modify `render_nonogram_overlay` (lines 172-253):

- `std::vector<cv::Point>` `cell_poly` → `std::vector<cv::Point>` still works, but now built by rounding the subpixel corners. `cv::fillConvexPoly` **requires integer `cv::Point`** (it asserts `CV_32S`), so read the float corners and round to `cv::Point` for the fill:
  ```cpp
  cell_poly[0] = cv::Point(main_locs.at<cv::Point2f>(r,     c));      // top-left
  cell_poly[1] = cv::Point(main_locs.at<cv::Point2f>(r,     c + 1));  // top-right
  cell_poly[2] = cv::Point(main_locs.at<cv::Point2f>(r + 1, c + 1));  // bottom-right
  cell_poly[3] = cv::Point(main_locs.at<cv::Point2f>(r + 1, c));      // bottom-left

  cv::fillConvexPoly(overlay, cell_poly, fill_color);
  ```
  (`std::vector<cv::Point>` `cell_poly` stays as-is; only the reads change.)
- Grid lines: `cv::line` accepts `cv::Point2f` points directly; update the reads to `.at<cv::Point2f>`:
  ```cpp
  cv::line(result,
           main_locs.at<cv::Point2f>(r, c),
           main_locs.at<cv::Point2f>(r, c + 1),
           line_color, line_thickness);
  ```

- [ ] **Step 3: Build the application**

Run: `cmake --build . -j$(nproc)`
Expected: application and tests build cleanly.

- [ ] **Step 4: Commit**

```bash
git add nonogram_detector_application/main.cpp
git commit -m "feat: emit float cross locs in .non and overlay render"
```

---

### Task 4: Update `.non` round-trip test to float coordinates

**Files:**
- Modify: `nonogram_detector_ut/non_file_test.cpp`

- [ ] **Step 1: Update the parser and test to float**

In `non_file_test.cpp`:
- `struct NonCustomBlock` field `cells`: `std::vector<cv::Point>` → `std::vector<cv::Point2f>`.
- In `parse_non_custom_block`, parse floats:
  ```cpp
  auto const comma = token.find(',');
  float x = std::stof(token.substr(0, comma));
  float y = std::stof(token.substr(comma + 1));
  block.cells.emplace_back(x, y);
  ```
- In `test_round_trip`, `main_locs` is `CV_32SC2` with `cv::Point`. Change to `CV_32FC2` and write `cv::Point2f`:
  ```cpp
  cv::Mat main_locs(3, 4, CV_32FC2);
  for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c)
          main_locs.at<cv::Point2f>(r, c) = cv::Point2f(100.0f + c * 50.0f, 200.0f + r * 60.0f);
  ```
- The writing block (mirrors `write_non_file`) writes float:
  ```cpp
  cv::Point2f const pt = main_locs.at<cv::Point2f>(r, c);
  out << std::fixed << std::setprecision(2) << pt.x << "," << pt.y;
  ```
- Comparison loop:
  ```cpp
  cv::Point2f const expected = main_locs.at<cv::Point2f>(r, c);
  cv::Point2f const actual = block.cells[r * 4 + c];
  if (actual != expected) { ... }
  ```
  `std::setprecision(2)` matches the 2-decimal write, so equality holds for these exact .0 values.
- Add `#include <iomanip>` if needed.

- [ ] **Step 2: Build and run the full test suite**

Run: `cmake --build . -j$(nproc)` then `./nonogram_detector_ut/nonogram_detector_ut`
Expected: `all tests passed`.

- [ ] **Step 3: Commit**

```bash
git add nonogram_detector_ut/non_file_test.cpp
git commit -m "test: round-trip float cross coordinates in .non"
```

---

### Task 5: Final verification

- [ ] **Step 1: Clean full rebuild and test**

```bash
cmake --build . -j$(nproc)
./nonogram_detector_ut/nonogram_detector_ut
```

Expected: builds clean; `all tests passed`.

- [ ] **Step 2: Manual smoke test on a real photo**

Run the application on an existing photo (e.g. one of the files under `nonograms/` or `digits_marked/`), with `NG_EXPORT_NON` and `NG_EXPORT_OVERLAY` set:

```bash
NG_SAVE_OUTPUT=1 ./nonogram_detector_application/nonogram_detector_application <photo> 1200
```

Inspect `grid.png` (the drawn cross circles) and the exported `.non` `#cells:` to confirm non-integer coordinates appear and the overlay aligns with line intersections.

- [ ] **Step 3: Report**

Confirm the grid dimensions and detection success remain unchanged from before the change by comparing the detected grid size on the same input against the pre-change baseline (Task baseline: tests report the same `grid WxH`).

---

## Self-Review Notes

- **Spec coverage:** paraboloid fit (Task 1), `CV_32FC2` storage (Task 2), consumers (Tasks 2-3), `.non` float export (Task 3), tests (Tasks 1 & 4), sentinel-preserving scale (Task 2 Step 7) — all covered.
- **Type consistency:** `cv::Point2f` is used consistently; grid-index keys remain `cv::Point`; `scale_cross_locs_mat` is declared in the header (Task 2 Step 7) and used in `detect()` (Task 2 Step 5).
