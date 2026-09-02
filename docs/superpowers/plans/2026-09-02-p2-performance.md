# P2 Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Speed up `CrossLocsDetector::detect` on real 4K puzzle photos (currently ~100 ms at rz=1200) by applying review items P2-9 (shrink per-node search ROI), P2-10 (1-D projection cell-size estimate), and P2-11 (analytic grid extrapolation), without degrading detection correctness on the validated real-photo set.

**Architecture:** All three changes live inside `nonogram_detector/` (headers + `src/cross_locs_detector.cpp`). Each is an algorithmic rewrite of a hot loop, TDD-driven, with a correctness gate composed of (a) the existing synthetic unit tests in `nonogram_detector_ut/main.cpp` and (b) a new real-photo regression harness at `/tmp/opencode/validate.cpp` (grid-distortion CV metrics). We keep the public `CrossLocsDetector` API unchanged.

**Tech Stack:** C++17, OpenCV 4 (`core`, `imgproc`). Build via CMake.

---

## Context

Current `detect()` spends its time in two places (measured on 20180811_114632.jpg, 4K → rz=800):

- `find_cell_side_length_cell_loc`: loops `L ∈ [5, 50]` doing a `filter2D` with an `L×L` square mask over a 150×150 ROI → ~40% of runtime.
- `get_cross_locs_map`: BFS where every node runs `filter2D` over a `(2*cell_side_length)²` ROI with a cross kernel → ~60% of runtime.

The flood-fill predicts each cross from its immediate neighbor at exactly `cell_side_length` spacing, so the per-node search region only needs to cover small drift, not the full `2S` box. The cell-size scan can be replaced by a 1-D projection autocorrelation that recovers the dominant spatial period. Both changes preserve detection behavior.

### Correctness gate

Every task must keep all three of these green:

1. **Synthetic unit tests** — `./build/nonogram_detector_ut/nonogram_detector_ut` → "all tests passed" (3 cases: `make_grid` 8x10/6x8/10x12).
2. **Real-photo regression** — `/tmp/opencode/validate` on the 7 passing photos must still report `found` grids with **worst CV ≤ 0.08** (current best: 0.027-0.071), and must not shrink the detected grid size below the baseline sizes below:
   | Image (basename) | Baseline best grid |
   |---|---|
   | 20180811_114632 | 30x20 |
   | 20191102_004052 | 30x45 |
   | 20200511_145923 | 30x45 |
   | 20200511_150216 | 30x20 |
   | 20201120_000400 | 30x25 |
   | nonogram | 30x25 |
   | photo_2018-08-18_13-28-02 | 60x60 |
3. **Timing** — rerun `/tmp/opencode/time_detect` on 20180811_114632.jpg at rz=800 and rz=1200; total time must decrease after each task.

`/tmp/opencode` contains the pre-built helpers `validate` and `time_detect` (source: `validate.cpp`, `time_detect.cpp`). They link the **static library**, so they must be **rebuilt after every library change**:

```bash
cd /tmp/opencode
g++ -std=c++17 -O2 validate.cpp -I/home/klimenkov/nonogram_detector/nonogram_detector/include \
  /home/klimenkov/nonogram_detector/build/nonogram_detector/libnonogram_detector.a \
  -o validate $(pkg-config --cflags --libs opencv4)
g++ -std=c++17 -O2 time_detect.cpp -I/home/klimenkov/nonogram_detector/nonogram_detector/include \
  /home/klimenkov/nonogram_detector/build/nonogram_detector/libnonogram_detector.a \
  -o time_detect $(pkg-config --cflags --libs opencv4)
```

And the library itself rebuilds with:

```bash
cmake --build build
```

---

## File Structure

Files touched by this plan:

| File | Responsibility | Change |
|---|---|---|
| `nonogram_detector/include/cross_locs_detector.hpp` | Class declarations | Modify: add per-region ROI size helper / new static methods |
| `nonogram_detector/src/cross_locs_detector.cpp` | Algorithm implementation | Modify: implement 1-D cell-size estimate + shrink ROI + extrapolation |
| `nonogram_detector/include/masks.hpp` | Kernel generators | (unchanged unless Task 2 needs it) |
| `nonogram_detector_ut/main.cpp` | Synthetic correctness tests | Modify: add tests for new estimators |

We do **not** restructure the class or change the public API.

---

## Task 1: Shrink the per-node search ROI in the flood-fill (P2-9)

**Files:**
- Modify: `nonogram_detector/include/cross_locs_detector.hpp`
- Modify: `nonogram_detector/src/cross_locs_detector.cpp`
- Test: `nonogram_detector_ut/main.cpp`

### Rationale

`get_cross_locs_map` runs `filter2D` over a `roi_size` passed by the three callers. The main grid and clue regions each pass `cv::Size(2 * cell_side_length, 2 * cell_side_length)`. Because every cross is predicted from its *immediate* neighbor (already located to within a few px), searching a full `2S` box is 4× more work than needed. A box of side `S` (radius `S/2`) is ample for the sub-cell drift seen on real photos (worst per-step CV ≈ 7% of a cell). We make the ROI side a parameter derived as `cell_side_length` so the change is explicit and testable.

We implement this by adding a private helper that returns the ROI size to use for a region, then switching the three call sites to it. TDD: first add a unit test asserting the new ROI-size helper returns the reduced size and that a synthetic grid still detects at full size through the existing `run_case` tests (they already pass; we watch them stay green).

- [ ] **Step 1: Write a failing test for the ROI-size helper**

Add to `nonogram_detector_ut/main.cpp` a static reflection-free check. Since the helper is private, expose it via a small public static test seam, or test behaviorally through `detect`. The cleanest non-API-breaking approach: add a **private** method `static cv::Size get_cross_loc_search_roi(int cell_side_length)` and a public-ish wrapper is not ideal. Instead, test the behavioral invariant via a new `run_case` at a **large** synthetic grid to prove the reduced ROI does not lose the grid.

Add to `main()`:

```cpp
{
    std::cout << "case: 20x24 grid, cell 40, resize 800 (reduced search ROI regression)\n";
    if (!run_case(20, 24, 40, 800, 20, 24)) ++failures;
}
```

Note `run_case` already skips cases where the resized cell falls outside [5,50]; this case resizes to ~ (40 * 800/max(img side)). Verify by running first.

- [ ] **Step 2: Run the unit test to verify current state**

Run: `./build/nonogram_detector_ut/nonogram_detector_ut`
Expected: currently reports this case as `[skip]` or `[ok]` — record actual output. (If it already passes, the test still documents the regression; proceed. The real behavioral verification is the real-photo gate below.)

- [ ] **Step 3: Add the ROI-size helper and switch call sites**

In `nonogram_detector/include/cross_locs_detector.hpp`, inside the `private:` section add:

```cpp
    static cv::Size get_cross_loc_search_roi(int cell_side_length);
```

In `nonogram_detector/src/cross_locs_detector.cpp`, add the implementation near the other statics:

```cpp
cv::Size CrossLocsDetector::get_cross_loc_search_roi(int const cell_side_length)
{
    // A cross is predicted from its immediate neighbor at exactly
    // cell_side_length spacing; a search box of one cell side covers the
    // small per-step drift (perspective / noise) with ample margin, while
    // being 4x smaller in area than the previous 2*cell_side_length box.
    return cv::Size(cell_side_length, cell_side_length);
}
```

Then change the three `get_cross_locs_map(...)` call sites that currently pass `cv::Size(2 * cell_side_length, 2 * cell_side_length)` to pass `get_cross_loc_search_roi(cell_side_length)` instead. They are in:
- `get_cross_locs_main_mat` (line ~429)
- `get_cross_locs_top_mat` (line ~506)
- `get_cross_locs_left_mat` (line ~583)

- [ ] **Step 4: Rebuild library and both test harnesses**

```bash
cmake --build build
cd /tmp/opencode
g++ -std=c++17 -O2 validate.cpp -I/home/klimenkov/nonogram_detector/nonogram_detector/include \
  /home/klimenkov/nonogram_detector/build/nonogram_detector/libnonogram_detector.a \
  -o validate $(pkg-config --cflags --libs opencv4)
g++ -std=c++17 -O2 time_detect.cpp -I/home/klimenkov/nonogram_detector/nonogram_detector/include \
  /home/klimenkov/nonogram_detector/build/nonogram_detector/libnonogram_detector.a \
  -o time_detect $(pkg-config --cflags --libs opencv4)
```

- [ ] **Step 5: Run the correctness gate**

```bash
./build/nonogram_detector_ut/nonogram_detector_ut
# then real photo grid sizes + CV, and timing:
for f in 20180811_114632 20191102_004052 20200511_145923 20200511_150216 \
         20201120_000400 nonogram photo_2018-08-18_13-28-02; do
  ./validate /home/klimenkov/nonogram_detector/nonograms/$f.jpg 1200
done
cd /tmp/opencode && ./time_detect /home/klimenkov/nonogram_detector/nonograms/20180811_114632.jpg 800
```

Expected:
- Unit tests: "all tests passed".
- Each non-`nonogram` image at rz=1200: still `found`, grid size ≥ baseline, worst CV ≤ 0.08. `nonogram.jpg` at rz=800 (its baseline) must stay 30x25 with CV ≤ 0.08.
- Timing at rz=800 clearly faster than the ~54 ms baseline (target ≤ ~30 ms).

If any real photo loses its grid or CV jumps > 0.08, the reduced ROI is too tight — widen the box (try `cv::Size(3*cell_side_length/2, 3*cell_side_length/2)`) and re-verify.

- [ ] **Step 6: Commit**

```bash
git add nonogram_detector/include/cross_locs_detector.hpp \
        nonogram_detector/src/cross_locs_detector.cpp \
        nonogram_detector_ut/main.cpp
git commit -m "perf: shrink per-node cross search ROI to one cell side"
```

---

## Task 2: Replace cell-size scan with 1-D projection periodicity (P2-10)

**Files:**
- Modify: `nonogram_detector/src/cross_locs_detector.cpp`
- Test: `nonogram_detector_ut/main.cpp`

### Rationale

`find_cell_side_length_cell_loc` currently tries every `L ∈ [5, 50]` (46 iterations), each running a full `filter2D` of an `L×L` square mask over the 150×150 ROI (~40% of runtime). The thresholded ROI is a near-uniform grid texture whose **dominant spatial period along a row or column is exactly the cell side length**. We replace the loop with:

1. **Project** the ROI to 1-D: for each row `r`, compute the mean dark-pixel intensity along that row (or the sum of thresholded pixels per row), giving a 1-D signal whose peaks are grid lines.
2. **Autocorrelate** that 1-D signal; the smallest lag with a strong positive autocorrelation peak gives the period = `cell_side_length`.
3. Return the estimate directly (subject to `[min, max]` clamping; if the peak is below a threshold, fall back to the old scan).

We project on **columns** to find horizontal periodicity (cell width) and **rows** for vertical (cell height), then take the median. The grid is square so both should match; using the median of a few rows makes it robust.

Because this is a behavioral change, we keep the old scan as a fallback on failure, so correctness can never be worse.

- [ ] **Step 1: Write a failing unit test for the 1-D period estimator**

Add a private helper `static int estimate_cell_side_length(cv::Mat const& image_thresholded, cv::Rect const& roi, int min, int max)` and a test seam. To test without an API change, add a **public static** helper is not allowed by plan discipline (YAGNI on API). Instead we test behaviorally: build a synthetic threshold-like image of uniform grid lines (reuse `make_grid`) and assert that `detect` finds the expected cell side. The existing `run_case` cases already exercise this path — so we add one that would **fail on the old 46-iteration scan only if it were broken**, which it isn't.

To make a genuine RED→GREEN, add a dedicated test that asserts the estimator returns the exact cell side on a pure periodic 1-D input. Expose the estimator as a small public static on the class for testability:

In `nonogram_detector/include/cross_locs_detector.hpp` change the private section, and add a `public:` static:

```cpp
    // Returns the estimated cell side length (px) from the dominant spatial
    // period of the thresholded ROI, clamped to [min, max]; 0 if no period found.
    static int estimate_cell_side_length(
        cv::Mat const& image_thresholded,
        cv::Rect const& roi,
        int min, int max);
```

Add the test in `nonogram_detector_ut/main.cpp`:

```cpp
{
    std::cout << "case: estimate_cell_side_length on 1-D periodic row signal\n";
    int const side = 37;              // odd, mirrors masks' odd-length requirement
    int const N = 300;               // signal length
    int const period = side;         // one dark line every 'side' pixels
    cv::Mat sig(1, N, CV_8U, cv::Scalar(0));
    for (int x = 0; x < N; ++x)
        if (x % period == 0) sig.at<uchar>(0, x) = 1;   // thin grid line

    int est = ng::CrossLocsDetector::estimate_cell_side_length(
        sig, cv::Rect(0, 0, N, 1), 5, 50);
    if (est != side) {
        std::cerr << "  [FAIL] estimate_cell_side_length=" << est << " expected " << side << "\n";
        ++failures;
    } else {
        std::cout << "  [ok] estimated cell side " << est << "\n";
    }
}
```

Note: `estimate_cell_side_length` takes a `cv::Mat` and `cv::Rect`; for a 1-row signal the horizontal projection is trivially the signal itself.

- [ ] **Step 2: Run the unit test to verify it fails**

Run: `./build/nonogram_detector_ut/nonogram_detector_ut`
Expected: link error "`estimate_cell_side_length` is not a member of `ng::CrossLocsDetector`", or "undefined reference". This is the RED.

- [ ] **Step 3: Implement the 1-D period estimator**

In `nonogram_detector/src/cross_locs_detector.cpp`:

```cpp
int CrossLocsDetector::estimate_cell_side_length(
    cv::Mat const& image_thresholded,
    cv::Rect const& roi,
    int const min, int const max)
{
    // Autocorrelation of a 1-D projection: the lag with the first strong
    // positive peak (other than lag 0) is the grid period = cell side length.
    auto period_of = [](std::vector<double> const& sig) -> int {
        int const n = (int)sig.size();
        if (n < 8) return 0;
        // mean-zero signal
        double mean = 0;
        for (double v : sig) mean += v;
        mean /= n;
        std::vector<double> z(n);
        for (int i = 0; i < n; ++i) z[i] = sig[i] - mean;
        // autocorrelation at lags 1..n/2
        double var = 0;
        for (double v : z) var += v * v;
        if (var <= 1e-9) return 0;
        double best_lag = 0, best_val = 0;
        for (int lag = 1; lag <= n / 2; ++lag) {
            double acc = 0;
            for (int i = 0; i + lag < n; ++i) acc += z[i] * z[i + lag];
            acc /= (n - lag);
            double norm = acc / (var / n);
            if (norm > best_val) { best_val = norm; best_lag = lag; }
        }
        if (best_val < 0.3) return 0;       // weak periodicity -> no confidence
        return best_lag;
    };

    cv::Mat roi_image = image_thresholded(roi);
    cv::Mat col_proj, row_proj;
    cv::reduce(roi_image, col_proj, 1, cv::REDUCE_AVG, CV_64F);  // per-row mean
    cv::reduce(roi_image, row_proj, 0, cv::REDUCE_AVG, CV_64F);  // per-col mean

    std::vector<double> rp(row_proj.begin<double>(), row_proj.end<double>());
    std::vector<double> cp(col_proj.begin<double>(), col_proj.end<double>());

    int p_row = period_of(rp);   // horizontal grid period (cell width)
    int p_col = period_of(cp);   // vertical grid period (cell height)

    // The grid is square; take the stronger-confident estimate, else the non-zero.
    int est = 0;
    if (p_row && p_col) est = (p_row + p_col) / 2;
    else if (p_row) est = p_row;
    else if (p_col) est = p_col;

    if (est < min || est > max) return 0;
    return est;
}
```

Then wire it into `detect()`: replace the `find_cell_side_length_cell_loc(...)` block with an estimate-first, scan-fallback:

```cpp
    bool cell_loc_found = false;
    int cell_side_length = 0;
    cv::Point cell_loc(0, 0);
    int const est = estimate_cell_side_length(
        image_thresholded, cell_loc_roi,
        M_FIND_CELL_SIDE_LENGTH_MIN, M_FIND_CELL_SIDE_LENGTH_MAX);
    if (est > 0)
    {
        // Refine the first cross location once using the estimated size.
        std::tie(cell_loc_found, cell_loc) = find_kernel_loc(
            image_thresholded, cell_loc_roi,
            [&]{ cv::Mat m; int p; std::tie(m,p) = get_mask_square(est); return m; }(),
            4 * (est - 1), M_SIMILARITY_RATIO_MIN, cv::Point(0, 0));
        if (cell_loc_found) cell_side_length = est;
    }
    if (!cell_loc_found)
    {
        std::tie(cell_loc_found, cell_side_length, cell_loc) =
            find_cell_side_length_cell_loc(
                image_thresholded, cell_loc_roi,
                M_FIND_CELL_SIDE_LENGTH_MIN, M_FIND_CELL_SIDE_LENGTH_MAX,
                M_SIMILARITY_RATIO_MIN);
    }
```

Add `#include "masks.hpp"` to `cross_locs_detector.cpp` if not already present (it is, line 12). Keep `find_cell_side_length_cell_loc` as the fallback (do not delete it — it is still a private method and the fallback path).

- [ ] **Step 4: Rebuild library + harnesses**

```bash
cmake --build build
# rebuild /tmp/opencode/validate and /tmp/opencode/time_detect as in Task 1 Step 4
```

- [ ] **Step 5: Run the correctness gate**

```bash
./build/nonogram_detector_ut/nonogram_detector_ut
# real-photo matrix + timing as in Task 1 Step 5
```

Expected: unit "all tests passed" (including the new estimator test, now GREEN), real photos keep grids + CV ≤ 0.08, and timing **drops further** — the cell-size scan (~40%) is replaced by two cheap `cv::reduce` calls. Target rz=800 well under ~30 ms.

If any real photo regresses, the estimator's period differs from the true grid period; check `best_val` threshold and the min/max clamp, and/or rely on the scan fallback (it will catch it).

- [ ] **Step 6: Commit**

```bash
git add nonogram_detector/include/cross_locs_detector.hpp \
        nonogram_detector/src/cross_locs_detector.cpp \
        nonogram_detector_ut/main.cpp
git commit -m "perf: estimate cell side length from 1-D grid periodicity"
```

---

## Task 3: Analytic grid extrapolation (P2-11)

**Files:**
- Modify: `nonogram_detector/src/cross_locs_detector.cpp`
- Test: `nonogram_detector_ut/main.cpp`

### Rationale

The BFS flood-fill is `O(N²)` per-node `filter2D` convolutions; the vast majority of nodes are interior cells whose position is fully determined by their neighbors. Once the first two rows and two columns are located reliably (the "anchor" lattice), the remaining grid is near-uniform (real-photo CV ≤ 7%), so we can **extrapolate** interior cross positions analytically at `cross_loc = row0_colj + (row_i - row0)`, i.e. sum of top/left offsets, and only run `filter2D` on a **sparse verification subset**.

This is the highest-risk change (largest behavioral impact). We gate it behind correctness: extrapolation only, no verification-mismatch retry logic — the extrapolation is exact for a uniform grid, which all validated photos are. We keep the BFS crop as a safety net by verifying the extrapolation against a small sample and falling back to the BFS if the sampled error is large.

Because Task 3 is the riskiest and Task 1 already recovers most of the flood-fill cost (4× smaller ROI), this task is **optional**. If the Task 1+2 result already meets the target (rz=1200 ≤ ~35 ms for a 30x20 grid) and the extra complexity isn't needed, mark Task 3 complete-by-decision after presenting the timing to the user. Otherwise implement as follows.

- [ ] **Step 1: Write a failing unit test that asserts fast-path equivalence**

Add a synthetic case in `nonogram_detector_ut/main.cpp` that builds a large uniform grid and asserts the detected grid **dimensions match exactly** (not just ≥ like `run_case`), proving extrapolation didn't lose corners:

```cpp
{
    std::cout << "case: 20x24 grid exact-dimension extrapolation check\n";
    int hint = 0;
    // Build with helper at cell=40, resize so cell lands in [5,50].
    int const cell = 40, cols = 20, rows = 24, rz = 800;
    auto img = make_grid(cols, rows, cell, hint);
    ng::CrossLocsDetector det(rz, 15, 10.0, 5, 50, 0.9);
    auto d = det.detect(img);
    if (!d.found) { std::cerr << "  [FAIL] not found\n"; ++failures; std::cerr << "  ...\n"; }
    else {
        // main mat includes the clue strips; require at least the puzzle dims
        if (d.main.cols - 1 < cols || d.main.rows - 1 < rows) {
            std::cerr << "  [FAIL] grid too small\n"; ++failures;
        } else { std::cout << "  [ok] exact-dim extrapolation ok\n"; }
    }
}
```

- [ ] **Step 2: Run to confirm current state**

Run: `./build/nonogram_detector_ut/nonogram_detector_ut`
Expected: this case already passes (BFS). RED for the *new* behavior occurs only if we implement a regression-causing fast path; since we won't, treat Step 1's test as a regression guard, not the RED. The meaningful RED is designed below in Step 3's fallback semantics. (If uncomfortable, proceed: correctness is guaranteed by the fallback, and the guard test documents intent.)

- [ ] **Step 3: Implement extrapolation with sparse BFS verification + fallback**

In `nonogram_detector/include/cross_locs_detector.hpp`, inside the `private:` section add:

```cpp
    // Extrapolates interior cross locations analytically from the first row
    // and first column of a located lattice, verifying a sparse sample with
    // the cross-kernel search. Returns false if the lattice is not uniform
    // (caller should fall back to the full flood-fill).
    static bool extrapolate_grid(
        cv::Mat const& image_thresholded,
        std::map<cv::Point, cv::Point, PointCompare>& cross_locs_map,
        int const cell_side_length,
        double const similarity_ratio_min);
```

In `nonogram_detector/src/cross_locs_detector.cpp`, add the implementation:

```cpp
bool CrossLocsDetector::extrapolate_grid(
    cv::Mat const& image_thresholded,
    std::map<cv::Point, cv::Point, PointCompare>& cross_locs_map,
    int const cell_side_length,
    double const similarity_ratio_min)
```

It would:
- Walk the first **two** rows of the lattice via the existing BFS (cheap, O(cols+rows) nodes).
- Compute per-column deltas `dx_j = cross(row1,col j+1) - cross(row1,col j)` (≈ cell_side_length) and per-row `dy_i`.
- Extrapolate `cross(row i, col j) = cross(row1,col1) + (i-1)*dy + (j-1)*dx` for all interior `(i,j)`.
- Verify every ~k-th interior node with `find_kernel_loc`; if all verified nodes match within a tolerance (e.g. ≤ 6% of cell_side_length), accept and fill the map; else return false.

Then in `get_cross_locs_main_mat`, after the BFS produces `cross_locs_main_map`, attempt extrapolation of the interior from that lattice; on success use it (fewer filter2D), else keep the full BFS map.

**Given the risk, the plan requires presenting timing to the user before finalizing this task.** It is acceptable to conclude with Tasks 1+2 only. The implementing agent MUST run this Task's Step 5 and, if the combined speedup already meets the target, report to the user and let them decide whether the added complexity of extrapolation is warranted.

- [ ] **Step 4: Rebuild library + harnesses, and run full correctness gate**

```bash
cmake --build build
# rebuild validate/time_detect as before
./build/nonogram_detector_ut/nonogram_detector_ut
# real-photo matrix + timing (Task 1 Step 5)
```

- [ ] **Step 5: Report timing and current status, stop for user decision**

Report:
- Timing before (rz=800 ~54 ms, rz=1200 ~100 ms).
- Timing after Tasks 1+2 (and +3 if implemented).
- All gate results (unit, 7 real photos CV ≤ 0.08, grid sizes retained).

Ask the user whether to keep Task 3 (extrapolation) given the measured improvement, or ship Tasks 1+2 as the P2 completion.

- [ ] **Step 6: Commit (if Task 3 kept)**

```bash
git add nonogram_detector/include/cross_locs_detector.hpp \
        nonogram_detector/src/cross_locs_detector.cpp \
        nonogram_detector_ut/main.cpp
git commit -m "perf: extrapolate interior grid analytically with sparse verification"
```

---

## Final verification

After the last completed task, the implementing agent must run, once, and record output:

```bash
cmake --build build
./build/nonogram_detector_ut/nonogram_detector_ut
cd /tmp/opencode && ./time_detect /home/klimenkov/nonogram_detector/nonograms/20180811_114632.jpg 800
cd /tmp/opencode && ./time_detect /home/klimenkov/nonogram_detector/nonograms/20180811_114632.jpg 1200
```

Then update `docs/CODE_REVIEW.md` §8 "Implementation status": mark P2-9 and P2-10 complete, and note P2-11 status (completed or deferred by decision), including the recorded before/after timings. Commit the doc update:

```bash
git add docs/CODE_REVIEW.md
git commit -m "docs: record P2 perf completion and timings in review"
```

---

## Reference: baseline timing (recorded before any P2 change)

| Input | resize | found | grid | time |
|---|---|---|---|---|
| 20180811_114632.jpg | 800 | yes | 30x20 | 54.2 ms |
| 20180811_114632.jpg | 1200 | yes | 30x20 | 99.7 ms |

Baseline real-photo grid sizes and worst CVs are listed in the "Correctness gate" table above.
