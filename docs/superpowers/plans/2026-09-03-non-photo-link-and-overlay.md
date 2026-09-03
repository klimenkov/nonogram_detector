# .non Photo-Link + Grid-Geometry + Overlay Rendering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When the application exports a solved nonogram to webpbn `.non`, record the source-photo reference and grid intersections in a `#`-prefixed custom block, and render the solved cells directly onto the photo as an overlay image.

**Architecture:** All new code lives in the application layer (`nonogram_detector_application/main.cpp`), because `write_non_file`/`render_nonogram_overlay` need both `nonogram_solver` types (`ClueConstraints`, `SolutionGrid`) and the `nonogram_detector` `Detection` type — and the two libraries must not depend on each other. `Detection.main` already holds the grid intersections in original-image pixels (`cross_locs_detector.cpp` divides by the resize scale), so no rescaling is needed. Validation follows the project's existing idiom: a standalone `/tmp/opencode` harness plus end-to-end output inspection.

**Tech Stack:** C++17, OpenCV (core, imgproc, imgcodecs), existing `ng` libraries.

**Spec:** `docs/superpowers/specs/2026-09-03-non-photo-link-and-overlay-design.md`

---

## Geometry reference (for every task)

- `detection.main` is `CV_32SC2`, with `(H+1)` rows × `(W+1)` cols of intersections, giving an `H × W` cell grid matching the solution.
- Solution grid: `solution[r][c]` for `r in [0,H)`, `c in [0,W)`; value `1` = filled, `0` = empty.
- Intersection at grid position `(r, c)` is `detection.main.at<cv::Point>(cv::Point(c, r))`.
- Cell `(r, c)` quadrilateral (same ordering as `get_cell_warped_images_vector`):
  top-left `(c,r)`, top-right `(c+1,r)`, bottom-right `(c+1,r+1)`, bottom-left `(c,r+1)`.

---

## Task 1: Extend `write_non_file` to emit the photo-link + geometry custom block

**Files:**
- Modify: `nonogram_detector_application/main.cpp:84-135` (current 3-arg `write_non_file`)

The current in-progress 3-arg `write_non_file(path, constraints, solution)` is already in the working tree (uncommitted). Change it to a 4-arg version that also writes the `#source` / `#grid` / `#cells` block.

- [ ] **Step 1: Replace the `write_non_file` signature and body**

Replace the *entire* current function (from the comment `// Writes the puzzle in webpbn...` through its closing brace) with:

```cpp
// Writes the puzzle in webpbn ".non" text format: the row/column clue lists
// (comma-separated) plus a "goal" line (the rows-major 0/1 bitmap of the solved
// grid). Appends a "#"-prefixed custom block recording the source photo
// (relative to this file) and the playing-area grid intersections (original
// image pixels) so the solved grid can be rendered back onto the photo.
void write_non_file(
    std::filesystem::path const& path,
    ng::ClueConstraints const& constraints,
    ng::SolutionGrid const& solution,
    cv::Mat const& main_locs,                 // Detection.main intersections
    std::filesystem::path const& image_path)  // source photo
{
    std::ofstream out(path);
    if (!out)
    {
        std::cerr << "write_non_file: cannot open " << path << "\n";
        return;
    }
    std::size_t const h = constraints.rows.size();
    std::size_t const w = constraints.cols.size();

    out << "title \"nonogram_detector\"\n";
    out << "width " << w << "\n";
    out << "height " << h << "\n\n";

    auto const clues = [](std::vector<int> const& line) {
        std::string s;
        for (std::size_t i = 0; i < line.size(); ++i)
        {
            if (i) s += ",";
            s += std::to_string(line[i]);
        }
        return s;
    };

    out << "rows\n";
    for (auto const& line : constraints.rows)
    {
        std::string const s = clues(line);
        out << (s.empty() ? "0" : s) << "\n";
    }
    out << "\n";

    out << "columns\n";
    for (auto const& line : constraints.cols)
    {
        std::string const s = clues(line);
        out << (s.empty() ? "0" : s) << "\n";
    }
    out << "\n";

    if (!solution.empty())
    {
        std::string goal;
        for (auto const& row : solution)
            for (int const cell : row)
                goal += (cell ? '1' : '0');
        out << "goal \"" << goal << "\"\n";
    }

    // Custom block: source photo path relative to this .non file.
    auto const out_dir = path.has_parent_path() ? path.parent_path()
                                                : std::filesystem::path(".");
    auto const rel = std::filesystem::relative(image_path, out_dir);
    out << "#source: " << (rel.empty() ? image_path.string() : rel.string()) << "\n";

    if (!main_locs.empty())
    {
        int const rows = main_locs.rows;
        int const cols = main_locs.cols;
        out << "#grid: " << rows << "x" << cols << "\n";
        out << "#cells:\n";
        for (int r = 0; r < rows; ++r)
        {
            for (int c = 0; c < cols; ++c)
            {
                cv::Point const p = main_locs.at<cv::Point>(cv::Point(c, r));
                out << p.x << "," << p.y;
                if (c + 1 < cols)
                    out << " ";
            }
            out << "\n";
        }
    }

    std::cout << "wrote .non: " << path << "\n";
}
```

- [ ] **Step 2: Update the call site**

At `nonogram_detector_application/main.cpp:325` the call is currently:
```cpp
if (char const* export_path = std::getenv("NG_EXPORT_NON"))
    write_non_file(export_path, constraints, result.solution);
```
Change it to:
```cpp
if (char const* export_path = std::getenv("NG_EXPORT_NON"))
    write_non_file(export_path, constraints, result.solution,
                   detection.main, image_path);
```

- [ ] **Step 3: Build**

Run: `cmake --build build --target nonogram_detector_application -j`
Expected: builds with no errors or warnings introduced.

- [ ] **Step 4: End-to-end verify the custom block**

Run:
```bash
cd /home/klimenkov/nonogram_detector
rm -f /tmp/opencode/out.non
NG_CLUE_FIXES=/tmp/opencode/col21_fixes.txt \
NG_EXPORT_NON=/tmp/opencode/out.non \
./build/nonogram_detector_application/nonogram_detector_application \
  nonograms/20180811_114632.jpg 2>&1 | grep -E "wrote .non|consistency|solutions"
```
Expected: prints `wrote .non: /tmp/opencode/out.non`, `consistent`, `solutions=1 line_solvable=true`.

Check the tail of the file:
```bash
tail -6 /tmp/opencode/out.non
```
Expected: last lines are `#source: <relative path to 20180811_114632.jpg>`, `#grid: ...x...`, `#cells:` followed by `x,y` pairs. The `#source` path preceded by `../` (since `.non` is under `/tmp/opencode` and the photo under `nonograms/`).

- [ ] **Step 5: Commit**

```bash
git add nonogram_detector_application/main.cpp
git commit -m "feat: record source photo and grid geometry in .non export"
```

---

## Task 2: Render the solved nonogram onto the photo (`NG_EXPORT_OVERLAY`)

**Files:**
- Modify: `nonogram_detector_application/main.cpp` (add `render_nonogram_overlay` helper + wire env)

- [ ] **Step 1: Add the render helper**

Insert this function directly after `write_non_file` (before `#endif`):

```cpp
// Draws the solved grid onto a copy of <photo>: fills each filled cell's
// quadrilateral (from intersections in <main_locs>, original-image pixels) with
// a translucent color, strokes the grid lines, and writes the result to
// <out_path>. Returns true on success.
bool render_nonogram_overlay(
    cv::Mat const& photo,
    cv::Mat const& main_locs,
    ng::SolutionGrid const& solution,
    std::filesystem::path const& out_path)
{
    if (photo.empty() || main_locs.empty() || solution.empty() || main_locs.rows < 2)
    {
        std::cerr << "render_nonogram_overlay: bad input\n";
        return false;
    }
    int const cells_h = main_locs.rows - 1;
    int const cells_w = main_locs.cols - 1;
    if (static_cast<int>(solution.size()) != cells_h ||
        static_cast<int>(solution[0].size()) != cells_w)
    {
        std::cerr << "render_nonogram_overlay: solution/grid size mismatch\n";
        return false;
    }

    cv::Mat overlay = photo.clone();
    cv::Mat base = photo.clone();

    auto corners = [&](int r, int c) {
        return std::vector<cv::Point>{
            main_locs.at<cv::Point>(cv::Point(c, r)),
            main_locs.at<cv::Point>(cv::Point(c + 1, r)),
            main_locs.at<cv::Point>(cv::Point(c + 1, r + 1)),
            main_locs.at<cv::Point>(cv::Point(c, r + 1))};
    };

    for (int r = 0; r < cells_h; ++r)
        for (int c = 0; c < cells_w; ++c)
            if (solution[r][c])
                cv::fillConvexPoly(overlay, corners(r, c),
                                   cv::Scalar(0, 255, 0), cv::LINE_AA);

    cv::addWeighted(overlay, 0.45, base, 0.55, 0.0, overlay);

    // Grid lines along every intersection edge.
    for (int r = 0; r < cells_h + 1; ++r)
        cv::line(overlay,
                 main_locs.at<cv::Point>(cv::Point(0, r)),
                 main_locs.at<cv::Point>(cv::Point(cells_w, r)),
                 cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    for (int c = 0; c < cells_w + 1; ++c)
        cv::line(overlay,
                 main_locs.at<cv::Point>(cv::Point(c, 0)),
                 main_locs.at<cv::Point>(cv::Point(c, cells_h)),
                 cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

    return cv::imwrite(out_path.string(), overlay);
}
```

- [ ] **Step 2: Wire `NG_EXPORT_OVERLAY`**

At the solve-output block (after the `NG_EXPORT_NON` handling at line ~326), add:
```cpp
            if (char const* overlay_path = std::getenv("NG_EXPORT_OVERLAY"))
                render_nonogram_overlay(image, detection.main, result.solution,
                                        overlay_path);
```

- [ ] **Step 3: Build**

Run: `cmake --build build --target nonogram_detector_application -j`
Expected: builds cleanly.

- [ ] **Step 4: End-to-end verify the overlay**

Run:
```bash
cd /home/klimenkov/nonogram_detector
rm -f /tmp/opencode/out_overlay.png
NG_CLUE_FIXES=/tmp/opencode/col21_fixes.txt \
NG_EXPORT_OVERLAY=/tmp/opencode/out_overlay.png \
./build/nonogram_detector_application/nonogram_detector_application \
  nonograms/20180811_114632.jpg 2>&1 | grep -E "consistency|solutions"
```
Expected: `consistent`, `solutions=1 line_solvable=true`. Then confirm the image was written and matches the photo dimensions:
```bash
python3 -c "import cv2; a=cv2.imread('/tmp/opencode/out_overlay.png'); b=cv2.imread('nonograms/20180811_114632.jpg'); print('overlay', a.shape, 'photo', b.shape); assert a.shape==b.shape; assert a.mean()!=b.mean(); print('OK: overlay differs from photo, same size')"
```
Expected: prints the shapes (equal) and `OK: overlay differs from photo, same size`.

- [ ] **Step 5: Commit**

```bash
git add nonogram_detector_application/main.cpp
git commit -m "feat: render solved nonogram overlay onto source photo"
```

---

## Task 3: Validate the .non custom block (round-trip harness)

**Files:**
- Create: `/tmp/opencode/non_roundtrip.cpp` (standalone harness; NOT committed — matches repo idiom)
- Modify: none

- [ ] **Step 1: Write the harness**

```cpp
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// Minimal parser for the "#source/#grid/#cells" custom block appended to a .non.
struct Geo { std::string source; int rows=0, cols=0; std::vector<std::vector<cv::Point>> cells; };

Geo parse_non(char const* path) {
    Geo g;
    std::ifstream in(path);
    std::string line;
    std::string cur;
    bool in_cells = false;
    while (std::getline(in, line)) {
        if (line.rfind("#source:", 0) == 0) { g.source = line.substr(8); }
        else if (line.rfind("#grid:", 0) == 0) {
            std::istringstream ss(line.substr(6));
            char x; ss >> g.rows >> x >> g.cols;
        }
        else if (line.rfind("#cells:", 0) == 0) { in_cells = true; }
        else if (in_cells && !line.empty()) {
            std::istringstream ss(line);
            std::string tok; std::vector<cv::Point> row;
            while (ss >> tok) {
                std::size_t comma = tok.find(',');
                row.push_back({std::stoi(tok.substr(0,comma)), std::stoi(tok.substr(comma+1))});
            }
            g.cells.push_back(row);
        }
    }
    return g;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: harness <non> <photo>\n"; return 2; }
    auto g = parse_non(argv[1]);
    if (g.rows != (int)g.cells.size() || (g.rows>0 && g.cols != (int)g.cells[0].size())) {
        std::cerr << "FAIL: #grid vs #cells mismatch\n"; return 1;
    }
    // #source must be a non-empty relative path.
    if (g.source.empty()) { std::cerr << "FAIL: empty #source\n"; return 1; }
    // Photo must exist relative to CWD (the .non dir in the e2e run is /tmp/opencode,
    // so join with the .non's dir).
    std::size_t slash = std::string(argv[1]).find_last_of('/');
    std::string base = slash==std::string::npos ? "" : std::string(argv[1]).substr(0, slash+1);
    cv::Mat photo = cv::imread(base + g.source);
    if (photo.empty()) { std::cerr << "FAIL: source photo not resolvable: " << g.source << "\n"; return 1; }
    // All points must be within the photo bounds.
    for (auto const& row : g.cells)
        for (auto const& p : row)
            if (p.x < 0 || p.y < 0 || p.x >= photo.cols || p.y >= photo.rows) {
                std::cerr << "FAIL: point out of bounds (" << p.x << "," << p.y << ")\n"; return 1; }
    std::cout << "OK grid=" << g.rows << "x" << g.cols
              << " source=" << g.source << "\n";
    std::cout << "  intersections=" << g.rows * g.cols << ", photo "
              << photo.cols << "x" << photo.rows << "\n";
    return 0;
}
```

- [ ] **Step 2: Build the harness**

Run:
```bash
g++ -std=c++17 -O2 /tmp/opencode/non_roundtrip.cpp -o /tmp/opencode/non_roundtrip $(pkg-config --cflags --libs opencv4)
```
Expected: compiles.

- [ ] **Step 3: Run against the exported .non**

First produce a `.non` (Task 1 output may still exist at `/tmp/opencode/out.non`; if not re-run the Task 1 Step 4 command). Then:
```bash
cd /home/klimenkov/nonogram_detector && /tmp/opencode/non_roundtrip /tmp/opencode/out.non nonograms/20180811_114632.jpg
```
Expected: prints
```
OK grid=<H+1>x<W+1> source=../nonograms/20180811_114632.jpg
  intersections=<H+1>*<W+1>, photo ...x...
```
and exit code 0. This confirms the custom block is well-formed, the photo link resolves, and every stored intersection lies inside the photo.

- [ ] **Step 4: Commit (nothing to commit — harness is /tmp-only)**. Skip commit; note the harness lives under `/tmp/opencode` per repo idiom.

---

## Task 4: Regression gate

- [ ] **Step 1: Full unit tests**

Run: `./build/nonogram_detector_ut/nonogram_detector_ut`
Expected: prints `all tests passed`, exit 0.

- [ ] **Step 2: Full end-to-end (both exports + fix)**

Run:
```bash
cd /home/klimenkov/nonogram_detector
NG_CLUE_FIXES=/tmp/opencode/col21_fixes.txt \
NG_EXPORT_NON=/tmp/opencode/out.non \
NG_EXPORT_OVERLAY=/tmp/opencode/out_overlay.png \
./build/nonogram_detector_application/nonogram_detector_application \
  nonograms/20180811_114632.jpg 2>&1 | grep -E "wrote .non|consistency|solutions"
```
Expected: `wrote .non`, `consistent (row tiles 199 vs col tiles 199)`, `solutions=1 line_solvable=true`.

- [ ] **Step 3: Confirm clean working tree**

Run: `git status --short`
Expected: only the untracked data dirs `digits_marked/` and `nonograms/` (no modified tracked files; all app changes committed in Tasks 1-3).

- [ ] **Step 4: Commit any stragglers** (should be none). If the spec doc was amended during implementation, commit it too.

---

## Self-review notes

- **Spec coverage:** `#source`/`#grid`/`#cells` (Task 1), overlay renderer + `NG_EXPORT_OVERLAY` (Task 2), round-trip validation (Task 3), regression gate (Task 4). The spec's "read/validation (optional, small)" is realized as the /tmp harness in Task 3.
- **Placement rationale:** functions live in `main.cpp` because they mix solver types (`SolutionGrid`) and detector type (`Detection`); keeping them out of the two libraries preserves the current dependency direction (application depends on both; libraries don't depend on each other). Validation thus uses the project's /tmp-harness idiom rather than the UT (which links only the libraries).
- **Type consistency:** `main_locs.at<cv::Point>(cv::Point(col,row))` used consistently across Task 1 (write) and Task 2 (render); `Detection` is not passed directly, only `detection.main` (a `cv::Mat`), keeping both helpers decoupled from the detector type.
