# .non Photo-Link + Grid-Geometry + Overlay Rendering

**Date:** 2026-09-03

## Goal

When the application exports a solved nonogram to webpbn `.non` format, it must
also:

1. Record a reference to the source photo (so the puzzle can be located from the
   `.non` file later), and
2. Record the grid geometry — the pixel coordinates of the detected grid — so a
   renderer can draw the nonogram directly onto that photo.

Additionally, implement an overlay renderer that uses the recorded geometry (or
the live detection) to fill the solved cells onto the original photo and save an
output image.

This makes the `.non` file self-describing for rendering while remaining a valid,
parseable webpbn file.

## Background / geometry facts

- `Detection.main` is a `CV_32SC2` matrix holding the **original-image pixel
  coordinates** of every grid intersection of the playing area. The detection
  pipeline divides resized-space coordinates by the resize scale (see
  `cross_locs_detector.cpp`), so these are already in full-resolution photo
  pixels — no further scaling is needed for rendering.
- The matrix has `(H+1)` rows × `(W+1)` cols of intersections, giving an `H × W`
  cell grid that maps 1:1 to the solved `SolutionGrid` (rows `0..H`, cols
  `0..W`).
- A cell at grid position `(r, c)` is the quadrilateral bounded by the four
  adjacent intersections:
  - top-left `(c,   r  )`
  - top-right `(c+1, r  )`
  - bottom-right `(c+1, r+1)`
  - bottom-left `(c,   r+1)`
  (indexing `main.at<cv::Point>(cv::Point(col, row))`, consistent with the
  existing cell-warping in `get_cell_warped_images_vector`.)

## Approach (chosen)

Append a custom block to the `.non` file, all lines prefixed with `#`. Standard
webpbn readers that tolerate/ignore `#` comment lines remain compatible; the
standard fields (`title`, `width`, `height`, `rows`, `columns`, `goal`) are
unchanged.

### Custom block format

Appended after the `goal` line (or after `columns` if no goal is available):

```
#source: <relative-path-to-photo>
#grid: <H+1>x<W+1>
#cells:
<x>,<y> <x>,<y> ...   (one line per intersection row, W+1 points per line,
                        H+1 lines total)
```

- `#source:` — path to the source photo, **relative to the `.non` file's parent
  directory**. If a path cannot be made relative (different drive on Windows),
  fall back to the absolute path.
- `#grid: <rows>x<cols>` — the intersection matrix dimensions; degenerate guard
  (must equal parsed point rows/cols) for validation on read.
- `#cells:` — the serialized `Detection.main` (flattened row-major, space
  separated `x,y` pairs, one line per intersection row for readability).

Keeping the standard fields first and the custom block last means a standard
parser can stop at the end of `goal`/`columns` and never misinterpret the extra
lines.

## Components

### 1. `write_non_file` (extend existing function in `main.cpp`)

Signature (unchanged externally):

```cpp
void write_non_file(
    std::filesystem::path const& path,
    ng::ClueConstraints const& constraints,
    ng::SolutionGrid const& solution,
    ng::Detection const& detection,      // NEW: for intersections + H/W
    std::filesystem::path const& image_path); // NEW: source, for relative path
```

Behavior additions over today:
- Resolve `#source` as the relative path from `path.parent_path()` to
  `image_path`; fall back to absolute.
- Emit `#grid` and `#cells` from `detection.main`.
- Guard against an empty `detection.main` (write only the standard fields).

### 2. Overlay renderer

New free function in `main.cpp` (or a small helper), independent of the file
format so it can also run from live detection:

```cpp
bool render_nonogram_overlay(
    cv::Mat const& photo,                 // original image
    cv::Mat const& main_locs,             // Detection.main intersections
    ng::SolutionGrid const& solution,     // 0/1 per cell
    std::filesystem::path const& out_path);
```

Behavior:
- Copy the source photo.
- For each cell `(r, c)` with `solution[r][c] == 1`, fill `cv::fillConvexPoly` on
  the cell quadrilateral with a semi-transparent overlay (e.g. a strong color
  blended onto a copy of the photo).
- Draw the grid lines connecting the intersections (thin lines over all
  quadrilaterals) so cell boundaries are visible.
- `cv::imwrite` the result to `out_path`; return whether it succeeded.

### 3. Wiring in `main.cpp` solve path

- `NG_EXPORT_NON=<path>` already writes the `.non` (base fields). After this
  change it also writes the custom block. The export already requires
  `result.solved`; keep that condition (we render from the solution). Since
  rendering needs the photo even when `NG_EXPORT_NON` is unset, add a separate
  `NG_EXPORT_OVERLAY=<path>` env that renders the overlay from the live
  detection + solution and writes the image. Both are independent so the user
  can write `.non` only, overlay only, or both.

### 4. Read/validation (optional, small)

Add a tiny reader used only by the UT to round-trip the custom block (parse
`#source`/`#grid`/`#cells`, verify the intersection array matches the written
one). This is a test-only helper, not a general `.non` loader.

## Testing

- **Unit test** (`nonogram_detector_ut`): extend `clue_corrector_test.cpp` or add
  a small test that builds a fake `Detection` with a known small intersection
  grid + a small solution, calls `write_non_file`, reparses the custom block, and
  asserts the `#source` path, `#grid` dims, and all `#cells` points round-trip.
- **End-to-end**: run the app on `nonograms/20180811_114632.jpg` with
  `NG_CLUE_FIXES` + `NG_EXPORT_NON=/tmp/opencode/out.non` and
  `NG_EXPORT_OVERLAY=/tmp/opencode/out_overlay.png`. Assert:
  - `out.non` is written, standard fields parseable, custom block present.
  - `out_overlay.png` exists and is non-trivial (readable by OpenCV).
  - Overlay dimensions match the original photo.

## Out of scope

- General `.non` *import* / re-solving from a `.non` file.
- Interactive editing of the rendered overlay.
- Persisting clue-strip (`top`/`left`) geometry (only playing-area `main` is
  needed to render the solved grid).
