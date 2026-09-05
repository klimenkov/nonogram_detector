# Code Style — `master` branch (2026-09-04)

Scope: the full `master` tree — `nonogram_detector` (lib), `nonogram_detector_application`, `nonogram_detector_test`. Small, self-contained C++/OpenCV codebase (5 headers, 5 sources, 2 entry points). Patterns below are drawn from actual usage, not inferred from a style guide — none exists in the repo.

## Formatting

- **Allman/BSD braces everywhere.** Opening brace always on its own line — functions, classes, `if`/`for`/`while`, and lambdas alike. No exceptions found.
- **4-space indentation** in `.hpp`/`.cpp`; `CMakeLists.txt` uses **tabs** instead ([nonogram_detector/CMakeLists.txt:2](nonogram_detector/CMakeLists.txt#L2)) — the one place indentation style diverges.
- **East `const`**: `cv::Mat const&`, not `const cv::Mat&`, applied consistently across every parameter and local.
- **Multi-parameter function signatures** break one parameter per line, 4-space continuation indent, closing paren on the line with the opening brace:
  ```cpp
  CrossLocsDetector::CrossLocsDetector(
      float const resize_width_height_max,
      int const threshold_block_size,
      ...)
  ```
- **Two blank lines between top-level function definitions** in `.cpp` files; **one blank line** after `namespace ng\n{` and before the matching `}`. Namespace contents are *not* indented.
- Trailing **double blank line before the closing namespace brace** is common but not universal.

## Naming

| Kind | Convention | Example |
|---|---|---|
| Types (classes/structs) | `PascalCase` | `CrossLocsDetector`, `PointCompare`, `WindowTrackbarDetector` |
| Free functions, methods, locals | `snake_case` | `find_kernel_loc`, `cross_locs_main_mat` |
| Const data members (set once, ctor-injected config) | `M_UPPER_SNAKE` | `M_RESIZE_WIDTH_HEIGHT_MAX` ([cross_locs_detector.hpp:36-41](nonogram_detector/include/cross_locs_detector.hpp#L36)) |
| Mutable data members | `m_lower_snake` | `m_window_name`, `m_threshold_c` ([nonogram_detector_test/main.cpp:118-123](nonogram_detector_test/main.cpp#L118)) |
| `static const` class constants (not ctor config) | `UPPER_SNAKE`, no prefix | `INDICES_DELTA_UP`, `INDICES_DELTAS` ([cross_locs_detector.hpp:43-48](nonogram_detector/include/cross_locs_detector.hpp#L43)) |
| Namespace | lower, single word | `ng` |

The `M_`/`m_` split is a deliberate, consistently-applied signal: **`M_` means immutable configuration baked in at construction; `m_` means state that changes over the object's lifetime.** Worth documenting explicitly since it's non-standard and easy for a newcomer to misread as inconsistency.

## Includes

Three groups, blank line between each, alphabetized within a group:
1. Standard library (`<vector>`, `<map>`, ...)
2. Project headers, quoted (`"point_compare.hpp"`)
3. Third-party, angle-bracketed (`<opencv2/opencv.hpp>`)

`#pragma once` is used in every header **except** [image_operations.hpp](nonogram_detector/include/image_operations.hpp#L1), which has no include guard at all — the one gap in an otherwise consistent convention.

## API / control-flow idioms

- **Multi-value return via `std::tuple`/`std::pair` + `std::tie`**, with a leading `bool` "found/succeeded" flag as the idiomatic error channel — no exceptions, no `std::optional` (9 uses each in `cross_locs_detector.cpp` / `grid_detector.cpp`; see [image_operations.hpp:29-34](nonogram_detector/include/image_operations.hpp#L29) for the pattern's declaration site). The codebase predates or simply avoids C++17 structured bindings even where they'd read more cleanly than `std::tie`.
- **`auto const` by default** for any local that isn't reassigned; plain `auto`/typed only when mutation follows.
- **Anonymous scoping blocks (`{ ... }`)** used liberally inside functions to bound the lifetime of temporaries and keep locals from leaking across unrelated sections of a long function — a manual substitute for smaller functions (e.g. [cross_locs_detector.cpp:44-79](nonogram_detector/src/cross_locs_detector.cpp#L44)).
- **Static helper methods over free functions inside a class**: `CrossLocsDetector` holds the constructor-injected config as the only non-static state; nearly every private method is `static` and takes all its inputs as parameters. The class is really a namespace with bundled config, not an object with behavior.
- **Header comments reference parameters with `<angle brackets>`** instead of Doxygen (`@param`), e.g. `// <indices_init> must correspond with <cross_locs_init>` ([cross_locs_detector.hpp:59](nonogram_detector/include/cross_locs_detector.hpp#L59)). Comments are sparse and reserved for non-obvious contracts, invariants, or return-value semantics — not routine "what this does" narration.
- **`assert()`** for precondition checks in low-level helpers ([masks.cpp](nonogram_detector/src/masks.cpp)), not exceptions.

## Notable inconsistencies / smells

1. **`grid_detector.hpp`/`.cpp` duplicate `cross_locs_detector.hpp`/`.cpp` almost byte-for-byte but the class inside is still named `CrossLocsDetector`** ([grid_detector.hpp:15](nonogram_detector/include/grid_detector.hpp#L15)) — reads as an in-progress rename or copy-paste split that was never finished. They aren't wired into [CMakeLists.txt](nonogram_detector/CMakeLists.txt), so they don't currently collide at link time, but as-is they're dead, drifting duplicates (the two `.cpp` files have already diverged slightly — `grid_detector.cpp`'s `detect()` dropped the debug `cv::imshow`/`waitKey` block and the `cell_loc_found` early-return that `cross_locs_detector.cpp` still has).
2. **Debug/visualization code is left in commented out**, sometimes extensively (e.g. the `//// Draw` block in `get_cross_locs_map`, [cross_locs_detector.cpp:198-206](nonogram_detector/src/cross_locs_detector.cpp#L198)). It's consistent enough to read as a working habit (scratch code kept "just in case") rather than accidental cruft, but it adds real reading overhead.
3. **`std::cout` debug logging is embedded directly in library code** ([cross_locs_detector.cpp:52](nonogram_detector/src/cross_locs_detector.cpp#L52), `detect()`, `get_cross_locs_main_mat()`), with no logging abstraction or verbosity flag — the library always prints to stdout and pops OpenCV windows (`cv::imshow`/`cv::waitKey(0)`) as a side effect of `detect()`. This makes the library unusable headless/non-interactively as-is.
4. **Hardcoded, developer-specific Windows paths** in both entry points, e.g. `R"(C:\Users\klimenkov\Desktop\nonograms\...)"` ([main.cpp:107](nonogram_detector_application/main.cpp#L107)) — no CLI args, no config file, so the app only runs unmodified on the original author's machine.
5. **`cmake_minimum_required(VERSION 2.8)`** with no `CMAKE_CXX_STANDARD` set anywhere, despite the code using C++11/14 features (`auto`, lambdas, range-for) — the standard is presumably coming from a global toolchain default rather than being declared by the build.
6. Two overloads of `get_mask_cross` (with/without `margin`, [masks.hpp:12](nonogram_detector/include/masks.hpp#L12) vs [masks.hpp:20](nonogram_detector/include/masks.hpp#L20)) differ only by arity rather than name — consistent with the "static free-function-ish API" style elsewhere, but arity-only overloading here relies on the reader remembering which call site wants the margin variant.

## Suggested improvements

- **Resolve the `grid_detector` duplication** — either finish the rename (drop `cross_locs_detector.*`) or delete the orphaned copy; right now both exist, neither is referenced from `CMakeLists.txt`, and they're already silently diverging.
- **Add `#pragma once` to `image_operations.hpp`** for consistency with every other header.
- **Replace the ad-hoc `bool`-first-tuple + `std::tie` pattern with `std::optional<T>`** (or a small named result struct) where only one value is being conditionally returned — clearer at call sites than an unused `cv::Point(-1, -1)` sentinel, and drops the boilerplate `std::tie` unpack. Where genuinely multiple values return together, C++17 structured bindings (`auto const [a, b] = ...`) would remove the pre-declared-then-`tie`'d locals throughout `cross_locs_detector.cpp`.
- **Pull `std::cout`/`cv::imshow` debug output out of the library** behind a verbosity flag or callback, so `nonogram_detector` can be linked into something other than an interactive debug tool.
- **Delete rather than comment out** dead visualization code, or gate it behind a `#ifdef NG_DEBUG_DRAW` if it's meant to be flipped back on regularly — right now it's indistinguishable at a glance from abandoned code.
- **Move hardcoded paths to `argv`/a config file** in both `main()`s so the app is runnable outside the original dev machine.
- **Set `CMAKE_CXX_STANDARD` explicitly** and bump `cmake_minimum_required` off 2.8 (EOL for a decade) so the build doesn't depend on whatever default the invoking toolchain happens to pick.
