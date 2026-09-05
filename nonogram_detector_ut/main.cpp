#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "cross_locs_detector.hpp"
#include "decode.hpp"
#include "digit_recognizer.hpp"
#include "grid_smooth_fit.hpp"
#include "image_operations.hpp"
#include "masks.hpp"


// Defined in digit_recognizer_test.cpp
int run_digit_recognizer_tests();

#ifdef NG_ENABLE_SOLVER
int run_solver_tests();   // defined in solver_test.cpp
int run_clue_corrector_tests();   // defined in clue_corrector_test.cpp
int run_non_file_tests();   // defined in non_file_test.cpp
#endif


namespace
{

// Builds a synthetic nonogram-style image in memory: a main grid plus top and
// left clue strips, rendered as black lines on white. The cell side length is
// chosen so that, after resizing to <resize_max>, the cells stay within the
// detector's side-length search window.
cv::Mat make_grid(int const cols, int const rows, int const cell, int& hint_cell)
{
    int const clues = 2;          // clue strips on top and left (in cells)
    int const top_h = clues * cell;
    int const left_w = clues * cell;
    int const H = top_h + rows * cell;
    int const W = left_w + cols * cell;
    int const line = 2;

    cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));

    auto draw_grid = [&](int bx, int by, int nc, int nr)
    {
        for (int i = 0; i <= nc; ++i)
            cv::line(img, cv::Point(bx + i * cell, by), cv::Point(bx + i * cell, by + nr * cell), 0, line);
        for (int j = 0; j <= nr; ++j)
            cv::line(img, cv::Point(bx, by + j * cell), cv::Point(bx + nc * cell, by + j * cell), 0, line);
    };

    draw_grid(left_w, 0, cols, clues);       // top clue strip
    draw_grid(0, top_h, clues, rows);        // left clue strip
    draw_grid(left_w, top_h, cols, rows);    // main grid

    hint_cell = cell;
    return img;
}

bool run_case(int const cols, int const rows, int const cell, int const resize_max, int expect_cols, int expect_rows)
{
    int hint_cell = 0;
    auto const img = make_grid(cols, rows, cell, hint_cell);

    // Largest side of the resized image must keep cells in [5, 50].
    int const resized_cell = static_cast<int>(cell * (static_cast<float>(resize_max) /
        static_cast<float>(std::max(img.cols, img.rows))));
    if (resized_cell < 5 || resized_cell > 50)
    {
        std::cerr << "  [skip] resized_cell=" << resized_cell << " out of search window\n";
        return true; // skip, not a failure
    }

    ng::CrossLocsDetector detector(resize_max, 15, 10.0, 5, 50, 0.9);
    auto const detection = detector.detect(img);

    if (!detection.found)
    {
        std::cerr << "  [FAIL] not found (cols=" << cols << " rows=" << rows << " cell=" << cell << " resize=" << resize_max << ")\n";
        return false;
    }

    // The main cross_locs matrix is cell_corners x cell_corners; number of cells
    // is (corners - 1) per side. The top/left clue strips add to the count.
    int const found_cells_w = detection.main.cols - 1;
    int const found_cells_h = detection.main.rows - 1;

    // Allow the main region to include the clue strips (they are contiguous).
    if (found_cells_w < expect_cols || found_cells_h < expect_rows)
    {
        std::cerr << "  [FAIL] grid too small: got " << found_cells_w << "x"
                  << found_cells_h << " expected at least " << expect_cols
                  << "x" << expect_rows << "\n";
        return false;
    }

    std::cout << "  [ok] cols=" << cols << " rows=" << rows << " cell=" << cell
              << " resize=" << resize_max << " -> grid " << found_cells_w << "x"
              << found_cells_h << "\n";
    return true;
}

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

// The real detection path rather than an ideal quadratic: a synthetic
// thresholded binary image (CV_8U, 0 background / 1 foreground) with a
// 2px-thick horizontal band (rows [29,30]) and a 2px-thick vertical band
// (cols [24,25]). The true intersection center sits at a half-pixel offset:
// (24.5, 29.5). find_kernel_loc with the plain cross mask (the same call
// shape the BFS in get_cross_locs_map uses, default anchor = kernel center)
// must locate the cross, and the paraboloid refinement on the filter2D
// response must recover the subpixel center. Measured: the response peaks
// flat (1.0) over the 2x2 plateau around the true center and the fit lands
// exactly on (24.5, 29.5); the integer peak (24,29) is displaced by ~0.5 px.
bool test_find_kernel_loc_subpixel_cross()
{
    std::cout << "case: find_kernel_loc recovers subpixel cross center\n";

    cv::Mat img(60, 60, CV_8U, cv::Scalar(0));
    img(cv::Rect(0, 29, img.cols, 2)) = 1;   // horizontal band, rows [29,30]
    img(cv::Rect(24, 0, 2, img.rows)) = 1;   // vertical band, cols [24,25]

    int const mask_length = 15;   // odd, as get_mask_cross requires
    cv::Mat mask_cross;
    int mask_cross_perimeter;
    std::tie(mask_cross, mask_cross_perimeter) = ng::get_mask_cross(mask_length);

    bool found = false;
    cv::Point2f refined(-1.0f, -1.0f);
    std::tie(found, refined) = ng::find_kernel_loc(
        img, mask_cross, mask_cross_perimeter, 0.5, cv::Point(-1, -1));

    cv::Point2f const true_center(24.5f, 29.5f);
    if (!found)
    {
        std::cerr << "  [FAIL] cross not found\n";
        return false;
    }

    float const tolerance = 0.01f;
    if (std::fabs(refined.x - true_center.x) > tolerance ||
        std::fabs(refined.y - true_center.y) > tolerance)
    {
        std::cerr << "  [FAIL] refined=" << refined << " expected ~"
                  << true_center << " (tolerance " << tolerance << ")\n";
        return false;
    }

    std::cout << "  [ok] found=" << found << " refined=" << refined << "\n";
    return true;
}

// A flat response — peak and all four neighbors equal — has a zero second
// difference; refine_peak_loc must return the integer peak unchanged (the
// denom <= 1e-6 guard).
bool test_refine_peak_loc_flat_response()
{
    std::cout << "case: refine_peak_loc returns int peak on flat response\n";

    cv::Mat resp(21, 31, CV_32F, cv::Scalar(1.0f));
    cv::Point const int_peak(15, 10);   // interior, all four neighbors present
    cv::Point2f const refined = ng::refine_peak_loc(resp, int_peak);

    if (refined != cv::Point2f(15.0f, 10.0f))
    {
        std::cerr << "  [FAIL] refined=" << refined << " expected (15,10)\n";
        return false;
    }

    std::cout << "  [ok] refined=" << refined << "\n";
    return true;
}

// Anti-aliased cross with an analytically known subpixel center: each pixel's
// ink coverage is the exact overlap of its footprint with the 2-px-wide
// vertical/horizontal bars, so the per-axis baseline-subtracted ink centroids
// equal the bar centers exactly. The input location is the approximate
// (response-peak stage) position; the sentinel entry must be left alone.
bool test_refine_cross_locs_ink_subpixel()
{
    std::cout << "case: refine_cross_locs_ink recovers subpixel cross center\n";

    float const cx = 24.3f, cy = 29.7f;   // true center, subpixel
    float const half = 1.0f;              // bars are 2 px wide
    cv::Mat gray(60, 60, CV_8U, cv::Scalar(255));
    auto const overlap = [](double a0, double a1, double b0, double b1) {
        return std::max(0.0, std::min(a1, b1) - std::max(a0, b0));
    };
    for (int y = 0; y < gray.rows; ++y)
    {
        for (int x = 0; x < gray.cols; ++x)
        {
            double const v = overlap(x - 0.5, x + 0.5, cx - half, cx + half);
            double const h = overlap(y - 0.5, y + 0.5, cy - half, cy + half);
            double const ink = std::max(v, h);
            gray.at<uchar>(y, x) = static_cast<uchar>(255 - 200 * ink);
        }
    }

    cv::Mat locs(1, 2, CV_32FC2, cv::Scalar(-1.0f, -1.0f));
    locs.at<cv::Point2f>(0, 0) = cv::Point2f(24.0f, 30.0f);  // approx position

    ng::refine_cross_locs_ink(gray, locs, 10);

    cv::Point2f const refined = locs.at<cv::Point2f>(0, 0);
    cv::Point2f const sentinel = locs.at<cv::Point2f>(0, 1);
    // Union ink coverage (max of the two bars) is not exactly additive at
    // partially-covered edge pixels, which bounds the recoverable accuracy
    // at ~0.015 px; 0.05 leaves margin.
    float const tolerance = 0.05f;
    if (std::fabs(refined.x - cx) > tolerance ||
        std::fabs(refined.y - cy) > tolerance ||
        sentinel != cv::Point2f(-1.0f, -1.0f))
    {
        std::cerr << "  [FAIL] refined=" << refined << " expected ~(" << cx
                  << "," << cy << ") sentinel=" << sentinel << "\n";
        return false;
    }

    std::cout << "  [ok] refined=" << refined << "\n";
    return true;
}

// Blank paper around the location: no line ink, so the location must be kept.
bool test_refine_cross_locs_ink_empty_window()
{
    std::cout << "case: refine_cross_locs_ink keeps location on empty paper\n";

    cv::Mat gray(60, 60, CV_8U, cv::Scalar(255));
    cv::Mat locs(1, 1, CV_32FC2);
    locs.at<cv::Point2f>(0, 0) = cv::Point2f(24.0f, 30.0f);

    ng::refine_cross_locs_ink(gray, locs, 10);

    cv::Point2f const refined = locs.at<cv::Point2f>(0, 0);
    if (refined != cv::Point2f(24.0f, 30.0f))
    {
        std::cerr << "  [FAIL] refined=" << refined << " expected (24,30)\n";
        return false;
    }

    std::cout << "  [ok] refined=" << refined << "\n";
    return true;
}

// A dark blob in the window but outside the refinement band: the ink does not
// belong to the cross, so the location must be kept on both axes.
bool test_refine_cross_locs_ink_rejects_non_cross_ink()
{
    std::cout << "case: refine_cross_locs_ink keeps location near non-cross ink\n";

    cv::Mat gray(60, 60, CV_8U, cv::Scalar(255));
    gray(cv::Rect(34, 24, 6, 6)) = 40;   // blob ~10 px right of the location
    cv::Mat locs(1, 1, CV_32FC2);
    locs.at<cv::Point2f>(0, 0) = cv::Point2f(24.0f, 27.0f);

    ng::refine_cross_locs_ink(gray, locs, 10);

    cv::Point2f const refined = locs.at<cv::Point2f>(0, 0);
    if (refined != cv::Point2f(24.0f, 27.0f))
    {
        std::cerr << "  [FAIL] refined=" << refined << " expected (24,27)\n";
        return false;
    }

    std::cout << "  [ok] refined=" << refined << "\n";
    return true;
}

// A bold (6 px wide) cross with the previous stage's position 2.5 px off the
// line center: one refinement pass truncates the line at the band edge and
// under-corrects (~0.5 px short), so the refinement must iterate — each pass
// re-centring the band on the refined location — to converge on the true
// center.
bool test_refine_cross_locs_ink_bold_line_converges()
{
    std::cout << "case: refine_cross_locs_ink converges on a bold off-center cross\n";

    float const cx = 24.0f, cy = 30.0f;   // true center of the 6 px cross
    float const half = 3.0f;
    cv::Mat gray(60, 60, CV_8U, cv::Scalar(255));
    auto const overlap = [](double a0, double a1, double b0, double b1) {
        return std::max(0.0, std::min(a1, b1) - std::max(a0, b0));
    };
    for (int y = 0; y < gray.rows; ++y)
    {
        for (int x = 0; x < gray.cols; ++x)
        {
            double const v = overlap(x - 0.5, x + 0.5, cx - half, cx + half);
            double const h = overlap(y - 0.5, y + 0.5, cy - half, cy + half);
            double const ink = std::max(v, h);
            gray.at<uchar>(y, x) = static_cast<uchar>(255 - 200 * ink);
        }
    }

    cv::Mat locs(1, 1, CV_32FC2);
    locs.at<cv::Point2f>(0, 0) = cv::Point2f(24.0f, 27.5f);  // 2.5 px off in y

    ng::refine_cross_locs_ink(gray, locs, 10);

    cv::Point2f const refined = locs.at<cv::Point2f>(0, 0);
    float const tolerance = 0.05f;
    if (std::fabs(refined.x - cx) > tolerance ||
        std::fabs(refined.y - cy) > tolerance)
    {
        std::cerr << "  [FAIL] refined=" << refined << " expected ~(" << cx
                  << "," << cy << ")\n";
        return false;
    }

    std::cout << "  [ok] refined=" << refined << "\n";
    return true;
}

// SNAP: synthetic smooth-but-curved grid (quadratic bow + skew) with per-cross
// noise. The global smooth fit should pull crossings close to the true smooth
// surface, clearly better than the noisy input.
bool test_grid_smooth_fit_approach1_recovers_smooth_grid()
{
    std::cout << "case: grid_smooth_fit_approach1 recovers a smooth curved grid\n";

    int const R = 10, C = 12;           // (R+1) x (C+1) crossings
    double const scale = 30.0;          // crossing spacing
    double const bow = 0.0008;          // quadratic curvature (sag over grid)
    double const skew = 0.06;           // shear in x as a function of row
    double const noise = 0.8;           // per-cross noise (px)

    cv::Mat locs(R + 1, C + 1, CV_32FC2);
    auto rng = [seed = 12345u]() mutable {
        seed = seed * 1103515245u + 12345u;
        return static_cast<double>(seed % 100000u) / 100000.0 - 0.5;
    };
    for (int r = 0; r <= R; ++r)
    {
        for (int c = 0; c <= C; ++c)
        {
            double const x = c * scale + r * skew * scale;   // shear
            double const y = r * scale + bow * (c * (C - c)) * scale * scale * 100.0;
            locs.at<cv::Point2f>(r, c) =
                cv::Point2f(static_cast<float>(x + noise * rng()),
                            static_cast<float>(y + noise * rng()));
        }
    }

    auto const eval_true = [&](int r, int c, double& tx, double& ty) {
        tx = c * scale + r * skew * scale;
        ty = r * scale + bow * (c * (C - c)) * scale * scale * 100.0;
    };

    // Residual of the noisy input vs the true smooth surface.
    auto const resid_of = [&](cv::Mat const& m) {
        double sum = 0.0;
        int n = 0;
        for (int r = 0; r <= R; ++r)
            for (int c = 0; c <= C; ++c)
            {
                double tx, ty;
                eval_true(r, c, tx, ty);
                cv::Point2f const p = m.at<cv::Point2f>(r, c);
                sum += std::sqrt((p.x - tx) * (p.x - tx) + (p.y - ty) * (p.y - ty));
                ++n;
            }
        return sum / n;
    };

    double const before = resid_of(locs);
    cv::Mat const snapped = ng::grid_smooth_fit_approach1(locs, 3, 2);
    double const after = resid_of(snapped);

    if (after > before * 0.5)
    {
        std::cerr << "  [FAIL] mean-resid before=" << before << " after=" << after
                  << " (expected improvement by fit)\n";
        return false;
    }

    std::cout << "  [ok] mean-resid before=" << before << " after=" << after << "\n";
    return true;
}

// SNAP: same synthetic smooth-but-curved grid (quadratic bow + skew) with
// per-cross noise. Approach 2 fits each grid line independently (a polynomial
// per row/column, no shared cross-family model). It should still reduce the
// noise residual relative to the raw input, though it cannot borrow strength
// across lines the way approach 1 does.
bool test_grid_smooth_fit_approach2_recovers_smooth_grid()
{
    std::cout << "case: grid_smooth_fit_approach2 recovers a smooth curved grid\n";

    int const R = 10, C = 12;           // (R+1) x (C+1) crossings
    double const scale = 30.0;          // crossing spacing
    double const bow = 0.0008;          // quadratic curvature (sag over grid)
    double const skew = 0.06;           // shear in x as a function of row
    double const noise = 0.8;           // per-cross noise (px)

    cv::Mat locs(R + 1, C + 1, CV_32FC2);
    auto rng = [seed = 12345u]() mutable {
        seed = seed * 1103515245u + 12345u;
        return static_cast<double>(seed % 100000u) / 100000.0 - 0.5;
    };
    for (int r = 0; r <= R; ++r)
    {
        for (int c = 0; c <= C; ++c)
        {
            double const x = c * scale + r * skew * scale;   // shear
            double const y = r * scale + bow * (c * (C - c)) * scale * scale * 100.0;
            locs.at<cv::Point2f>(r, c) =
                cv::Point2f(static_cast<float>(x + noise * rng()),
                            static_cast<float>(y + noise * rng()));
        }
    }

    auto const eval_true = [&](int r, int c, double& tx, double& ty) {
        tx = c * scale + r * skew * scale;
        ty = r * scale + bow * (c * (C - c)) * scale * scale * 100.0;
    };

    auto const resid_of = [&](cv::Mat const& m) {
        double sum = 0.0;
        int n = 0;
        for (int r = 0; r <= R; ++r)
            for (int c = 0; c <= C; ++c)
            {
                double tx, ty;
                eval_true(r, c, tx, ty);
                cv::Point2f const p = m.at<cv::Point2f>(r, c);
                sum += std::sqrt((p.x - tx) * (p.x - tx) + (p.y - ty) * (p.y - ty));
                ++n;
            }
        return sum / n;
    };

    double const before = resid_of(locs);
    cv::Mat const snapped = ng::grid_smooth_fit_approach2(locs, 2);
    double const after = resid_of(snapped);

    if (after > before * 0.5)
    {
        std::cerr << "  [FAIL] mean-resid before=" << before << " after=" << after
                  << " (expected improvement by fit)\n";
        return false;
    }

    std::cout << "  [ok] mean-resid before=" << before << " after=" << after << "\n";
    return true;
}

// SNAP: same synthetic smooth-but-curved grid (quadratic bow + skew) with
// per-cross noise. Approach 3 fits x(r,c) and y(r,c) as one global bivariate
// polynomial over the rectangular index domain, so every crossing contributes
// to a single shared surface (rigid row/col coupling). A biquadratic (order 2)
// exactly spans this grid's true form, so it should recover it from noise.
bool test_grid_smooth_fit_approach3_recovers_smooth_grid()
{
    std::cout << "case: grid_smooth_fit_approach3 recovers a smooth curved grid\n";

    int const R = 10, C = 12;           // (R+1) x (C+1) crossings
    double const scale = 30.0;          // crossing spacing
    double const bow = 0.0008;          // quadratic curvature (sag over grid)
    double const skew = 0.06;           // shear in x as a function of row
    double const noise = 0.8;           // per-cross noise (px)

    cv::Mat locs(R + 1, C + 1, CV_32FC2);
    auto rng = [seed = 12345u]() mutable {
        seed = seed * 1103515245u + 12345u;
        return static_cast<double>(seed % 100000u) / 100000.0 - 0.5;
    };
    for (int r = 0; r <= R; ++r)
    {
        for (int c = 0; c <= C; ++c)
        {
            double const x = c * scale + r * skew * scale;   // shear
            double const y = r * scale + bow * (c * (C - c)) * scale * scale * 100.0;
            locs.at<cv::Point2f>(r, c) =
                cv::Point2f(static_cast<float>(x + noise * rng()),
                            static_cast<float>(y + noise * rng()));
        }
    }

    auto const eval_true = [&](int r, int c, double& tx, double& ty) {
        tx = c * scale + r * skew * scale;
        ty = r * scale + bow * (c * (C - c)) * scale * scale * 100.0;
    };

    auto const resid_of = [&](cv::Mat const& m) {
        double sum = 0.0;
        int n = 0;
        for (int r = 0; r <= R; ++r)
            for (int c = 0; c <= C; ++c)
            {
                double tx, ty;
                eval_true(r, c, tx, ty);
                cv::Point2f const p = m.at<cv::Point2f>(r, c);
                sum += std::sqrt((p.x - tx) * (p.x - tx) + (p.y - ty) * (p.y - ty));
                ++n;
            }
        return sum / n;
    };

    double const before = resid_of(locs);
    cv::Mat const snapped = ng::grid_smooth_fit_approach3(locs, 2);
    double const after = resid_of(snapped);

    if (after > before * 0.5)
    {
        std::cerr << "  [FAIL] mean-resid before=" << before << " after=" << after
                  << " (expected improvement by fit)\n";
        return false;
    }

    std::cout << "  [ok] mean-resid before=" << before << " after=" << after << "\n";
    return true;
}

}

bool test_prepare_input_accepts_thin_digit()
{
    // A 20x20 cell with a 2 px-wide vertical stem (like the thin "1" that used
    // to be rejected): prepare_input must produce a blob.
    cv::Mat cell(20, 20, CV_8UC3, cv::Scalar(205, 205, 205));
    cv::rectangle(cell, cv::Rect(9, 3, 2, 11), cv::Scalar(40, 40, 40), cv::FILLED);
    cv::Mat blob;
    if (!ng::DigitRecognizer::prepare_input(cell, blob))
    {
        std::cout << "FAIL: thin digit rejected by prepare_input\n";
        return false;
    }
    return true;
}

bool test_prepare_input_rejects_speck()
{
    // A tiny 1x2 speck (2 fg px < area gate) must be treated as empty.
    cv::Mat cell(20, 20, CV_8UC3, cv::Scalar(205, 205, 205));
    cv::rectangle(cell, cv::Rect(10, 10, 1, 2), cv::Scalar(40, 40, 40), cv::FILLED);
    cv::Mat blob;
    if (ng::DigitRecognizer::prepare_input(cell, blob))
    {
        std::cout << "FAIL: speck accepted by prepare_input\n";
        return false;
    }
    return true;
}

bool test_prepare_input_rejects_empty()
{
    cv::Mat cell(20, 20, CV_8UC3, cv::Scalar(205, 205, 205));
    cv::Mat blob;
    if (ng::DigitRecognizer::prepare_input(cell, blob))
    {
        std::cout << "FAIL: empty cell accepted by prepare_input\n";
        return false;
    }
    return true;
}

bool test_guard_implausible_split_falls_back()
{
    // Aligned 42 on a 30-wide puzzle: implausible -> whole-cell read.
    int const d = ng::resolve_two_digit(
        42, 7, 0.95, 0.9, 0.9, 30, 0.3, 0.9);
    if (d != 7)
    {
        std::cout << "FAIL: implausible split did not fall back (got " << d << ")\n";
        return false;
    }
    return true;
}

bool test_guard_whole_cell_wins_on_counter_fp()
{
    // Counter FP: single "5" split confidently wrongly? -> halves not both
    // confident (conf < kSplitConfidenceMin=0.3) but whole-cell is a high-conf 5
    // -> prefer whole.
    int const d = ng::resolve_two_digit(
        18, 5, 0.95, 0.15, 0.15, 30, 0.3, 0.9);
    if (d != 5)
    {
        std::cout << "FAIL: high-conf whole-cell not preferred (got " << d << ")\n";
        return false;
    }
    return true;
}

bool test_guard_genuine_two_digit_kept()
{
    // Genuine 13: whole-cell read is low-conf garbage; halves confident.
    int const d = ng::resolve_two_digit(
        13, 1, 0.4, 0.9, 0.9, 30, 0.3, 0.9);
    if (d != 13)
    {
        std::cout << "FAIL: genuine two-digit split not kept (got " << d << ")\n";
        return false;
    }
    return true;
}

bool test_guard_zero_means_empty()
{
    // A standalone 0 is never a valid nonogram clue; it must be treated as
    // empty (-1) so it cannot inject a phantom clue into the solver.
    if (ng::sanitize_clue_digit(0) != -1)
    {
        std::cout << "FAIL: standalone 0 not treated as empty\n";
        return false;
    }
    if (ng::sanitize_clue_digit(5) != 5)
    {
        std::cout << "FAIL: genuine single digit altered\n";
        return false;
    }
    if (ng::sanitize_clue_digit(10) != 10)
    {
        std::cout << "FAIL: two-digit value containing a zero was altered\n";
        return false;
    }
    if (ng::sanitize_clue_digit(-1) != -1)
    {
        std::cout << "FAIL: empty sentinel altered\n";
        return false;
    }
    return true;
}

bool test_resize_empty_image()
{
    auto const res = ng::resize(cv::Mat(), 1200);
    if (!res.first.empty() || res.second != 0.0f)
    {
        std::cout << "FAIL: resize on empty image did not return empty result\n";
        return false;
    }
    return true;
}

bool test_get_cell_warped_images_vector_empty_or_small()
{
    cv::Mat const dummy = cv::Mat::zeros(100, 100, CV_8UC3);
    if (!ng::get_cell_warped_images_vector(dummy, cv::Mat()).empty())
    {
        std::cout << "FAIL: get_cell_warped_images_vector did not return empty for empty cross_locs\n";
        return false;
    }
    if (!ng::get_cell_warped_images_vector(dummy, cv::Mat(1, 1, CV_32FC2)).empty())
    {
        std::cout << "FAIL: get_cell_warped_images_vector did not return empty for 1x1 cross_locs\n";
        return false;
    }
    if (!ng::get_cell_warped_images_vector(cv::Mat(), cv::Mat(2, 2, CV_32FC2)).empty())
    {
        std::cout << "FAIL: get_cell_warped_images_vector did not return empty for empty image\n";
        return false;
    }
    return true;
}

bool test_get_cell_warped_images_vector_missing_cross_sentinel()
{
    cv::Mat const img = cv::Mat(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat cross_locs(2, 2, CV_32FC2);
    cross_locs.at<cv::Point2f>(0, 0) = cv::Point2f(10.0f, 10.0f);
    cross_locs.at<cv::Point2f>(0, 1) = cv::Point2f(30.0f, 10.0f);
    cross_locs.at<cv::Point2f>(1, 0) = cv::Point2f(-1.0f, -1.0f); // missing corner
    cross_locs.at<cv::Point2f>(1, 1) = cv::Point2f(30.0f, 30.0f);

    auto const cells = ng::get_cell_warped_images_vector(img, cross_locs);
    if (cells.size() != 1 || cells[0].size() != 1)
    {
        std::cout << "FAIL: cells grid size mismatch\n";
        return false;
    }
    if (cells[0][0].rows != 20 || cells[0][0].cols != 20)
    {
        std::cout << "FAIL: cell dimension mismatch\n";
        return false;
    }
    cv::Mat gray;
    cv::cvtColor(cells[0][0], gray, cv::COLOR_BGR2GRAY);
    if (cv::countNonZero(gray) != 0)
    {
        std::cout << "FAIL: cell with invalid corner is not zero-filled\n";
        return false;
    }
    return true;
}

int main()
{
    int failures = 0;

    {
        std::cout << "case: 8x10 grid, cell 40, resize 400\n";
        if (!run_case(8, 10, 40, 400, 8, 10)) ++failures;
    }
    {
        std::cout << "case: 6x8 grid, cell 50, resize 400\n";
        if (!run_case(6, 8, 50, 400, 6, 8)) ++failures;
    }
    {
        std::cout << "case: 10x12 grid, cell 60, resize 600\n";
        if (!run_case(10, 12, 60, 600, 10, 12)) ++failures;
    }
    {
        std::cout << "case: 20x24 grid, cell 40, resize 800 (reduced search ROI regression)\n";
        if (!run_case(20, 24, 40, 800, 20, 24)) ++failures;
    }

    {
        std::cout << "case: estimate_cell_side_length on 1-D periodic row signal\n";
        int const side = 37;              // odd, mirrors masks' odd-length requirement
        int const N = 300;               // signal length
        cv::Mat sig(1, N, CV_8U, cv::Scalar(0));
        for (int x = 0; x < N; ++x)
            if (x % side == 0) sig.at<uchar>(0, x) = 1;   // thin grid line

        int est = ng::CrossLocsDetector::estimate_cell_side_length(
            sig, cv::Rect(0, 0, N, 1), 5, 50);
        if (est != side) {
            std::cerr << "  [FAIL] estimate_cell_side_length=" << est << " expected " << side << "\n";
            ++failures;
        } else {
            std::cout << "  [ok] estimated cell side " << est << "\n";
        }
    }

    {
        std::cout << "case: subpixel peak refinement\n";
        test_refine_peak_loc_x();
        test_refine_peak_loc_fallback_on_boundary();
    }

    {
        std::cout << "case: end-to-end subpixel cross detection\n";
        if (!test_find_kernel_loc_subpixel_cross()) ++failures;
        if (!test_refine_peak_loc_flat_response()) ++failures;
        if (!test_refine_cross_locs_ink_subpixel()) ++failures;
        if (!test_refine_cross_locs_ink_empty_window()) ++failures;
        if (!test_refine_cross_locs_ink_rejects_non_cross_ink()) ++failures;
        if (!test_refine_cross_locs_ink_bold_line_converges()) ++failures;
        if (!test_grid_smooth_fit_approach1_recovers_smooth_grid()) ++failures;
        if (!test_grid_smooth_fit_approach2_recovers_smooth_grid()) ++failures;
        if (!test_grid_smooth_fit_approach3_recovers_smooth_grid()) ++failures;
        if (!test_prepare_input_accepts_thin_digit()) ++failures;
        if (!test_prepare_input_rejects_speck()) ++failures;
        if (!test_prepare_input_rejects_empty()) ++failures;
        if (!test_guard_implausible_split_falls_back()) ++failures;
        if (!test_guard_whole_cell_wins_on_counter_fp()) ++failures;
        if (!test_guard_genuine_two_digit_kept()) ++failures;
        if (!test_guard_zero_means_empty()) ++failures;
        if (!test_resize_empty_image()) ++failures;
        if (!test_get_cell_warped_images_vector_empty_or_small()) ++failures;
        if (!test_get_cell_warped_images_vector_missing_cross_sentinel()) ++failures;
    }

    failures += run_digit_recognizer_tests();

#ifdef NG_ENABLE_SOLVER
    failures += run_solver_tests();
    failures += run_clue_corrector_tests();
    failures += run_non_file_tests();
#endif

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }

    std::cout << "all tests passed\n";
    return 0;
}
