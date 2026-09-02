#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "cross_locs_detector.hpp"


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

    if (failures > 0)
    {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }

    std::cout << "all tests passed\n";
    return 0;
}
