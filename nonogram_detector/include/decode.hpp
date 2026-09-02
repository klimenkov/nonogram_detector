#pragma once

#include <vector>

#include <opencv2/opencv.hpp>

#include "detection.hpp"
#include "digit_recognizer.hpp"

namespace ng
{

// The decoded clue strips of a detected nonogram.
//
// `top` and `left` are row-major grids of the same shape as the corresponding
// detection region: element [i][j] is the digit in the clue cell at row i,
// column j, or -1 when that cell is empty. Each row of the grid is one clue
// line; consecutive non-empty entries within a row form a single clue number's
// digits (e.g. {2, 3} -> 23).
struct ClueGrid
{
    std::vector<std::vector<int>> top;
    std::vector<std::vector<int>> left;
};

// Decodes every clue cell in the top and left strips of <detection> on <image>
// using <recognizer>. Returns false if the recognizer could not run or the
// detection has no clue regions.
bool decode_clues(
    cv::Mat const& image,
    Detection const& detection,
    DigitRecognizer const& recognizer,
    ClueGrid& out);

}
