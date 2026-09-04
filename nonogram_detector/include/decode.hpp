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
// column j, or -1 when that cell is empty. Each non-empty cell holds one full
// clue number: a single digit (1..9) or a two-digit value (10..99, produced by
// the counter-driven split reader). Adjacent non-empty cells are adjacent
// *distinct* clues; they are never concatenated into digits of one number.
//
// `top_count` and `left_count` mirror the grids with each non-empty cell's
// digit count (1 or 2) as classified by the counter model; cells that are
// empty/unreliable hold 0.
struct ClueGrid
{
    std::vector<std::vector<int>> top;
    std::vector<std::vector<int>> left;
    std::vector<std::vector<int>> top_count;
    std::vector<std::vector<int>> left_count;
};

// Decodes every clue cell in the top and left strips of <detection> on <image>
// using <recognizer>. Returns false if the recognizer could not run or the
// detection has no clue regions.
bool decode_clues(
    cv::Mat const& image,
    Detection const& detection,
    DigitRecognizer const& recognizer,
    ClueGrid& out);

// Guard decision for a counter-flagged two-digit cell. <split> is the composed
// split read (-1 if a half failed), <whole>/<whole_conf> the whole-cell read.
// Returns the chosen digit (or -1). Exposed for unit testing.
int resolve_two_digit(int split, int whole, double whole_conf,
                      double conf_l, double conf_r,
                      int max_clue,
                      double split_conf_min, double whole_high_conf_min);

}
