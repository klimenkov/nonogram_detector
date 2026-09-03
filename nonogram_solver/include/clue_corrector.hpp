#pragma once

#include <cstdint>
#include <vector>

namespace ng
{

// One decoded digit in a clue cell.
struct ClueCell
{
    int digit = -1;          // -1 = empty cell; 0..9 = recognized digit
    double confidence = 0.0; // softmax confidence of the digit (0 when empty)
    int digit_count = 1;     // 1 or 2: how many digits physically occupy the cell
                             // (from the counter model; 1 for a normal single-
                             // digit cell, 2 for a genuine two-digit clue cell).
};

// Raw decoded rectangular clue-cell grids.
//
// rows[r][c] is the LEFT-strip cell at column c of grid row r (row-major).
// cols[c][r] is the TOP-strip cell of grid column c (the top strip is
// transposed so each grid column maps to one clue line).
struct DecodedCells
{
    std::vector<std::vector<ClueCell>> rows;
    std::vector<std::vector<ClueCell>> cols;
};

// Groups the non-empty digit cells of one clue line into clue numbers. Each
// non-empty cell holds one full clue value: a single digit (1..9) or a
// two-digit cell (10..99, from the counter-driven split reader). So every
// non-empty cell of digit >= 1 is its own clue number, and adjacent non-empty
// cells are adjacent *distinct* clues (never concatenated into digits of one
// number). A lone '0' is a bogus stray read and is dropped. Returns one
// integer per clue number.
std::vector<int> group_clue_line(std::vector<ClueCell> const& cells);

// Returns the min line size (segments + separators) for a clue number sequence:
// sum(segments) + (segments-1), or 0 for an empty line.
std::int64_t clue_line_min_size(std::vector<int> const& segments);

struct ConsistencyReport
{
    bool consistent = false;        // all checks below pass
    std::int64_t row_tiles = 0;     // total filled tiles demanded by row clues
    std::int64_t col_tiles = 0;     // total filled tiles demanded by col clues
    std::vector<int> row_overflow;  // row indices whose clues don't fit width
    std::vector<int> col_overflow;  // col indices whose clues don't fit height
};

// Validates global nonogram consistency of <cells> for a <width> x <height>
// grid: the row and column clue tile totals must be equal, and every line's
// clues must fit the grid dimension on its axis.
ConsistencyReport analyze_consistency(DecodedCells const& cells, int width, int height);

// Best-effort correction of decoded cells toward global consistency, applied
// between decode and solve:
//   - drops bogus standalone 0-cells (border/stray reads), treating them as
//     empty; zeros inside a multi-digit clue (e.g. "10") are preserved;
//   - grid-fit repair: when a run of adjacent non-empty cells concatenates
//     into a clue number larger than the grid axis (an impossible clue, caused
//     by the decoder not emitting gap cells between distinct clues), and every
//     digit of that run is 1-9, splits the run into single-digit clues by
//     inserting gap cells -- but only keeps the edit if it makes the line fit.
// <report> is filled with the consistency *after* correction. The corrected
// cells may still be inconsistent when structural constraints are insufficient
// to disambiguate (the exact case for high-confidence single-digit misreads),
// which is why the caller should surface <report> as diagnostics rather than
// assume a true puzzle is recovered.
DecodedCells correct_clues(DecodedCells const& cells, int width, int height,
                           ConsistencyReport& report);

}
