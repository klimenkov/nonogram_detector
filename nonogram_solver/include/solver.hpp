#pragma once

#include <string>
#include <vector>

namespace ng
{

// Clue constraints for a nonogram, as decoded from the left (rows) and top
// (cols) clue strips. Each line is a sequence of positive segment lengths; an
// empty vector means "no clue for this line". Values must be >= 1; 0/negative
// entries (empty clue cells) are treated as absent.
struct ClueConstraints
{
    std::vector<std::vector<int>> rows;   // one clue per grid row (left strip)
    std::vector<std::vector<int>> cols;   // one clue per grid column (top strip)
};

// A solved grid: rows x cols of 0 (empty) / 1 (filled).
using SolutionGrid = std::vector<std::vector<int>>;

struct SolveResult
{
    bool solved = false;         // true when a full solution was produced
    bool line_solvable = false;  // whether it was solved without guessing
    std::size_t solution_count = 0;
    SolutionGrid solution;       // first solution (or partial grid if not solved)
    std::string message;         // human-readable status / error
};

// Solves a nonogram via the third-party picross-solver library and returns the
// result. This adapter is the only translation unit that depends on picross.
SolveResult solve_nonogram(ClueConstraints const& clues);

}
