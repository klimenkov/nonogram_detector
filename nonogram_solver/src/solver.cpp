#include "solver.hpp"

#include <picross/picross.h>

namespace ng
{

namespace
{

// Strips empty/zero entries out of one decoded clue line so it becomes a
// picross constraint (segment sizes; empty line -> no clue).
picross::InputGrid::Constraint to_constraint(std::vector<int> const& cells)
{
    picross::InputGrid::Constraint constraint;
    for (int const v : cells)
        if (v > 0)
            constraint.push_back(static_cast<unsigned int>(v));
    return constraint;
}

std::string status_to_string(picross::Solver::Status status)
{
    switch (status)
    {
        case picross::Solver::Status::OK: return "ok";
        case picross::Solver::Status::ABORTED: return "aborted";
        case picross::Solver::Status::CONTRADICTORY_GRID: return "contradictory grid (no solution)";
        case picross::Solver::Status::NOT_LINE_SOLVABLE: return "not line solvable";
    }
    return "unknown";
}

}

SolveResult solve_nonogram(ClueConstraints const& clues)
{
    SolveResult result;

    picross::InputGrid::Constraints rows;
    rows.reserve(clues.rows.size());
    for (auto const& line : clues.rows)
        rows.push_back(to_constraint(line));

    picross::InputGrid::Constraints cols;
    cols.reserve(clues.cols.size());
    for (auto const& line : clues.cols)
        cols.push_back(to_constraint(line));

    picross::InputGrid puzzle(rows, cols, "nonogram_detector");

    // Reject malformed/inconsistent constraints up front instead of letting the
    // solver throw. picross::check_input_grid verifies that both dimensions are
    // non-zero, that every constraint's min_line_size fits the grid on that
    // axis, and that the total number of filled tiles matches on rows and cols.
    const auto check = picross::check_input_grid(puzzle);
    if (!check.first)
    {
        result.message = "invalid constraints: " + check.second;
        return result;
    }

    const auto solver = picross::get_ref_solver();
    if (!solver)
    {
        result.message = "failed to obtain the picross solver";
        return result;
    }

    // Look for all solutions; enough to detect uniqueness.
    const auto solve_result = solver->solve(puzzle);
    result.message = status_to_string(solve_result.status);

    switch (solve_result.status)
    {
        case picross::Solver::Status::OK:
            result.solved = true;
            result.solution_count = solve_result.solutions.size();
            if (!solve_result.solutions.empty())
            {
                auto const& grid = solve_result.solutions.front().grid;
                result.line_solvable = solve_result.solutions.front().branching_depth == 0;
                result.solution.assign(grid.height(), std::vector<int>(grid.width(), 0));
                for (std::size_t y = 0; y < grid.height(); ++y)
                    for (std::size_t x = 0; x < grid.width(); ++x)
                        result.solution[y][x] =
                            grid.get_tile(static_cast<unsigned int>(x), static_cast<unsigned int>(y))
                            == picross::Tile::FILLED ? 1 : 0;
            }
            break;

        case picross::Solver::Status::NOT_LINE_SOLVABLE:
            // A partial grid is carried in the solutions when branching was
            // needed; surface it as-is rather than dropping it.
            result.solved = false;
            result.solution_count = solve_result.solutions.size();
            if (!solve_result.solutions.empty())
            {
                auto const& grid = solve_result.solutions.front().grid;
                result.line_solvable = false;
                result.solution.assign(grid.height(), std::vector<int>(grid.width(), 0));
                for (std::size_t y = 0; y < grid.height(); ++y)
                    for (std::size_t x = 0; x < grid.width(); ++x)
                        result.solution[y][x] =
                            grid.get_tile(static_cast<unsigned int>(x), static_cast<unsigned int>(y))
                            == picross::Tile::FILLED ? 1 : 0;
            }
            break;

        case picross::Solver::Status::ABORTED:
        case picross::Solver::Status::CONTRADICTORY_GRID:
        default:
            result.solved = false;
            result.solution_count = 0;
            break;
    }

    return result;
}

}
