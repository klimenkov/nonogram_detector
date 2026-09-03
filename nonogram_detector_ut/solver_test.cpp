#include <iostream>
#include <vector>

#include "solver.hpp"


// Runs the solver adapter against a known tiny puzzle (the 6x6 example from
// the picross-solver README). Confirms the solver produces the expected grid
// and that the public adapter types map correctly.
int run_solver_tests()
{
    int failures = 0;

    {
        std::cout << "case: 6x6 README puzzle\n";

        ng::ClueConstraints clues;
        clues.rows = {
            { 3 }, { 1, 1 }, { 1, 1 }, { 3 }, { 3 }, { },
        };
        clues.cols = {
            { }, { 2 }, { 2 }, { 5 }, { 1 }, { 3 },
        };

        auto const result = ng::solve_nonogram(clues);

        if (!result.solved)
        {
            std::cerr << "  [FAIL] not solved: " << result.message << "\n";
            ++failures;
        }
        else if (result.solution_count != 1u)
        {
            std::cerr << "  [FAIL] expected 1 solution, got " << result.solution_count << "\n";
            ++failures;
        }
        else
        {
            std::vector<std::vector<int>> const expected = {
                { 0, 0, 0, 1, 1, 1 },
                { 0, 0, 0, 1, 0, 1 },
                { 0, 0, 0, 1, 0, 1 },
                { 0, 1, 1, 1, 0, 0 },
                { 0, 1, 1, 1, 0, 0 },
                { 0, 0, 0, 0, 0, 0 },
            };
            if (result.solution != expected)
            {
                std::cerr << "  [FAIL] solution grid mismatch\n";
                for (auto const& row : result.solution)
                {
                    std::cerr << "    ";
                    for (int const cell : row)
                        std::cerr << (cell ? '#' : '.');
                    std::cerr << "\n";
                }
                ++failures;
            }
            else
            {
                std::cout << "  [ok] solved 6x6 with 1 solution\n";
            }
        }
    }

    {
        std::cout << "case: contradictory grid reports failure\n";

        // Two columns both demand a filled cell on a single row that can only
        // satisfy one of them -> no solution.
        ng::ClueConstraints clues;
        clues.rows = {
            { 1 },
        };
        clues.cols = {
            { 1 }, { 1 },
        };

        auto const result = ng::solve_nonogram(clues);
        if (result.solved)
        {
            std::cerr << "  [FAIL] expected contradictory grid to be unsolved\n";
            ++failures;
        }
        else
        {
            std::cout << "  [ok] contradictory grid reported unsolved (" << result.message << ")\n";
        }
    }

    {
        // A clue line whose segments exceed the grid dimension on that axis is
        // invalid; the adapter must report it gracefully instead of letting the
        // solver throw (regression: line_size < min_line_size).
        std::cout << "case: clue overflowing grid dimension reports failure\n";

        ng::ClueConstraints clues;
        clues.rows = {
            { 1, 1, 1, 1, 1 },   // 5 segments need >= 9 cells, but grid is 1x?
        };
        clues.cols = {
            { 1 },
        };

        auto const result = ng::solve_nonogram(clues);
        if (result.solved)
        {
            std::cerr << "  [FAIL] expected overflowing clue to be unsolved\n";
            ++failures;
        }
        else
        {
            std::cout << "  [ok] overflowing clue reported unsolved (" << result.message << ")\n";
        }
    }

    if (failures > 0)
        std::cerr << failures << " solver test(s) FAILED\n";
    else
        std::cout << "all solver tests passed\n";
    return failures;
}
