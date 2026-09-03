#include "clue_corrector.hpp"

#include <algorithm>

namespace ng
{

namespace
{

// The decoder emits one clue per cell: a single-digit cell (1..9) or a
// two-digit cell (10..99, produced by the counter-driven split reader) is a
// complete clue number of its own. Adjacent non-empty cells are adjacent
// *distinct* clues; there is no run of cells that concatenates into one number.
bool is_complete_clue(ClueCell const& c)
{
    return c.digit >= 1; // a non-empty cell always holds one full clue value
}

}

std::vector<int> group_clue_line(std::vector<ClueCell> const& cells)
{
    std::vector<int> results;
    for (ClueCell const& c : cells)
    {
        // Every non-empty cell is its own clue number. A lone '0' is a bogus
        // border/stray read (a line with no fill has no clue) and is dropped.
        if (c.digit >= 1)
            results.push_back(c.digit);
    }
    return results;
}

std::int64_t clue_line_min_size(std::vector<int> const& segments)
{
    if (segments.empty())
        return 0;
    std::int64_t sum = 0;
    for (int const s : segments)
        sum += s;
    return sum + static_cast<std::int64_t>(segments.size() - 1);
}


namespace
{

std::int64_t line_tiles(std::vector<ClueCell> const& cells)
{
    std::int64_t total = 0;
    for (int const s : ng::group_clue_line(cells))
        total += s;
    return total;
}

// Returns the min line size of a cell line (0 for an empty line).
std::int64_t cell_line_min_size(std::vector<ClueCell> const& cells)
{
    return ng::clue_line_min_size(ng::group_clue_line(cells));
}

// Marks every cell of a zero-valued run (e.g. a lone '0') as empty. Zeros that
// are part of a multi-digit number ("10") are preserved because that run's
// value is non-zero.
void sanitize_line(std::vector<ClueCell>& cells)
{
    std::size_t i = 0;
    while (i < cells.size())
    {
        if (cells[i].digit < 0)
        {
            ++i;
            continue;
        }
        // Find the contiguous non-empty run starting at i.
        std::size_t run_end = i;
        std::int64_t value = 0;
        bool complete = false; // run contains a full multi-digit clue cell
        while (run_end < cells.size() && cells[run_end].digit >= 0)
        {
            if (is_complete_clue(cells[run_end]))
            {
                complete = true;
                value = 1; // a full multi-digit clue makes the run non-zero
            }
            else
            {
                value = value * 10 + cells[run_end].digit;
            }
            ++run_end;
        }
        if (!complete && value == 0)
        {
            for (std::size_t k = i; k < run_end; ++k)
                cells[k] = ClueCell{}; // digit=-1, confidence=0
        }
        i = run_end;
    }
}

// True when the whole run can be re-interpreted as separate single-digit clues.
bool is_single_digit_run(std::vector<ClueCell> const& line, std::size_t begin, std::size_t end)
{
    for (std::size_t k = begin; k < end; ++k)
    {
        if (line[k].digit < 1 || line[k].digit > 9)
            return false;
    }
    return true;
}

// Returns the concatenated numeric value of the contiguous run [begin, end).
// A cell holding a full multi-digit clue value (>=10) is not a decimal digit to
// concatenate; its presence always yields a value larger than a single clue.
std::int64_t run_value(std::vector<ClueCell> const& line, std::size_t begin, std::size_t end)
{
    std::int64_t value = 0;
    for (std::size_t k = begin; k < end; ++k)
    {
        if (is_complete_clue(line[k]))
            value = std::max<std::int64_t>(value * 10 + 9, line[k].digit);
        else
            value = value * 10 + line[k].digit;
    }
    return value;
}

// Grid-fit repair (legacy, inert under Model B): if <line> overflows <axis>,
// split every run whose concatenated value exceeds <axis> (an impossible single
// clue) into single-digit clues separated by gap cells. The rebuild is kept
// only if it makes the line fit; otherwise the line is left unchanged.
//
// Under Model B each non-empty cell is already its own complete clue, so
// splitting a run into gap-separated single-digit cells never changes the
// grouped clues or min line size: this function is only a safety net retained
// for robustness and is not exercised by Model B decoder output.
void split_overflowing_line(std::vector<ClueCell>& line, std::int64_t axis)
{
    if (cell_line_min_size(line) <= axis)
        return;

    std::vector<ClueCell> rebuilt;
    std::size_t i = 0;
    while (i < line.size())
    {
        if (line[i].digit < 0)
        {
            rebuilt.push_back(line[i]);
            ++i;
            continue;
        }
        std::size_t end = i;
        while (end < line.size() && line[end].digit >= 0)
            ++end;

        if (run_value(line, i, end) > axis && is_single_digit_run(line, i, end))
        {
            for (std::size_t k = i; k < end; ++k)
            {
                rebuilt.push_back(line[k]);
                if (k + 1 < end)
                    rebuilt.push_back(ClueCell{}); // inter-clue gap
            }
        }
        else
        {
            for (std::size_t k = i; k < end; ++k)
                rebuilt.push_back(line[k]);
        }
        i = end;
    }

    if (cell_line_min_size(rebuilt) <= axis)
        line = std::move(rebuilt);
}

} // namespace


ConsistencyReport analyze_consistency(DecodedCells const& cells, int width, int height)
{
    ConsistencyReport rep;
    std::int64_t row_tiles = 0;
    std::int64_t col_tiles = 0;

    for (std::size_t r = 0; r < cells.rows.size(); ++r)
    {
        row_tiles += line_tiles(cells.rows[r]);
        if (cell_line_min_size(cells.rows[r]) > width)
            rep.row_overflow.push_back(static_cast<int>(r));
    }
    for (std::size_t c = 0; c < cells.cols.size(); ++c)
    {
        col_tiles += line_tiles(cells.cols[c]);
        if (cell_line_min_size(cells.cols[c]) > height)
            rep.col_overflow.push_back(static_cast<int>(c));
    }

    rep.row_tiles = row_tiles;
    rep.col_tiles = col_tiles;
    rep.consistent = (row_tiles == col_tiles) && rep.row_overflow.empty() && rep.col_overflow.empty();
    return rep;
}

DecodedCells correct_clues(DecodedCells const& cells, int width, int height,
                           ConsistencyReport& report)
{
    DecodedCells out = cells;
    for (auto& line : out.rows)
        sanitize_line(line);
    for (auto& line : out.cols)
        sanitize_line(line);
    for (auto& line : out.rows)
        split_overflowing_line(line, width);
    for (auto& line : out.cols)
        split_overflowing_line(line, height);

    report = analyze_consistency(out, width, height);
    return out;
}

}
