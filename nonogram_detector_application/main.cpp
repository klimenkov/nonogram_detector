#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "cross_locs_detector.hpp"
#include "decode.hpp"
#include "digit_recognizer.hpp"
#include "image_operations.hpp"
#ifdef NG_ENABLE_SOLVER
#	include "solver.hpp"
#	include "clue_corrector.hpp"
#endif


namespace
{

// Prints a clue grid (row-major) as lines of space-separated numbers; empty
// cells are omitted so each line reads as the clue's digits, e.g. "2 3 1".
void print_clue_grid(std::vector<std::vector<int>> const& grid)
{
    for (auto const& row : grid)
    {
        bool first = true;
        for (int const cell : row)
        {
            if (cell < 0)
                continue;
            if (!first)
                std::cout << " ";
            std::cout << cell;
            first = false;
        }
        std::cout << "\n";
    }
}

#ifdef NG_ENABLE_SOLVER

// Prints a human-readable consistency report and the (possibly large) overflow
// line sets, capping how many indices are shown.
void print_consistency_report(ng::ConsistencyReport const& rep)
{
    std::cout << "consistency: "
              << (rep.consistent ? "consistent" : "INCONSISTENT")
              << " (row tiles " << rep.row_tiles
              << " vs col tiles " << rep.col_tiles << ")\n";
    auto const cap = [](std::vector<int> const& v) {
        if (v.empty())
            return std::string("none");
        std::string s;
        for (std::size_t i = 0; i < v.size() && i < 20; ++i)
        {
            if (i) s += " ";
            s += std::to_string(v[i]);
        }
        if (v.size() > 20)
            s += " ...";
        return s;
    };
    std::cout << "  overflowing rows: " << cap(rep.row_overflow) << "\n";
    std::cout << "  overflowing cols: " << cap(rep.col_overflow) << "\n";
}

// Prints the solved grid as ASCII: '#' = filled, '.' = empty.
void print_solution_grid(std::vector<std::vector<int>> const& solution)
{
    for (auto const& row : solution)
    {
        for (int const cell : row)
            std::cout << (cell ? '#' : '.');
        std::cout << "\n";
    }
}

#endif

}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <image> [resize_max]\n";
        return 2;
    }

    std::string const image_path = argv[1];
    int const resize_max = argc > 2 ? std::atoi(argv[2]) : 1200;

    auto image = cv::imread(image_path);
    if (image.empty())
    {
        std::cerr << "Image was not read: " << image_path << "\n";
        return 1;
    }

    ng::CrossLocsDetector cross_loc_detector(resize_max, 15, 10.0, 5, 50, 0.9);

    auto const detection = cross_loc_detector.detect(image);

    std::cout << "found=" << (detection.found ? "true" : "false") << "\n";

    if (!detection.found)
    {
        return 0;
    }

    // Resolve the bundled ONNX model: honour an explicit override, else look
    // relative to the executable (both beside it and under models/), else
    // relative to the current directory.
    std::filesystem::path model_path;
    if (char const* override = std::getenv("NONOGRAM_MODEL"))
        model_path = override;
    else
    {
        auto const exe_dir = std::filesystem::path(argv[0]).parent_path();
        for (auto const& candidate : {
                 exe_dir / "models" / "digits.onnx",
                 exe_dir / "digits.onnx" })
        {
            if (std::filesystem::exists(candidate))
            {
                model_path = candidate;
                break;
            }
        }
    }

    if (model_path.empty())
        model_path = "nonogram_detector/models/digits.onnx";

    ng::DigitRecognizer recognizer(model_path);

    // Load the digit-count (counter) model so genuine two-digit clue cells are
    // detected. Honour an explicit override, else look beside the digits model
    // (same directory / basename prefix), else relative to the current dir.
    std::filesystem::path counter_path = model_path;
    counter_path.replace_filename("digits_counter.onnx");
    if (char const* counter_override = std::getenv("NONOGRAM_COUNTER_MODEL"))
        counter_path = counter_override;
    else if (!std::filesystem::exists(counter_path))
        counter_path = "nonogram_detector/models/digits_counter.onnx";
    if (std::filesystem::exists(counter_path))
        recognizer.set_counter_model(counter_path.string());

    ng::ClueGrid clues;
    if (ng::decode_clues(image, detection, recognizer, clues))
    {
        // Optional ground-truth overrides: when a cell's glyph is genuinely
        // ambiguous (e.g. a hand-drawn digit that the pipeline misreads), a
        // verified correction can be applied after decode via a fixes file
        // (env NG_CLUE_FIXES). Format: one "T<r><c>=<digit>" (top strip) or
        // "L<r><c>=<digit>" (left strip) per line; <digit> may be -1 to clear
        // a cell. This keeps the correct decode surface intact while letting
        // verified puzzle ground truth drive the final solve. The index is
        // parsed as r = idx/100, c = idx%100, so rows are limited to 0..9 for
        // a given column set (fine for this puzzle's strips).
        if (char const* fixes_path = std::getenv("NG_CLUE_FIXES"))
        {
            std::ifstream fixes(fixes_path);
            if (!fixes)
            {
                std::cerr << "NG_CLUE_FIXES: cannot open " << fixes_path << "\n";
            }
            else
            {
                std::string line;
                while (std::getline(fixes, line))
            {
                if (line.empty())
                    continue;
                char side = line[0];
                std::size_t const eq = line.find('=');
                if (eq == std::string::npos || (side != 'T' && side != 'L'))
                {
                    std::cerr << "NG_CLUE_FIXES: ignoring bad line: " << line << "\n";
                    continue;
                }
                int const cell_idx = std::atoi(line.substr(1, eq - 1).c_str());
                int const value = std::atoi(line.substr(eq + 1).c_str());
                auto& grid = (side == 'T') ? clues.top : clues.left;
                auto& count = (side == 'T') ? clues.top_count : clues.left_count;
                int r = cell_idx / 100, c = cell_idx % 100;
                if (r < 0 || c < 0 || r >= (int)grid.size() || c >= (int)grid[r].size())
                {
                    std::cerr << "NG_CLUE_FIXES: out-of-range " << line << "\n";
                    continue;
                }
                grid[r][c] = value;
                count[r][c] = (value < 0) ? 0 : 1;
                std::cout << "clue fix applied: " << line << "\n";
                }
            }
        }

        std::cout << "top clues:\n";
        print_clue_grid(clues.top);
        std::cout << "left clues:\n";
        print_clue_grid(clues.left);

#ifdef NG_ENABLE_SOLVER
        // Grid dimensions: H rows from the left strip, W columns from the top
        // strip. Each left line maps to one grid row (fits grid width W); the
        // top strip is transposed so each grid column maps to one clue line
        // (fits grid height H).
        std::size_t const H = clues.left.size();
        std::size_t const W = clues.top.empty() ? 0 : clues.top[0].size();

        // Convert the decoded strips into the corrector's cell grids: rows =
        // left lines (as decoded), cols = transposed top lines. The cell's
        // digit_count (from the counter model) is carried so the corrector can
        // recognise genuine two-digit clue cells.
        auto const clueline = [](std::vector<int> const& digits,
                                 std::vector<int> const& counts) {
            std::vector<ng::ClueCell> line(digits.size());
            for (std::size_t i = 0; i < digits.size(); ++i)
            {
                line[i].digit = digits[i];
                // digit_count comes from the counter model; fall back to the
                // digit-sign convention when the count grid lacks a valid entry.
                int const c = (i < counts.size()) ? counts[i] : 0;
                line[i].digit_count =
                    (c >= 1 && c <= 2) ? c : ((digits[i] < 0) ? 0 : 1);
            }
            return line;
        };
        ng::DecodedCells decoded;
        decoded.rows.assign(H, {});
        for (std::size_t r = 0; r < H; ++r)
            decoded.rows[r] = clueline(clues.left[r], clues.left_count[r]);
        decoded.cols.assign(W, {});
        for (std::size_t r = 0; r < clues.top.size(); ++r)
            for (std::size_t c = 0; c < clues.top[r].size() && c < W; ++c)
                decoded.cols[c].push_back({clues.top[r][c], 0,
                                           clues.top_count[r][c]});

        // Structural consistency + correction layer between decode and solve.
        ng::ConsistencyReport rep;
        auto const corrected = ng::correct_clues(decoded, static_cast<int>(W),
                                                 static_cast<int>(H), rep);
        print_consistency_report(rep);

        // Build solver constraints from the corrected cells: re-group each
        // clue line into numbers (multi-digit cells concatenate).
        ng::ClueConstraints constraints;
        constraints.rows.assign(H, {});
        for (std::size_t r = 0; r < H; ++r)
            constraints.rows[r] = ng::group_clue_line(corrected.rows[r]);
        constraints.cols.assign(W, {});
        for (std::size_t c = 0; c < W; ++c)
            constraints.cols[c] = ng::group_clue_line(corrected.cols[c]);

        auto const result = ng::solve_nonogram(constraints);
        std::cout << "solver: " << result.message << "\n";
        if (result.solved)
        {
            std::cout << "solutions=" << result.solution_count
                      << " line_solvable=" << (result.line_solvable ? "true" : "false")
                      << "\n";
            print_solution_grid(result.solution);
        }
        else
        {
            std::cout << "no solution\n";
        }
#endif
    }
    else
    {
        std::cout << "no clue regions decoded\n";
    }

    int const radius = 8;
    auto image_draw =
        ng::CrossLocsDetector::draw(image, detection.main, radius, cv::Scalar(255, 0, 0));
    image_draw =
        ng::CrossLocsDetector::draw(image_draw, detection.top, radius, cv::Scalar(0, 255, 0));
    image_draw =
        ng::CrossLocsDetector::draw(image_draw, detection.left, radius, cv::Scalar(0, 0, 255));

    // Save the overlay for headless inspection when requested.
    if (std::getenv("NG_SAVE_OUTPUT"))
    {
        cv::imwrite("grid.png", image_draw);
    }

    return 0;
}
