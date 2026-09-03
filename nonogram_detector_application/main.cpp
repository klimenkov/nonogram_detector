#include <cstdlib>
#include <filesystem>
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

    ng::ClueGrid clues;
    if (ng::decode_clues(image, detection, recognizer, clues))
    {
        std::cout << "top clues:\n";
        print_clue_grid(clues.top);
        std::cout << "left clues:\n";
        print_clue_grid(clues.left);

#ifdef NG_ENABLE_SOLVER
        // Convert the decoded clue strips into solver constraints: the left
        // strip maps directly to row constraints; the top strip is transposed
        // so it becomes one clue per grid column.
        ng::ClueConstraints constraints;
        constraints.rows = clues.left;
        constraints.cols.assign(clues.top.size() ? clues.top[0].size() : 0, {});
        for (std::size_t row = 0; row < clues.top.size(); ++row)
            for (std::size_t col = 0; col < clues.top[row].size(); ++col)
                constraints.cols[col].push_back(clues.top[row][col]);

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
