#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "solver.hpp"

#include <opencv2/opencv.hpp>

namespace ng
{

// Parsed custom block of a .non file written by write_non_file: the source
// photo reference, the intersection-matrix dimensions, and the flattened
// row-major intersection points. Missing sections keep their default values.
struct NonCustomBlock
{
    std::string source;
    int grid_rows = 0;
    int grid_cols = 0;
    std::vector<cv::Point2f> cells;  // flattened row-major
};

// Writes the puzzle in webpbn ".non" text format: the row/column clue lists
// (comma-separated) plus a "goal" line, the rows-major 0/1 bitmap of the solved
// grid. When <detection_main> is non-empty, also appends a custom block
// recording the source photo reference and the grid-intersection coordinates
// (see NonCustomBlock).
void write_non_file(
    std::filesystem::path const& path,
    ClueConstraints const& constraints,
    SolutionGrid const& solution,
    cv::Mat const& detection_main,
    std::filesystem::path const& image_path);

// Reads back the custom block (#source / #grid / #cells) of a .non file
// written by write_non_file.
NonCustomBlock parse_non_custom_block(std::filesystem::path const& path);

}
