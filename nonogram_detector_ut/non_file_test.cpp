#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "non_file.hpp"
#include "solver.hpp"


namespace
{

std::vector<std::string> failures;

void check(bool cond, std::string const& name)
{
    if (!cond)
    {
        std::cerr << "  [FAIL] " << name << "\n";
        failures.push_back(name);
    }
}

// Round-trips a synthetic puzzle through the REAL ng::write_non_file /
// ng::parse_non_custom_block pair and verifies the custom block, the standard
// fields, and the fractional subpixel coordinates.
void test_round_trip()
{
    std::cout << "case: write_non_file custom block round-trip\n";

    // Build a small fake Detection.main: 3x4 intersections (2x3 cell grid)
    // with fractional (subpixel) coordinates.
    cv::Mat main_locs(3, 4, CV_32FC2);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            main_locs.at<cv::Point2f>(r, c) =
                cv::Point2f(100.5f + c * 50.25f, 200.75f + r * 60.5f);

    ng::SolutionGrid solution = {
        {1, 0, 1},
        {0, 1, 0}
    };

    ng::ClueConstraints constraints;
    constraints.rows = {{2}, {1}};
    constraints.cols = {{1}, {1}, {1}};

    auto const tmp = std::filesystem::temp_directory_path() / "non_test_output.non";
    auto const photo = std::filesystem::temp_directory_path() / "some_photo.jpg";

    ng::write_non_file(tmp, constraints, solution, main_locs, photo);

    auto const block = ng::parse_non_custom_block(tmp);

    check(block.source == photo.filename().string(),
          "source path round-trips (relative to .non dir)");
    check(block.grid_rows == 3, "grid rows round-trip");
    check(block.grid_cols == 4, "grid cols round-trips");
    check(static_cast<int>(block.cells.size()) == 3 * 4,
          "cell count matches grid dims (3x4)");

    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            cv::Point2f const expected = main_locs.at<cv::Point2f>(r, c);
            cv::Point2f const actual = block.cells[r * 4 + c];
            // Coordinates are serialized with 2 decimals, so allow a half-ulp
            // of the last digit.
            bool const close =
                std::fabs(actual.x - expected.x) <= 0.005f &&
                std::fabs(actual.y - expected.y) <= 0.005f;
            if (!close)
            {
                std::string name = "cell[" + std::to_string(r) + "][" +
                                   std::to_string(c) + "] round-trips";
                check(false, name);
            }
        }
    }

    {
        std::ifstream in(tmp);
        std::string line;
        bool found_title = false, found_width = false, found_height = false;
        bool found_rows = false, found_columns = false, found_goal = false;
        while (std::getline(in, line))
        {
            if (line.find("title ") == 0) found_title = true;
            if (line.find("width ") == 0) found_width = true;
            if (line.find("height ") == 0) found_height = true;
            if (line.find("rows") == 0) found_rows = true;
            if (line.find("columns") == 0) found_columns = true;
            if (line.find("goal ") == 0) found_goal = true;
        }
        check(found_title, "standard title field present");
        check(found_width, "standard width field present");
        check(found_height, "standard height field present");
        check(found_rows, "standard rows field present");
        check(found_columns, "standard columns field present");
        check(found_goal, "goal field present for a solved puzzle");
    }

    std::filesystem::remove(tmp);
}

// Test: empty detection.main produces no custom block.
void test_empty_detection_no_block()
{
    std::cout << "case: empty detection.main omits custom block\n";

    ng::SolutionGrid solution = {{1}, {0}};
    ng::ClueConstraints constraints;
    constraints.rows = {{1}, {1}};
    constraints.cols = {{1}};

    auto const tmp = std::filesystem::temp_directory_path() / "non_test_empty.non";
    auto const photo = std::filesystem::temp_directory_path() / "photo.jpg";

    ng::write_non_file(tmp, constraints, solution, cv::Mat(), photo);

    auto const block = ng::parse_non_custom_block(tmp);
    check(block.source.empty(), "no source when detection is empty");
    check(block.grid_rows == 0, "no grid rows when detection is empty");
    check(block.grid_cols == 0, "no grid cols when detection is empty");
    check(block.cells.empty(), "no cells when detection is empty");

    std::filesystem::remove(tmp);
}

// Test: relative path computation.
void test_relative_source_path()
{
    std::cout << "case: #source is relative to .non file parent\n";

    auto const dir = std::filesystem::temp_directory_path() / "non_rel_test";
    std::filesystem::create_directories(dir);
    auto const non_path = dir / "puzzle.non";
    auto const photo_path = dir / "photo.jpg";

    ng::SolutionGrid solution = {{1}};
    ng::ClueConstraints constraints;
    constraints.rows = {{1}};
    constraints.cols = {{1}};

    cv::Mat main_locs(2, 2, CV_32FC2);
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 2; ++c)
            main_locs.at<cv::Point2f>(r, c) = cv::Point2f(10 + c * 5, 20 + r * 5);

    ng::write_non_file(non_path, constraints, solution, main_locs, photo_path);

    auto const block = ng::parse_non_custom_block(non_path);
    check(block.source == "photo.jpg", "relative source is just the filename");

    std::filesystem::remove_all(dir);
}

void test_parse_malformed_non_file()
{
    std::cout << "case: parse_non_custom_block handles malformed tokens\n";

    auto const tmp = std::filesystem::temp_directory_path() / "malformed_test.non";
    {
        std::ofstream out(tmp);
        out << "title \"malformed\"\n";
        out << "#source: photo.jpg\n";
        out << "#grid: invalid_dims_without_separator\n";
        out << "#cells:\n";
        out << "missing_comma 12.34,56.78 incomplete,trailing_comma,\n";
    }

    try
    {
        auto const block = ng::parse_non_custom_block(tmp);
        check(block.source == "photo.jpg", "malformed file source parsed");
        check(block.grid_rows == 0, "malformed grid rows defaults to 0");
        check(block.grid_cols == 0, "malformed grid cols defaults to 0");
        check(block.cells.size() >= 1, "valid token parsed despite malformed neighbors");
        if (!block.cells.empty())
        {
            check(std::fabs(block.cells[0].x - 12.34f) <= 0.01f, "cell x parsed");
            check(std::fabs(block.cells[0].y - 56.78f) <= 0.01f, "cell y parsed");
        }
    }
    catch (std::exception const& e)
    {
        check(false, std::string("parse_non_custom_block threw exception: ") + e.what());
    }

    std::error_code ec;
    std::filesystem::remove(tmp, ec);
}

}

int run_non_file_tests()
{
    failures.clear();
    test_round_trip();
    test_empty_detection_no_block();
    test_relative_source_path();
    test_parse_malformed_non_file();

    for (auto const& f : failures)
        std::cerr << "non_file: " << f << "\n";
    if (!failures.empty())
    {
        std::cerr << failures.size() << " non_file test(s) FAILED\n";
        return static_cast<int>(failures.size());
    }
    std::cout << "all non_file tests passed\n";
    return 0;
}
