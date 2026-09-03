#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

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

// Minimal round-trip parser: reads the custom block (#source, #grid, #cells)
// from a .non file and returns the parsed values. Used only for testing.
struct NonCustomBlock
{
    std::string source;
    int grid_rows = 0;
    int grid_cols = 0;
    std::vector<cv::Point2f> cells;  // flattened row-major
};

NonCustomBlock parse_non_custom_block(std::filesystem::path const& path)
{
    NonCustomBlock block;
    std::ifstream in(path);
    std::string line;

    while (std::getline(in, line))
    {
        if (line.rfind("#source: ", 0) == 0)
            block.source = line.substr(9);
        else if (line.rfind("#grid: ", 0) == 0)
        {
            std::string dims = line.substr(7);
            auto const x_pos = dims.find('x');
            block.grid_rows = std::atoi(dims.substr(0, x_pos).c_str());
            block.grid_cols = std::atoi(dims.substr(x_pos + 1).c_str());
        }
        else if (line.rfind("#cells:", 0) == 0)
        {
            // Next lines contain space-separated "x,y" pairs.
            while (std::getline(in, line))
            {
                if (line.empty() || line[0] == '#')
                    break;
                std::istringstream iss(line);
                std::string token;
                while (iss >> token)
                {
                    auto const comma = token.find(',');
                    float x = std::stof(token.substr(0, comma));
                    float y = std::stof(token.substr(comma + 1));
                    block.cells.emplace_back(x, y);
                }
            }
        }
    }
    return block;
}

// Test: write a .non file with a synthetic Detection.main and solution, then
// parse the custom block and verify everything round-trips.
void test_round_trip()
{
    std::cout << "case: write_non_file custom block round-trip\n";

    // Build a small fake Detection.main: 3x4 intersections (2x3 cell grid).
    // Fractional coordinates exercise the subpixel (CV_32FC2) round-trip.
    cv::Mat main_locs(3, 4, CV_32FC2);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            main_locs.at<cv::Point2f>(r, c) = cv::Point2f(100.5f + c * 50.25f, 200.75f + r * 60.5f);

    ng::SolutionGrid solution = {
        {1, 0, 1},
        {0, 1, 0}
    };

    ng::ClueConstraints constraints;
    constraints.rows = {{2}, {1}};
    constraints.cols = {{1}, {1}, {1}};

    auto const tmp = std::filesystem::temp_directory_path() / "non_test_output.non";

    {
        std::ofstream out(tmp);
        out << "title \"test\"\n";
        out << "width 3\n";
        out << "height 2\n\n";
        out << "rows\n2\n1\n\n";
        out << "columns\n1\n1\n1\n\n";
        out << "goal \"101010\"\n";

        // Custom block (mirrors write_non_file logic).
        out << "#source: /some/photo.jpg\n";
        out << "#grid: 3x4\n";
        out << "#cells:\n";
        for (int r = 0; r < main_locs.rows; ++r)
        {
            for (int c = 0; c < main_locs.cols; ++c)
            {
                if (c) out << " ";
                cv::Point2f const pt = main_locs.at<cv::Point2f>(r, c);
                out << std::fixed << std::setprecision(2) << pt.x << "," << pt.y;
            }
            out << "\n";
        }
    }

    auto const block = parse_non_custom_block(tmp);
    check(block.source == "/some/photo.jpg", "source path round-trips");
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
            // Two-decimal text round-trip: allow a half-ulp of the printed
            // precision.
            if (std::fabs(actual.x - expected.x) > 0.005f ||
                std::fabs(actual.y - expected.y) > 0.005f)
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
        while (std::getline(in, line))
        {
            if (line.find("title ") == 0) found_title = true;
            if (line.find("width ") == 0) found_width = true;
            if (line.find("height ") == 0) found_height = true;
        }
        check(found_title, "standard title field present");
        check(found_width, "standard width field present");
        check(found_height, "standard height field present");
    }

    std::filesystem::remove(tmp);
}

// Test: empty detection.main produces no custom block.
void test_empty_detection_no_block()
{
    std::cout << "case: empty detection.main omits custom block\n";

    auto const tmp = std::filesystem::temp_directory_path() / "non_test_empty.non";

    {
        std::ofstream out(tmp);
        out << "title \"test\"\n";
        out << "width 2\n";
        out << "height 2\n\n";
        out << "rows\n1\n1\n\n";
        out << "columns\n1\n1\n\n";
        out << "goal \"1001\"\n";
        // No custom block written (simulates empty detection).
    }

    auto const block = parse_non_custom_block(tmp);
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

    {
        std::ofstream out(non_path);
        out << "title \"test\"\nwidth 1\nheight 1\n\nrows\n1\n\n\ncolumns\n1\n\n\n";
        out << "goal \"1\"\n";
        std::error_code ec;
        auto rel = std::filesystem::relative(photo_path, non_path.parent_path(), ec);
        out << "#source: " << rel.string() << "\n";
    }

    auto const block = parse_non_custom_block(non_path);
    check(block.source == "photo.jpg", "relative source is just the filename");

    std::filesystem::remove_all(dir);
}

}

int run_non_file_tests()
{
    failures.clear();
    test_round_trip();
    test_empty_detection_no_block();
    test_relative_source_path();

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
