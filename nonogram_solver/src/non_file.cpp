#include "non_file.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace ng
{

namespace
{

// atoi-like float parse: malformed tokens read as 0 instead of throwing, so a
// malformed .non file still yields a parseable block.
float parse_float(std::string const& token)
{
    try
    {
        return std::stof(token);
    }
    catch (...)
    {
        return 0.0f;
    }
}

}

void write_non_file(
    std::filesystem::path const& path,
    ClueConstraints const& constraints,
    SolutionGrid const& solution,
    cv::Mat const& detection_main,
    std::filesystem::path const& image_path)
{
    std::ofstream out(path);
    if (!out)
    {
        std::cerr << "write_non_file: cannot open " << path << "\n";
        return;
    }
    std::size_t const h = constraints.rows.size();
    std::size_t const w = constraints.cols.size();

    out << "title \"nonogram_detector\"\n";
    out << "width " << w << "\n";
    out << "height " << h << "\n\n";

    auto const clues = [](std::vector<int> const& line) {
        std::string s;
        for (std::size_t i = 0; i < line.size(); ++i)
        {
            if (i) s += ",";
            s += std::to_string(line[i]);
        }
        return s;
    };

    out << "rows\n";
    for (auto const& line : constraints.rows)
    {
        std::string const s = clues(line);
        out << (s.empty() ? "0" : s.c_str()) << "\n";
    }
    out << "\n";

    out << "columns\n";
    for (auto const& line : constraints.cols)
    {
        std::string const s = clues(line);
        out << (s.empty() ? "0" : s.c_str()) << "\n";
    }
    out << "\n";

    if (!solution.empty())
    {
        std::string goal;
        for (auto const& row : solution)
            for (int const cell : row)
                goal += (cell ? '1' : '0');
        out << "goal \"" << goal << "\"\n";
    }

    // Custom block: source photo reference + grid geometry.
    // Guard against empty detection (e.g. exported from a non-detected image).
    if (!detection_main.empty())
    {
        // #source: relative path from the .non file's directory to the photo.
        std::error_code ec;
        auto rel = std::filesystem::relative(image_path, path.parent_path(), ec);
        auto const source_path = (!ec) ? rel : image_path;
        out << "#source: " << source_path.string() << "\n";

        // #grid: intersection-matrix dimensions (rows x cols).
        out << "#grid: " << detection_main.rows << "x" << detection_main.cols << "\n";

        // #cells: flattened row-major intersection points, one row per line.
        out << "#cells:\n";
        for (int r = 0; r < detection_main.rows; ++r)
        {
            for (int c = 0; c < detection_main.cols; ++c)
            {
                if (c) out << " ";
                cv::Point2f const pt = detection_main.at<cv::Point2f>(r, c);
                out << std::fixed << std::setprecision(2) << pt.x << "," << pt.y;
            }
            out << "\n";
        }
    }

    std::cout << "wrote .non: " << path << "\n";
}


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
            if (x_pos != std::string::npos)
            {
                block.grid_rows = std::atoi(dims.substr(0, x_pos).c_str());
                block.grid_cols = std::atoi(dims.substr(x_pos + 1).c_str());
            }
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
                    if (comma == std::string::npos)
                        continue;
                    float x = parse_float(token.substr(0, comma));
                    float y = parse_float(token.substr(comma + 1));
                    block.cells.emplace_back(x, y);
                }
            }
        }
    }
    return block;
}

}
