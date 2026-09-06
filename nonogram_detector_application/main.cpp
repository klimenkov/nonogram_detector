#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
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
#	include "non_file.hpp"
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

static cv::Mat make_detection_overlay(
    cv::Mat const& image,
    ng::Detection const& detection)
{
    cv::Mat vis = image.clone();
    int const max_dim = std::max(image.cols, image.rows);
    int const radius = std::max(2, max_dim / 700);
    int const line_thickness = std::max(1, max_dim / 1500);

    auto draw_grid = [&](cv::Mat const& mat, cv::Scalar pt_color, cv::Scalar line_color) {
        if (mat.empty()) return;
        for (int r = 0; r < mat.rows; ++r) {
            for (int c = 0; c < mat.cols; ++c) {
                cv::Point2f p1 = mat.at<cv::Point2f>(r, c);
                if (p1.x < 0 || p1.y < 0) continue;
                if (c + 1 < mat.cols) {
                    cv::Point2f p2 = mat.at<cv::Point2f>(r, c + 1);
                    if (p2.x >= 0 && p2.y >= 0)
                        cv::line(vis, p1, p2, line_color, line_thickness, cv::LINE_AA);
                }
                if (r + 1 < mat.rows) {
                    cv::Point2f p2 = mat.at<cv::Point2f>(r + 1, c);
                    if (p2.x >= 0 && p2.y >= 0)
                        cv::line(vis, p1, p2, line_color, line_thickness, cv::LINE_AA);
                }
            }
        }
        for (int r = 0; r < mat.rows; ++r) {
            for (int c = 0; c < mat.cols; ++c) {
                cv::Point2f p = mat.at<cv::Point2f>(r, c);
                if (p.x >= 0 && p.y >= 0)
                    cv::circle(vis, p, radius, pt_color, -1, cv::LINE_AA);
            }
        }
    };

    draw_grid(detection.main, cv::Scalar(0, 0, 255), cv::Scalar(0, 70, 255));
    draw_grid(detection.top, cv::Scalar(0, 230, 0), cv::Scalar(0, 180, 0));
    draw_grid(detection.left, cv::Scalar(255, 180, 0), cv::Scalar(220, 140, 0));

    return vis;
}

bool export_clue_cells(
    std::string const& photo,
    std::vector<std::vector<ng::ClueCellInfo>> const& top_info,
    std::vector<std::vector<ng::ClueCellInfo>> const& left_info,
    std::filesystem::path const& dir,
    cv::Mat const& image,
    ng::Detection const& detection,
    int const resize_max,
    int const threshold_block_size,
    double const threshold_c,
    double const similarity_ratio)
{
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec)
    {
        std::cerr << "NG_EXPORT_CELLS: cannot create " << dir << ": " << ec.message() << "\n";
        return false;
    }

    // Save detection visualization overlay
    if (!image.empty() && detection.found)
    {
        cv::Mat vis = make_detection_overlay(image, detection);
        int const max_vis_dim = std::max(vis.cols, vis.rows);
        if (max_vis_dim > 1600)
        {
            float const factor = 1600.0f / static_cast<float>(max_vis_dim);
            cv::Mat vis_small;
            cv::resize(vis, vis_small, cv::Size(), factor, factor, cv::INTER_AREA);
            vis = vis_small;
        }
        std::vector<int> jpeg_params = { cv::IMWRITE_JPEG_QUALITY, 85 };
        cv::imwrite((dir / "detection.jpg").string(), vis, jpeg_params);
    }

    struct Entry
    {
        int pos, row, col;
        std::string png;
        int predicted, count;
        double whole_conf, conf_l, conf_r;
    };
    std::vector<Entry> entries;
    auto const emit = [&](std::string const& strip,
                          std::vector<std::vector<ng::ClueCellInfo>> const& grid) {
        int pos = 0;
        for (std::size_t row = 0; row < grid.size(); ++row)
            for (std::size_t col = 0; col < grid[row].size(); ++col, ++pos)
            {
                char name[96];
                std::snprintf(name, sizeof name, "%s_%s_%04zu_%04zu.png",
                              photo.c_str(), strip.c_str(), row, col);
                std::filesystem::path p = dir / name;
                if (!grid[row][col].cell.empty())
                    cv::imwrite(p.string(), grid[row][col].cell);
                else
                    cv::imwrite(p.string(), cv::Mat::zeros(20, 20, CV_8UC3));
                entries.push_back({pos, static_cast<int>(row), static_cast<int>(col),
                                   std::string(name),
                                   grid[row][col].digit, grid[row][col].count,
                                   grid[row][col].whole_conf,
                                   grid[row][col].conf_l, grid[row][col].conf_r});
            }
    };
    emit("top", top_info);
    std::size_t top_count = entries.size();
    emit("left", left_info);

    std::ofstream out(dir / "index.json");
    if (!out)
    {
        std::cerr << "NG_EXPORT_CELLS: cannot write index.json in " << dir << "\n";
        return false;
    }
    out << "{\n";
    out << "  \"photo\": \"" << photo << "\",\n";
    out << "  \"meta\": {\n";
    out << "    \"detection_image\": \"detection.jpg\",\n";
    out << "    \"image_width\": " << image.cols << ",\n";
    out << "    \"image_height\": " << image.rows << ",\n";
    if (!detection.main.empty())
    {
        out << "    \"main_grid\": {\"crossings_rows\": " << detection.main.rows
            << ", \"crossings_cols\": " << detection.main.cols
            << ", \"cells_width\": " << std::max(0, detection.main.cols - 1)
            << ", \"cells_height\": " << std::max(0, detection.main.rows - 1) << "},\n";
    }
    if (!detection.top.empty())
    {
        out << "    \"top_grid\": {\"crossings_rows\": " << detection.top.rows
            << ", \"crossings_cols\": " << detection.top.cols
            << ", \"cells_width\": " << std::max(0, detection.top.cols - 1)
            << ", \"cells_height\": " << std::max(0, detection.top.rows - 1) << "},\n";
    }
    if (!detection.left.empty())
    {
        out << "    \"left_grid\": {\"crossings_rows\": " << detection.left.rows
            << ", \"crossings_cols\": " << detection.left.cols
            << ", \"cells_width\": " << std::max(0, detection.left.cols - 1)
            << ", \"cells_height\": " << std::max(0, detection.left.rows - 1) << "},\n";
    }
    out << "    \"params\": {\n";
    out << "      \"resize_max\": " << resize_max << ",\n";
    out << "      \"threshold_block_size\": " << threshold_block_size << ",\n";
    out << "      \"threshold_c\": " << threshold_c << ",\n";
    out << "      \"similarity_ratio_min\": " << similarity_ratio << "\n";
    out << "    }\n";
    out << "  },\n";
    out << "  \"cells\": [\n";
    for (std::size_t i = 0; i < entries.size(); ++i)
    {
        auto const& e = entries[i];
        out << "    {\"pos\":" << e.pos
            << ",\"strip\":\"" << (i < top_count ? "top" : "left") << "\""
            << ",\"row\":" << e.row
            << ",\"col\":" << e.col
            << ",\"png\":\"" << e.png << "\""
            << ",\"predicted\":" << e.predicted
            << ",\"count\":" << e.count
            << ",\"whole_conf\":" << e.whole_conf
            << ",\"conf_l\":" << e.conf_l
            << ",\"conf_r\":" << e.conf_r
            << "}";
        if (i + 1 < entries.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n}\n";
    return true;
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

// Writes the puzzle in webpbn ".non" text format (see ng::write_non_file in
// nonogram_solver/non_file.hpp).

// Renders the solved nonogram overlay onto the original photo: filled cells
// for solution[r][c]==1, thin grid lines connecting all intersections.
// Returns true if the image was written successfully.
bool render_nonogram_overlay(
    cv::Mat const& photo,
    cv::Mat const& main_locs,
    ng::SolutionGrid const& solution,
    std::filesystem::path const& out_path)
{
    if (main_locs.empty() || photo.empty())
    {
        std::cerr << "render_nonogram_overlay: empty input\n";
        return false;
    }

    cv::Mat result = photo.clone();

    int const H = static_cast<int>(solution.size());
    int const W = H > 0 ? static_cast<int>(solution[0].size()) : 0;

    // A cell needs intersections at (r,c), (r,c+1), (r+1,c+1), (r+1,c), so
    // main_locs must be at least (H+1) x (W+1).
    if (main_locs.rows < H + 1 || main_locs.cols < W + 1)
    {
        std::cerr << "render_nonogram_overlay: dimension mismatch\n";
        return false;
    }

    // Fill solved cells (solution[r][c] == 1) with a translucent color onto a
    // single overlay, then blend it onto the result once. Cell (r, c) is
    // bounded by intersections at (r,c), (r,c+1), (r+1,c+1), (r+1,c).
    cv::Scalar const fill_color(60, 20, 10, 255);  // BGR, very dark blue

    cv::Mat overlay = result.clone();
    cv::Point2f const invalid_pt(-1.0f, -1.0f);

    for (int r = 0; r < H; ++r)
    {
        for (int c = 0; c < W; ++c)
        {
            if (solution[r][c] != 1)
                continue;

            cv::Point2f const p0 = main_locs.at<cv::Point2f>(r,     c);
            cv::Point2f const p1 = main_locs.at<cv::Point2f>(r,     c + 1);
            cv::Point2f const p2 = main_locs.at<cv::Point2f>(r + 1, c + 1);
            cv::Point2f const p3 = main_locs.at<cv::Point2f>(r + 1, c);

            if (p0 == invalid_pt || p1 == invalid_pt || p2 == invalid_pt || p3 == invalid_pt)
                continue;

            std::vector<cv::Point> cell_poly(4);
            cell_poly[0] = cv::Point(p0);      // top-left
            cell_poly[1] = cv::Point(p1);      // top-right
            cell_poly[2] = cv::Point(p2);      // bottom-right
            cell_poly[3] = cv::Point(p3);      // bottom-left

            cv::fillConvexPoly(overlay, cell_poly, fill_color);
        }
    }
    cv::addWeighted(overlay, 0.70, result, 0.30, 0, result);

    // Draw grid lines through all intersections for visibility.
    cv::Scalar const line_color(0, 0, 0);  // black
    int const line_thickness = 1;

    // Horizontal lines (connect intersections across each row).
    for (int r = 0; r < main_locs.rows; ++r)
    {
        for (int c = 0; c < main_locs.cols - 1; ++c)
        {
            cv::Point2f const p1 = main_locs.at<cv::Point2f>(r, c);
            cv::Point2f const p2 = main_locs.at<cv::Point2f>(r, c + 1);
            if (p1 != invalid_pt && p2 != invalid_pt)
            {
                cv::line(result, p1, p2, line_color, line_thickness);
            }
        }
    }

    // Vertical lines (connect intersections down each column).
    for (int c = 0; c < main_locs.cols; ++c)
    {
        for (int r = 0; r < main_locs.rows - 1; ++r)
        {
            cv::Point2f const p1 = main_locs.at<cv::Point2f>(r, c);
            cv::Point2f const p2 = main_locs.at<cv::Point2f>(r + 1, c);
            if (p1 != invalid_pt && p2 != invalid_pt)
            {
                cv::line(result, p1, p2, line_color, line_thickness);
            }
        }
    }

    bool const ok = cv::imwrite(out_path.string(), result);
    if (!ok)
        std::cerr << "render_nonogram_overlay: failed to write " << out_path << "\n";
    return ok;
}


// Runs the clue-correction + solve + export pipeline on the decoded clues.
// Both clue strips are required: rows come from the left clues and columns
// from the top clues. decode_clues succeeds when either strip decoded (so
// partial clues still print in main), but solving with a strip missing would
// only surface as an opaque "invalid constraints" solver failure, so check
// up front and say why instead.
void solve_and_export(
    ng::ClueGrid const& clues,
    cv::Mat const& image,
    cv::Mat const& main_locs,
    std::string const& image_path)
{
    if (clues.top.empty() || clues.left.empty())
    {
        std::cout << "cannot solve: incomplete clue strips (need both top and left)\n";
        return;
    }

    // Grid dimensions: H rows from the left strip, W columns from the top
    // strip. Each left line maps to one grid row (fits grid width W); the
    // top strip is transposed so each grid column maps to one clue line
    // (fits grid height H).
    std::size_t const H = clues.left.size();
    std::size_t const W = clues.top[0].size();

    // Convert the decoded strips into the corrector's cell grids: rows =
    // left lines (as decoded), cols = transposed top lines. Both go through
    // the same clueline helper so the digit_count fallback (no valid counter
    // entry -> 1 for a read digit, 0 for an empty cell) is applied
    // identically to rows and columns.
    auto const clueline = [](std::vector<int> const& digits,
                             std::vector<int> const& counts) {
        std::vector<ng::ClueCell> line(digits.size());
        for (std::size_t i = 0; i < digits.size(); ++i)
        {
            line[i].digit = digits[i];
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
    for (std::size_t c = 0; c < W; ++c)
    {
        std::vector<int> digits, counts;
        for (std::size_t r = 0; r < clues.top.size() && c < clues.top[r].size(); ++r)
        {
            digits.push_back(clues.top[r][c]);
            counts.push_back(clues.top_count[r][c]);
        }
        decoded.cols[c] = clueline(digits, counts);
    }

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
        if (char const* export_path = std::getenv("NG_EXPORT_NON"))
            ng::write_non_file(export_path, constraints, result.solution,
                               main_locs, image_path);
        if (char const* overlay_path = std::getenv("NG_EXPORT_OVERLAY"))
        {
            if (render_nonogram_overlay(image, main_locs,
                                        result.solution, overlay_path))
                std::cout << "wrote overlay: " << overlay_path << "\n";
        }
    }
    else
    {
        std::cout << "no solution\n";
    }

    // Dump the corrected puzzle (clues + optional solution) as JSON whenever
    // the environment asks, so downstream tooling can render the recognition
    // and solving stages without re-running recognition.
    if (char const* dump_path = std::getenv("NG_EXPORT_PUZZLE"))
    {
        std::ofstream out(dump_path);
        if (!out)
        {
            std::cerr << "NG_EXPORT_PUZZLE: cannot write " << dump_path << "\n";
        }
        else
        {
            auto const group = [](std::vector<std::vector<int>> const& lines) {
                std::string s = "[";
                for (std::size_t i = 0; i < lines.size(); ++i)
                {
                    if (i) s += ",";
                    s += "[";
                    for (std::size_t j = 0; j < lines[i].size(); ++j)
                    {
                        if (j) s += ",";
                        s += std::to_string(lines[i][j]);
                    }
                    s += "]";
                }
                return s + "]";
            };
            out << "{\n";
            out << "  \"width\": " << W << ",\n";
            out << "  \"height\": " << H << ",\n";
            out << "  \"rows\": " << group(constraints.rows) << ",\n";
            out << "  \"columns\": " << group(constraints.cols) << ",\n";
            out << "  \"solved\": " << (result.solved ? "true" : "false") << "\n";
            if (result.solved)
            {
                std::string goal;
                for (auto const& row : result.solution)
                    for (int const cell : row)
                        goal += (cell ? '1' : '0');
                out << "  ,\"goal\": \"" << goal << "\"\n";
            }
            out << "}\n";
            std::cout << "wrote puzzle: " << dump_path << "\n";
        }
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
    int const threshold_block_size = 15;
    double const threshold_c = 10.0;
    double const similarity_ratio_min = 0.80;

    auto image = cv::imread(image_path);
    if (image.empty())
    {
        std::cerr << "Image was not read: " << image_path << "\n";
        return 1;
    }

    ng::CrossLocsDetector cross_loc_detector(
        resize_max, threshold_block_size, threshold_c, 5, 50, similarity_ratio_min);

    auto const detection = cross_loc_detector.detect(image);

    std::cout << "found=" << (detection.found ? "true" : "false") << "\n";
    std::cout << "main=" << detection.main.size()
              << " top=" << detection.top.size()
              << " left=" << detection.left.size() << "\n";

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

    // The digit model is required for any clue decoding: its constructor
    // throws on a missing/corrupt ONNX file (e.g. the fallback relative path
    // only resolves from the repo root). Exit cleanly with a message, like
    // every other failure path in main().
    std::unique_ptr<ng::DigitRecognizer> recognizer_ptr;
    try
    {
        recognizer_ptr = std::make_unique<ng::DigitRecognizer>(model_path);
    }
    catch (std::exception const& e)
    {
        std::cerr << "digit model load failed: " << e.what() << "\n";
        return 1;
    }
    auto& recognizer = *recognizer_ptr;

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
    {
        try
        {
            recognizer.set_counter_model(counter_path.string());
        }
        catch (std::exception const& e)
        {
            // The counter model is optional: continue without two-digit
            // support rather than failing the whole run.
            std::cerr << "counter model load failed: " << e.what()
                      << " (continuing without two-digit support)\n";
        }
    }

    ng::ClueGrid clues;
    if (ng::decode_clues(image, detection, recognizer, clues))
    {

        std::cout << "top clues:\n";
        print_clue_grid(clues.top);
        std::cout << "left clues:\n";
        print_clue_grid(clues.left);

        if (char const* cells_dir = std::getenv("NG_EXPORT_CELLS"))
        {
            std::vector<std::vector<ng::ClueCellInfo>> top_info, left_info;
            if (ng::decode_clues_ex(image, detection, recognizer, clues,
                                    top_info, left_info))
            {
                std::filesystem::path photo_id =
                    std::filesystem::path(image_path).stem();
                export_clue_cells(
                    photo_id.string(),
                    top_info,
                    left_info,
                    cells_dir,
                    image,
                    detection,
                    resize_max,
                    threshold_block_size,
                    threshold_c,
                    similarity_ratio_min);
            }
            else
            {
                std::cerr << "NG_EXPORT_CELLS: decode_clues_ex produced no strips\n";
            }
        }

#ifdef NG_ENABLE_SOLVER
        solve_and_export(clues, image, detection.main, image_path);
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
