#include <algorithm>
#include <array>

#include <iterator>
#include <numeric>
#include <set>
#include <tuple>
#include <queue>
#include <vector>

#include "cross_locs_detector.hpp"
#include "grid_smooth_fit.hpp"
#include "image_operations.hpp"
#include "masks.hpp"

namespace ng
{

cv::Point const CrossLocsDetector::INDICES_DELTA_UP(0, -1);
cv::Point const CrossLocsDetector::INDICES_DELTA_RIGHT(1, 0);
cv::Point const CrossLocsDetector::INDICES_DELTA_DOWN(0, 1);
cv::Point const CrossLocsDetector::INDICES_DELTA_LEFT(-1, 0);

std::vector<cv::Point> const CrossLocsDetector::INDICES_DELTAS = { INDICES_DELTA_UP, INDICES_DELTA_RIGHT, INDICES_DELTA_DOWN, INDICES_DELTA_LEFT };


CrossLocsDetector::CrossLocsDetector(
    float const resize_width_height_max,
    int const threshold_block_size,
    double const threshold_c,
    int const find_cell_side_length_min,
    int const find_cell_side_length_max,
    double const similarity_ratio_min)
    : M_RESIZE_WIDTH_HEIGHT_MAX(resize_width_height_max)
    , M_THRESHOLD_BLOCK_SIZE(threshold_block_size)
    , M_THRESHOLD_C(threshold_c)
    , M_FIND_CELL_SIDE_LENGTH_MIN(find_cell_side_length_min)
    , M_FIND_CELL_SIDE_LENGTH_MAX(find_cell_side_length_max)
    , M_SIMILARITY_RATIO_MIN(similarity_ratio_min)
{
}


int CrossLocsDetector::estimate_cell_side_length(
    cv::Mat const& image_thresholded,
    cv::Rect const& roi,
    int const min, int const max)
{
    // Autocorrelation of a 1-D projection: the lag with the first strong
    // positive peak (other than lag 0) is the grid period = cell side length.
    auto period_of = [](std::vector<double> const& sig) -> int {
        int const n = (int)sig.size();
        if (n < 8) return 0;
        double mean = 0;
        for (double v : sig) mean += v;
        mean /= n;
        std::vector<double> z(n);
        for (int i = 0; i < n; ++i) z[i] = sig[i] - mean;
        double var = 0;
        for (double v : z) var += v * v;
        if (var <= 1e-9) return 0;
        // Normalized autocorrelation per lag: correlation of the projection
        // with itself shifted by <lag>. A periodic grid's projection has a
        // strong peak at every multiple of the true period (cell side length);
        // the FUNDAMENTAL period is therefore the first (smallest-lag) strong
        // local maximum, not the argmax (which arbitrarily selects a harmonic).
        int const half = n / 2;
        std::vector<double> norm(half + 1, 0.0);
        for (int lag = 1; lag <= half; ++lag) {
            double acc = 0, var_ov = 0;
            for (int i = 0; i + lag < n; ++i) {
                acc += z[i] * z[i + lag];
                var_ov += z[i] * z[i];
            }
            if (var_ov > 1e-12) norm[lag] = acc / var_ov;
        }
        // Smallest lag that is a strict/local strong peak.
        for (int lag = 2; lag < half; ++lag) {
            if (norm[lag] >= norm[lag - 1] &&
                norm[lag] >= norm[lag + 1] &&
                norm[lag] >= 0.5)
            {
                return lag;
            }
        }
        return 0;
    };

    cv::Mat roi_image = image_thresholded(roi);
    cv::Mat col_proj, row_proj;
    cv::reduce(roi_image, col_proj, 1, cv::REDUCE_AVG, CV_64F);  // per-row mean
    cv::reduce(roi_image, row_proj, 0, cv::REDUCE_AVG, CV_64F);  // per-col mean

    std::vector<double> rp(row_proj.begin<double>(), row_proj.end<double>());
    std::vector<double> cp(col_proj.begin<double>(), col_proj.end<double>());

    int p_row = period_of(rp);
    int p_col = period_of(cp);

    int est = 0;
    if (p_row && p_col) est = (p_row + p_col) / 2;
    else if (p_row) est = p_row;
    else if (p_col) est = p_col;

    if (est < min || est > max) return 0;
    return est;
}


Detection CrossLocsDetector::detect(cv::Mat const& image)
{
    // INTER_LINEAR downscaling preserves the digit strips exactly as the
    // recognizer expects, so it stays the primary pass. But it attenuates
    // thin grid rules: when a photo's crossings never reach the strict
    // similarity ratio it drops every junction (empty grid) even though the
    // grid is plainly visible. INTER_AREA averages pixels on downscale and
    // keeps those thin rules intact, so retry with it when the LINEAR pass
    // finds no grid.
    Detection detection =
        detect_impl(image, cv::InterpolationFlags::INTER_LINEAR);
    if (!detection.found)
    {
        detection = detect_impl(image, cv::InterpolationFlags::INTER_AREA);
    }
    return detection;
}


Detection CrossLocsDetector::detect_impl(
    cv::Mat const& image,
    cv::InterpolationFlags const interpolation_flag)
{
    cv::Mat image_resized;
    float scale;
    std::tie(image_resized, scale) =
        resize(image, M_RESIZE_WIDTH_HEIGHT_MAX, interpolation_flag);

    cv::Mat image_gray;
    cv::cvtColor(image_resized, image_gray, cv::COLOR_BGR2GRAY);

    auto const image_thresholded =
        threshold(image_gray, M_THRESHOLD_BLOCK_SIZE, M_THRESHOLD_C);

    Detection detection;

    bool cell_loc_found = false;
    int cell_side_length = 0;
    cv::Point2f cell_loc(0.0f, 0.0f);
    cv::Point const image_center(image_thresholded.size() / 2);
    auto const cell_loc_roi = get_roi(image_center, { 150, 150 });

    int const est = estimate_cell_side_length(
        image_thresholded, cell_loc_roi,
        M_FIND_CELL_SIDE_LENGTH_MIN, M_FIND_CELL_SIDE_LENGTH_MAX);
    if (est > 0)
    {
        cv::Mat mask_square;
        int mask_square_perimeter;
        std::tie(mask_square, mask_square_perimeter) = get_mask_square(est);
        std::tie(cell_loc_found, cell_loc) = find_kernel_loc(
            image_thresholded, cell_loc_roi, mask_square,
            mask_square_perimeter, M_SIMILARITY_RATIO_MIN, cv::Point(0, 0));
        if (cell_loc_found) cell_side_length = est;
    }
    if (!cell_loc_found)
    {
        std::tie(cell_loc_found, cell_side_length, cell_loc) =
            find_cell_side_length_cell_loc(
                image_thresholded, cell_loc_roi,
                M_FIND_CELL_SIDE_LENGTH_MIN, M_FIND_CELL_SIDE_LENGTH_MAX,
                M_SIMILARITY_RATIO_MIN);
    }

    if (!cell_loc_found)
    {
        return detection;
    }

    // Grid-junction match strength varies with the photo (thin/broken rulers,
    // ink bleeding into the cells near crossings, printed grid corners). The
    // strict ratio is tuned for crisp grids; some photos never reach it at any
    // junction, so the cross walk aborts on the very first step and the grid
    // comes back empty. Retry the whole propagation with a relaxed ratio and
    // keep the result only when it is a solid grid (a tiny blob would be a
    // degenerate match, not a detected puzzle).
    double similarity_ratio_min = M_SIMILARITY_RATIO_MIN;

    cv::Mat cross_locs_main_mat = get_cross_locs_main_mat(
        image_thresholded,
        cell_loc,
        cell_side_length,
        similarity_ratio_min);

    if (cross_locs_main_mat.empty())
    {
        double const similarity_ratio_min_loose = 0.7;
        cross_locs_main_mat = get_cross_locs_main_mat(
            image_thresholded,
            cell_loc,
            cell_side_length,
            similarity_ratio_min_loose);
        if (!cross_locs_main_mat.empty() &&
            (cross_locs_main_mat.cols < 8 || cross_locs_main_mat.rows < 8))
        {
            cross_locs_main_mat = cv::Mat();
        }
        else
        {
            similarity_ratio_min = similarity_ratio_min_loose;
        }
    }

    if (cross_locs_main_mat.empty())
    {
        return detection;
    }

    // Second refinement stage: snap each found cross to the geometric center
    // of the drawn line intersection (ink centroid of the grayscale). Runs
    // before scaling and before the top/left searches so their seed positions
    // inherit the refined locations.
    int const ink_window_radius = std::max(4, cell_side_length / 4);
    refine_cross_locs_ink(image_gray, cross_locs_main_mat, ink_window_radius);

    cross_locs_main_mat = grid_smooth_fit_approach1(cross_locs_main_mat, 3, 2);

    detection.found = true;
    detection.main = scale_cross_locs_mat(cross_locs_main_mat, scale);

    auto cross_locs_top_mat = get_cross_locs_top_mat(
        image_thresholded,
        cross_locs_main_mat,
        cell_side_length,
        similarity_ratio_min);

    if (!cross_locs_top_mat.empty())
    {
        detection.top = scale_cross_locs_mat(cross_locs_top_mat, scale);
    }

    auto cross_locs_left_mat = get_cross_locs_left_mat(
        image_thresholded,
        cross_locs_main_mat,
        cell_side_length,
        similarity_ratio_min);

    if (!cross_locs_left_mat.empty())
    {
        detection.left = scale_cross_locs_mat(cross_locs_left_mat, scale);
    }

    return detection;
}


std::tuple<bool, int, cv::Point2f> CrossLocsDetector::find_cell_side_length_cell_loc(
    cv::Mat const& image_thresholded,
    cv::Rect const& image_thresholded_roi,
    int const cell_side_length_min,
    int const cell_side_length_max,
    double const similarity_ratio_min)
{
    for (auto cell_side_length = cell_side_length_min; cell_side_length <= cell_side_length_max; ++cell_side_length)
    {
        cv::Mat mask_square;
        int mask_square_perimeter;
        std::tie(mask_square, mask_square_perimeter) = get_mask_square(cell_side_length);

        bool cell_loc_found;
        cv::Point2f cell_loc;
        std::tie(cell_loc_found, cell_loc) = find_kernel_loc(
            image_thresholded,
            image_thresholded_roi,
            mask_square,
            mask_square_perimeter,
            similarity_ratio_min,
            cv::Point(0, 0));

        if (cell_loc_found)
        {
            return std::make_tuple(true, cell_side_length, cell_loc);
        }
    }

    return std::make_tuple(false, -1, cv::Point2f(-1.0f, -1.0f));
}


cv::Size CrossLocsDetector::get_cross_loc_search_roi(int const cell_side_length)
{
    // A cross is predicted from its immediate neighbor at exactly
    // cell_side_length spacing. The cross kernel itself spans ~1.5 a cell side
    // (see get_mask_cross), so a 1x box (radius cell_side_length/2) cannot
    // contain the full cross pattern for filter2D, and 1x proved too tight on
    // both synthetic and real grids. A 1.5x box (radius 3*cell_side_length/4)
    // just covers the cross extent plus small per-step drift, and is ~56%
    // smaller in area than the previous 2x box.
    return cv::Size(3 * cell_side_length / 2, 3 * cell_side_length / 2);
}


cv::Size CrossLocsDetector::get_cross_loc_search_roi_top(int const cell_side_length)
{
    // For top clue search, columns run vertically. To avoid jumping to adjacent
    // columns (e.g. thicker outer borders ~1.0 cell side away), limit horizontal
    // search radius to cell_side_length / 3, while allowing standard height.
    int const rx = std::max(4, cell_side_length / 3);
    int const ry = 3 * cell_side_length / 4;
    return cv::Size(2 * rx + 1, 2 * ry + 1);
}


cv::Size CrossLocsDetector::get_cross_loc_search_roi_left(int const cell_side_length)
{
    // For left clue search, rows run horizontally. To avoid jumping to adjacent
    // rows, limit vertical search radius to cell_side_length / 3.
    int const rx = 3 * cell_side_length / 4;
    int const ry = std::max(4, cell_side_length / 3);
    return cv::Size(2 * rx + 1, 2 * ry + 1);
}


std::map<cv::Point, cv::Point2f, PointCompare> CrossLocsDetector::get_cross_locs_map(
    cv::Mat const& image_thresholded,
    std::vector<cv::Point> const& indices_init,
    std::vector<cv::Point2f> const& cross_locs_init,
    std::vector<cv::Point> const& indices_deltas,
    std::vector<cv::Point2f> const& cross_loc_deltas,
    cv::Size const roi_size,
    cv::Mat const& mask_cross,
    int const mask_cross_perimeter,
    double const similarity_ratio_min,
    int const cell_side_length,
    int const min_x,
    int const max_x,
    int const min_y,
    int const max_y)
{
    std::queue<cv::Point> indices_queue;
    std::set<cv::Point, PointCompare> was_in_indices_queue_set;
    for (auto const& indices_init : indices_init)
    {
        indices_queue.push(indices_init);
        was_in_indices_queue_set.insert(indices_init);
    }

    // Stores predicted initial values
    std::map<cv::Point, cv::Point2f, PointCompare> cross_locs_init_map;
    for (int i = 0; i < indices_init.size(); ++i)
    {
        cross_locs_init_map[indices_init[i]] = cross_locs_init[i];
    }

    std::map<cv::Point, cv::Point2f, PointCompare> cross_locs_map;

    while (!indices_queue.empty())
    {
        auto const indices = indices_queue.front();
        indices_queue.pop();

        auto const& cross_loc_init = cross_locs_init_map[indices];

        bool cross_loc_found;
        cv::Point2f cross_loc;
        cv::Point const cross_loc_init_rounded(
            cvRound(cross_loc_init.x), cvRound(cross_loc_init.y));
        std::tie(cross_loc_found, cross_loc) = find_kernel_loc(
            image_thresholded,
            get_roi(cross_loc_init_rounded, roi_size),
            mask_cross,
            mask_cross_perimeter,
            similarity_ratio_min);

        if (cross_loc_found)
        {
            // Reject candidate if it snapped onto an adjacent already-known crossing (prevents degenerate loops)
            bool too_close = false;
            if (cell_side_length > 0)
            {
                for (size_t d = 0; d < indices_deltas.size(); ++d)
                {
                    auto it_adj = cross_locs_map.find(indices - indices_deltas[d]);
                    if (it_adj != cross_locs_map.end())
                    {
                        if (cv::norm(cross_loc - it_adj->second) < 0.6f * cell_side_length)
                        {
                            too_close = true;
                            break;
                        }
                    }
                }
            }
            if (too_close)
            {
                continue;
            }

            cross_locs_map[indices] = cross_loc;

            for (int i = 0; i < indices_deltas.size(); ++i)
            {
                auto const indices_neighbor = indices + indices_deltas[i];
                if (indices_neighbor.x < min_x || indices_neighbor.x > max_x ||
                    indices_neighbor.y < min_y || indices_neighbor.y > max_y)
                {
                    continue;
                }

                bool const was_in_indices_queue =
                    was_in_indices_queue_set.find(indices_neighbor) != was_in_indices_queue_set.end();

                if (!was_in_indices_queue)
                {
                    indices_queue.push(indices_neighbor);
                    was_in_indices_queue_set.insert(indices_neighbor);

                    cv::Point const indices_prev = indices - indices_deltas[i];
                    cv::Point2f step = cross_loc_deltas[i];
                    auto it_prev = cross_locs_map.find(indices_prev);
                    if (it_prev != cross_locs_map.end())
                    {
                        cv::Point2f const candidate_step = cross_loc - it_prev->second;
                        float const d = cv::norm(candidate_step);
                        if (cell_side_length > 0)
                        {
                            if (d >= 0.6f * cell_side_length && d <= 1.6f * cell_side_length)
                            {
                                step = candidate_step;
                            }
                        }
                        else
                        {
                            step = candidate_step;
                        }
                    }
                    auto const cross_loc_neighbor_init = cross_loc + step;
                    cross_locs_init_map[indices_neighbor] = cross_loc_neighbor_init;
                }
            }
        }
    }

    return cross_locs_map;
}


cv::Rect CrossLocsDetector::get_bounding_rectangle(
    std::map<cv::Point, cv::Point2f, PointCompare> const& cross_locs_map)
{
    auto const x_min_max_it = std::minmax_element(
        cross_locs_map.cbegin(),
        cross_locs_map.cend(),
        [](auto const& p_1, auto const& p_2)
        {
            auto const& x_1 = p_1.first.x;
            auto const& x_2 = p_2.first.x;

            return x_1 < x_2;
        });

    auto const y_min_max_it = std::minmax_element(
        cross_locs_map.cbegin(),
        cross_locs_map.cend(),
        [](auto const& p_1, auto const& p_2)
        {
            auto const& y_1 = p_1.first.y;
            auto const& y_2 = p_2.first.y;

            return y_1 < y_2;
        });

    cv::Rect const bounding_rectangle(
        cv::Point(x_min_max_it.first->first.x, y_min_max_it.first->first.y),
        cv::Point(x_min_max_it.second->first.x, y_min_max_it.second->first.y));

    return bounding_rectangle;
}


cv::Mat CrossLocsDetector::convert_to_mat(
    std::map<cv::Point, cv::Point2f, PointCompare> const& cross_locs_map)
{
    if (cross_locs_map.empty())
    {
        return cv::Mat();
    }

    return convert_to_mat(cross_locs_map, get_bounding_rectangle(cross_locs_map));
}


cv::Mat CrossLocsDetector::convert_to_mat(
    std::map<cv::Point, cv::Point2f, PointCompare> const& cross_locs_map,
    cv::Rect const& bounding_rectangle)
{
    if (cross_locs_map.empty() || bounding_rectangle.empty())
    {
        return cv::Mat();
    }

    auto const cross_loc_mat_size = bounding_rectangle.size() + cv::Size(1, 1);

    cv::Mat cross_locs_mat(cross_loc_mat_size, CV_32FC2, cv::Scalar(-1.0, -1.0));

    for (auto x_map = bounding_rectangle.tl().x, x_mat = 0; x_map <= bounding_rectangle.br().x; ++x_map, ++x_mat)
    {
        for (auto y_map = bounding_rectangle.tl().y, y_mat = 0; y_map <= bounding_rectangle.br().y; ++y_map, ++y_mat)
        {
            cv::Point indices_map(x_map, y_map);
            cv::Point indices_mat(x_mat, y_mat);

            auto const indices_cross_loc_it = cross_locs_map.find(indices_map);
            if (indices_cross_loc_it != cross_locs_map.end())
            {
                cross_locs_mat.at<cv::Point2f>(indices_mat) = indices_cross_loc_it->second;
            }
        }
    }

    return cross_locs_mat;
}


cv::Mat CrossLocsDetector::augment(
    cv::Mat const& cross_locs_mat,
    int const cell_side_length)
{
    std::set<cv::Point, PointCompare> indices_empty_set;

    for (int y = 0; y < cross_locs_mat.rows; ++y)
    {
        for (int x = 0; x < cross_locs_mat.cols; ++x)
        {
            cv::Point indices(x, y);

            if (cross_locs_mat.at<cv::Point2f>(indices) == cv::Point2f(-1.0f, -1.0f))
            {
                indices_empty_set.insert(indices);
            }
        }
    }

    cv::Rect const indices_roi(cv::Point(0, 0), cross_locs_mat.size());
    cv::Mat cross_locs_mat_augmented = cross_locs_mat.clone();

    while (!indices_empty_set.empty())
    {
        std::map<cv::Point, cv::Point2f, PointCompare> indices_cross_locs_interpolated_map;
        for (auto const& indices : indices_empty_set)
        {
            std::vector<cv::Point2f> cross_locs_interpolated;

            for (auto const& indices_delta : INDICES_DELTAS)
            {
                auto const indices_neighbor_1 = indices + indices_delta;
                auto const indices_neighbor_2 = indices_neighbor_1 + indices_delta;

                bool const n1_in_range = indices_roi.contains(indices_neighbor_1);
                if (n1_in_range && cross_locs_mat_augmented.at<cv::Point2f>(indices_neighbor_1) != cv::Point2f(-1.0f, -1.0f))
                {
                    cv::Point2f const p1 = cross_locs_mat_augmented.at<cv::Point2f>(indices_neighbor_1);
                    bool const n2_in_range = indices_roi.contains(indices_neighbor_2);
                    cv::Point2f cross_loc_interpolated;
                    if (n2_in_range && cross_locs_mat_augmented.at<cv::Point2f>(indices_neighbor_2) != cv::Point2f(-1.0f, -1.0f))
                    {
                        cv::Point2f const p2 = cross_locs_mat_augmented.at<cv::Point2f>(indices_neighbor_2);
                        cross_loc_interpolated = p1 + (p1 - p2);
                    }
                    else
                    {
                        auto const direction = indices - indices_neighbor_1;
                        cv::Point2f const direction_float(
                            static_cast<float>(direction.x), static_cast<float>(direction.y));
                        cross_loc_interpolated =
                            p1 + static_cast<float>(cell_side_length) * direction_float;
                    }
                    cross_locs_interpolated.push_back(cross_loc_interpolated);
                }
            }

            if (!cross_locs_interpolated.empty())
            {
                auto const cross_locs_interpolated_sum = std::accumulate(
                    cross_locs_interpolated.begin(),
                    cross_locs_interpolated.end(),
                    cv::Point2f());
                auto const cross_locs_interpolated_n = static_cast<float>(cross_locs_interpolated.size());
                auto const cross_loc_interpolated =
                    cross_locs_interpolated_sum / cross_locs_interpolated_n;

                indices_cross_locs_interpolated_map[indices] = cross_loc_interpolated;
            }
        }

        for (auto const& indices_cross_loc_interpolated : indices_cross_locs_interpolated_map)
        {
            cv::Point indices;
            cv::Point2f cross_loc_interpolated;
            std::tie(indices, cross_loc_interpolated) = indices_cross_loc_interpolated;

            cross_locs_mat_augmented.at<cv::Point2f>(indices) = cross_loc_interpolated;

            indices_empty_set.erase(indices);
        }
    }

    return cross_locs_mat_augmented;
}


cv::Mat CrossLocsDetector::convert_pad_augment(
    std::map<cv::Point, cv::Point2f, PointCompare> const& cross_locs_map,
    cv::Point const& offset,
    cv::Size const& pad,
    int const cell_side_length)
{
    if (cross_locs_map.empty())
    {
        return cv::Mat();
    }

    return convert_pad_augment(
        cross_locs_map, get_bounding_rectangle(cross_locs_map), offset, pad, cell_side_length);
}


cv::Mat CrossLocsDetector::convert_pad_augment(
    std::map<cv::Point, cv::Point2f, PointCompare> const& cross_locs_map,
    cv::Rect const& bounding_rectangle,
    cv::Point const& offset,
    cv::Size const& pad,
    int const cell_side_length)
{
    auto const cross_locs_mat = convert_to_mat(cross_locs_map, bounding_rectangle);

    if (cross_locs_mat.empty())
    {
        return cv::Mat();
    }

    // Embed into a sentinel-filled canvas: <pad> extra entries overall, with
    // the searched matrix placed at <offset>.
    cv::Mat cross_locs_padded_mat(
        cross_locs_mat.size() + pad,
        cross_locs_mat.type(),
        cv::Scalar(-1.0, -1.0));

    cv::Rect const roi(offset, cross_locs_mat.size());
    cross_locs_mat.copyTo(cross_locs_padded_mat(roi));

    return augment(cross_locs_padded_mat, cell_side_length);
}


cv::Mat CrossLocsDetector::get_cross_locs_main_mat(
    cv::Mat const& image_thresholded,
    cv::Point2f const& cross_loc_init,
    int const cell_side_length,
    double const similarity_ratio_min)
{
    auto const mask_length = static_cast<int>(cell_side_length * 1.5f);
    auto const mask_length_odd = mask_length / 2 * 2 + 1;

    auto const line_width = static_cast<int>(cell_side_length / 4);
    auto const line_width_half = line_width / 2;

    cv::Mat mask_cross;
    int mask_cross_perimeter;
    std::tie(mask_cross, mask_cross_perimeter) =
        get_mask_cross(mask_length_odd, line_width_half);

    std::vector<cv::Point2f> const cross_loc_deltas = {
        cv::Point2f(0.0f, -cell_side_length),
        cv::Point2f(cell_side_length, 0.0f),
        cv::Point2f(0.0f, cell_side_length),
        cv::Point2f(-cell_side_length, 0.0f) };

    auto const cross_locs_main_map = get_cross_locs_map(
        image_thresholded,
        { cv::Point(0, 0) },
        { cross_loc_init },
        INDICES_DELTAS,
        cross_loc_deltas,
        get_cross_loc_search_roi(cell_side_length),
        mask_cross,
        mask_cross_perimeter,
        similarity_ratio_min,
        cell_side_length);

    // Add extra lines on perimeter
    return convert_pad_augment(
        cross_locs_main_map, cv::Point(1, 1), cv::Size(2, 2), cell_side_length);
}


cv::Mat CrossLocsDetector::get_cross_locs_top_mat(
    cv::Mat const& image_thresholded,
    cv::Mat const& cross_locs_main_mat,
    int const cell_side_length,
    double const similarity_ratio_min)
{
    std::vector<cv::Point> indices_neighbors_init;
    std::vector<cv::Point2f> cross_locs_neighbors_init;

    // The last is not a cross
    for (auto x = 0; x < cross_locs_main_mat.cols - 1; ++x)
    {
        cv::Point const indices(x, 0);
        auto const& cross_loc = cross_locs_main_mat.at<cv::Point2f>(indices);

        if (cross_loc != cv::Point2f(-1.0f, -1.0f))
        {
            indices_neighbors_init.push_back(indices);

            auto const cross_loc_neighbor_init = cross_loc;
            cross_locs_neighbors_init.push_back(cross_loc_neighbor_init);
        }
    }

    std::vector<cv::Point> const indices_deltas = {
        cv::Point(0, -1),
        cv::Point(1, 0),
        cv::Point(-1, 0) };

    std::vector<cv::Point2f> const cross_loc_deltas = {
        cv::Point2f(0.0f, -cell_side_length),
        cv::Point2f(cell_side_length, 0.0f),
        cv::Point2f(-cell_side_length, 0.0f) };

    auto const cell_side_length_odd = cell_side_length / 2 * 2 + 1;

    cv::Mat mask_cross;
    int mask_cross_perimeter;
    std::tie(mask_cross, mask_cross_perimeter) =
        get_mask_cross(cell_side_length_odd);

    int const max_top_clue_rows = std::min(30, std::max(12, cross_locs_main_mat.rows / 2));

    auto cross_locs_top_map = get_cross_locs_map(
        image_thresholded,
        indices_neighbors_init,
        cross_locs_neighbors_init,
        indices_deltas,
        cross_loc_deltas,
        get_cross_loc_search_roi_top(cell_side_length),
        mask_cross,
        mask_cross_perimeter,
        similarity_ratio_min,
        cell_side_length,
        0, cross_locs_main_mat.cols - 1,
        -max_top_clue_rows, 0);

    if (cross_locs_top_map.empty())
    {
        return cv::Mat();
    }

    auto bbox = get_bounding_rectangle(cross_locs_top_map);

    if (bbox.br().x < cross_locs_main_mat.cols - 2)
    {
        std::vector<cv::Point> missing_indices;
        std::vector<cv::Point2f> missing_locs;
        for (int x = bbox.br().x + 1; x < cross_locs_main_mat.cols - 1; ++x)
        {
            cv::Point const idx(x, 0);
            auto const& pt = cross_locs_main_mat.at<cv::Point2f>(idx);
            if (pt != cv::Point2f(-1.0f, -1.0f))
            {
                missing_indices.push_back(idx);
                missing_locs.push_back(pt);
            }
        }

        if (!missing_indices.empty())
        {
            double const similarity_ratio_min_loose = 0.75;
            auto const loose_map = get_cross_locs_map(
                image_thresholded,
                missing_indices,
                missing_locs,
                indices_deltas,
                cross_loc_deltas,
                get_cross_loc_search_roi_top(cell_side_length),
                mask_cross,
                mask_cross_perimeter,
                similarity_ratio_min_loose,
                cell_side_length,
                bbox.br().x + 1, cross_locs_main_mat.cols - 1,
                -max_top_clue_rows, 0);

            int const min_x = bbox.br().x + 1;
            for (auto const& item : loose_map)
            {
                if (item.first.x >= min_x && item.first.x < cross_locs_main_mat.cols - 1)
                {
                    cross_locs_top_map[item.first] = item.second;
                }
            }
        }
        bbox = get_bounding_rectangle(cross_locs_top_map);
    }

    // Filter out rows with too few points to avoid false-positive spikes from headers/banners
    int const min_points_for_row = std::min(3, std::max(1, (cross_locs_main_mat.cols - 1) / 4));
    int effective_min_y = 0;
    for (int y = -1; y >= bbox.tl().y; --y)
    {
        int cnt = 0;
        for (auto const& item : cross_locs_top_map)
        {
            if (item.first.y == y) cnt++;
        }
        if (cnt >= min_points_for_row)
        {
            effective_min_y = y;
        }
    }

    int const num_clue_rows = std::abs(effective_min_y);
    if (num_clue_rows <= 0)
    {
        return cv::Mat();
    }
    int const num_top_rows = num_clue_rows + 2;
    int const num_cols = cross_locs_main_mat.cols;

    std::vector<std::vector<float>> row_h(num_top_rows, std::vector<float>(num_cols, 0.0f));

    for (int y = -1; y >= -num_clue_rows; --y)
    {
        int const r = num_top_rows - 1 + y;
        std::vector<double> xs, hs;
        for (int c = 0; c < num_cols - 1; ++c)
        {
            auto it = cross_locs_top_map.find(cv::Point(c, y));
            if (it != cross_locs_top_map.end())
            {
                int const k = std::min(5, cross_locs_main_mat.rows - 1);
                cv::Point2f const p0 = cross_locs_main_mat.at<cv::Point2f>(0, c);
                cv::Point2f const pk = cross_locs_main_mat.at<cv::Point2f>(k, c);
                cv::Point2f u = p0 - pk;
                float const len = cv::norm(u);
                if (len > 1e-3f)
                {
                    u /= len;
                    float const h = (it->second - p0).dot(u);
                    xs.push_back(c);
                    hs.push_back(h);
                }
            }
        }

        if (xs.size() >= 2)
        {
            double sum_x = 0, sum_y = 0;
            for (size_t i = 0; i < xs.size(); ++i)
            {
                sum_x += xs[i];
                sum_y += hs[i];
            }
            double mean_x = sum_x / xs.size();
            double mean_y = sum_y / xs.size();
            double sxx = 0, sxy = 0;
            for (size_t i = 0; i < xs.size(); ++i)
            {
                double const dx = xs[i] - mean_x;
                double const dy = hs[i] - mean_y;
                sxx += dx * dx;
                sxy += dx * dy;
            }
            double b = (sxx > 1e-6) ? (sxy / sxx) : 0.0;
            double a = mean_y - b * mean_x;

            std::vector<double> xs_clean, hs_clean;
            double const max_residual = 0.4 * cell_side_length;
            for (size_t i = 0; i < xs.size(); ++i)
            {
                if (std::abs(hs[i] - (a + b * xs[i])) < max_residual)
                {
                    xs_clean.push_back(xs[i]);
                    hs_clean.push_back(hs[i]);
                }
            }
            if (xs_clean.size() >= 2)
            {
                sum_x = 0; sum_y = 0;
                for (size_t i = 0; i < xs_clean.size(); ++i)
                {
                    sum_x += xs_clean[i];
                    sum_y += hs_clean[i];
                }
                mean_x = sum_x / xs_clean.size();
                mean_y = sum_y / xs_clean.size();
                sxx = 0; sxy = 0;
                for (size_t i = 0; i < xs_clean.size(); ++i)
                {
                    double const dx = xs_clean[i] - mean_x;
                    double const dy = hs_clean[i] - mean_y;
                    sxx += dx * dx;
                    sxy += dx * dy;
                }
                b = (sxx > 1e-6) ? (sxy / sxx) : 0.0;
                a = mean_y - b * mean_x;
            }

            for (int c = 0; c < num_cols; ++c)
            {
                row_h[r][c] = static_cast<float>(a + b * c);
            }
        }
        else if (!xs.empty())
        {
            for (int c = 0; c < num_cols; ++c)
            {
                row_h[r][c] = static_cast<float>(hs[0]);
            }
        }
        else
        {
            for (int c = 0; c < num_cols; ++c)
            {
                row_h[r][c] = row_h[r + 1][c] + static_cast<float>(cell_side_length);
            }
        }
    }

    for (int c = 0; c < num_cols; ++c)
    {
        row_h[0][c] = 2.0f * row_h[1][c] - (num_clue_rows >= 2 ? row_h[2][c] : 0.0f);
    }

    cv::Mat cross_locs_top_mat(num_top_rows, num_cols, CV_32FC2);
    for (int c = 0; c < num_cols; ++c)
    {
        int const k = std::min(5, cross_locs_main_mat.rows - 1);
        cv::Point2f const p0 = cross_locs_main_mat.at<cv::Point2f>(0, c);
        cv::Point2f const pk = cross_locs_main_mat.at<cv::Point2f>(k, c);
        cv::Point2f u = p0 - pk;
        float const len = cv::norm(u);
        if (len > 1e-3f)
        {
            u /= len;
        }
        else
        {
            u = cv::Point2f(0.0f, -1.0f);
        }

        for (int r = 0; r < num_top_rows - 1; ++r)
        {
            cross_locs_top_mat.at<cv::Point2f>(r, c) = p0 + row_h[r][c] * u;
        }
        cross_locs_top_mat.at<cv::Point2f>(num_top_rows - 1, c) = p0;
    }

    return cross_locs_top_mat;
}


cv::Mat CrossLocsDetector::get_cross_locs_left_mat(
    cv::Mat const& image_thresholded,
    cv::Mat const& cross_locs_main_mat,
    int const cell_side_length,
    double const similarity_ratio_min)
{
    std::vector<cv::Point> indices_neighbors_init;
    std::vector<cv::Point2f> cross_locs_neighbors_init;

    for (auto y = 0; y < cross_locs_main_mat.rows; ++y)
    {
        cv::Point const indices(0, y);
        auto const& cross_loc = cross_locs_main_mat.at<cv::Point2f>(indices);

        if (cross_loc != cv::Point2f(-1.0f, -1.0f))
        {
            indices_neighbors_init.push_back(indices);

            auto const cross_loc_neighbor_init = cross_loc;
            cross_locs_neighbors_init.push_back(cross_loc_neighbor_init);
        }
    }

    std::vector<cv::Point> const indices_deltas = {
        cv::Point(0, -1),
        cv::Point(0, 1),
        cv::Point(-1, 0) };

    std::vector<cv::Point2f> const cross_loc_deltas = {
        cv::Point2f(0.0f, -cell_side_length),
        cv::Point2f(0.0f, cell_side_length),
        cv::Point2f(-cell_side_length, 0.0f) };

    auto const cell_side_length_odd = cell_side_length / 2 * 2 + 1;

    cv::Mat mask_cross;
    int mask_cross_perimeter;
    std::tie(mask_cross, mask_cross_perimeter) =
        ng::get_mask_cross(cell_side_length_odd);

    int const max_left_clue_cols = std::min(30, std::max(12, cross_locs_main_mat.cols / 2));

    auto cross_locs_left_map = get_cross_locs_map(
        image_thresholded,
        indices_neighbors_init,
        cross_locs_neighbors_init,
        indices_deltas,
        cross_loc_deltas,
        get_cross_loc_search_roi_left(cell_side_length),
        mask_cross,
        mask_cross_perimeter,
        similarity_ratio_min,
        cell_side_length,
        -max_left_clue_cols, 0,
        0, cross_locs_main_mat.rows - 1);

    if (cross_locs_left_map.empty())
    {
        return cv::Mat();
    }

    auto bbox = get_bounding_rectangle(cross_locs_left_map);
    if (bbox.br().y < cross_locs_main_mat.rows - 1)
    {
        std::vector<cv::Point> missing_indices;
        std::vector<cv::Point2f> missing_locs;
        for (int y = bbox.br().y + 1; y < cross_locs_main_mat.rows; ++y)
        {
            cv::Point const idx(0, y);
            auto const& pt = cross_locs_main_mat.at<cv::Point2f>(idx);
            if (pt != cv::Point2f(-1.0f, -1.0f))
            {
                missing_indices.push_back(idx);
                missing_locs.push_back(pt);
            }
        }

        if (!missing_indices.empty())
        {
            double const similarity_ratio_min_loose = 0.75;
            auto const loose_map = get_cross_locs_map(
                image_thresholded,
                missing_indices,
                missing_locs,
                indices_deltas,
                cross_loc_deltas,
                get_cross_loc_search_roi_left(cell_side_length),
                mask_cross,
                mask_cross_perimeter,
                similarity_ratio_min_loose,
                cell_side_length,
                -max_left_clue_cols, 0,
                bbox.br().y + 1, cross_locs_main_mat.rows - 1);

            int const min_y = bbox.br().y + 1;
            for (auto const& item : loose_map)
            {
                if (item.first.y >= min_y && item.first.y < cross_locs_main_mat.rows)
                {
                    cross_locs_left_map[item.first] = item.second;
                }
            }
        }
        bbox = get_bounding_rectangle(cross_locs_left_map);
    }

    int const min_points_for_col = std::min(2, std::max(1, (cross_locs_main_mat.rows - 1) / 5));
    int effective_min_x = 0;
    for (int x = -1; x >= bbox.tl().x; --x)
    {
        int cnt = 0;
        for (auto const& item : cross_locs_left_map)
        {
            if (item.first.x == x) cnt++;
        }
        if (cnt >= min_points_for_col)
        {
            effective_min_x = x;
        }
    }

    int const num_clue_cols = std::abs(effective_min_x);
    if (num_clue_cols <= 0)
    {
        return cv::Mat();
    }
    int const num_left_cols = num_clue_cols + 2;
    int const num_rows = cross_locs_main_mat.rows;

    std::vector<std::vector<float>> col_w(num_left_cols, std::vector<float>(num_rows, 0.0f));

    std::vector<cv::Point2f> row_u(num_rows);
    for (int r = 0; r < num_rows; ++r)
    {
        cv::Point2f const p0 = cross_locs_main_mat.at<cv::Point2f>(r, 0);
        cv::Point2f sum_v(0.0f, 0.0f);
        int count = 0;
        for (int x = -1; x >= -num_clue_cols; --x)
        {
            auto it = cross_locs_left_map.find(cv::Point(x, r));
            if (it != cross_locs_left_map.end())
            {
                sum_v += (it->second - p0);
                count++;
            }
        }
        int const k = std::min(5, cross_locs_main_mat.cols - 1);
        cv::Point2f const pk = cross_locs_main_mat.at<cv::Point2f>(r, k);
        cv::Point2f u_grid = p0 - pk;
        if (cv::norm(u_grid) > 1e-3f)
        {
            u_grid /= cv::norm(u_grid);
        }
        else
        {
            u_grid = cv::Point2f(-1.0f, 0.0f);
        }

        if (count >= 2 && cv::norm(sum_v) > 1e-3f)
        {
            cv::Point2f const u_cand = sum_v / cv::norm(sum_v);
            if (u_cand.dot(u_grid) > 0.85f)
            {
                row_u[r] = u_cand;
            }
            else
            {
                row_u[r] = u_grid;
            }
        }
        else
        {
            row_u[r] = u_grid;
        }
    }

    for (int x = -1; x >= -num_clue_cols; --x)
    {
        int const c = num_left_cols - 1 + x;
        std::vector<double> ys, ws;
        for (int r = 0; r < num_rows; ++r)
        {
            auto it = cross_locs_left_map.find(cv::Point(x, r));
            if (it != cross_locs_left_map.end())
            {
                cv::Point2f const p0 = cross_locs_main_mat.at<cv::Point2f>(r, 0);
                cv::Point2f const u = row_u[r];
                float const w = (it->second - p0).dot(u);
                ys.push_back(r);
                ws.push_back(w);
            }
        }

        if (ys.size() >= 2)
        {
            double sum_x = 0, sum_y = 0;
            for (size_t i = 0; i < ys.size(); ++i)
            {
                sum_x += ys[i];
                sum_y += ws[i];
            }
            double mean_x = sum_x / ys.size();
            double mean_y = sum_y / ys.size();
            double sxx = 0, sxy = 0;
            for (size_t i = 0; i < ys.size(); ++i)
            {
                double const dx = ys[i] - mean_x;
                double const dy = ws[i] - mean_y;
                sxx += dx * dx;
                sxy += dx * dy;
            }
            double b = (sxx > 1e-6) ? (sxy / sxx) : 0.0;
            double a = mean_y - b * mean_x;

            std::vector<double> ys_clean, ws_clean;
            double const max_residual = 0.4 * cell_side_length;
            for (size_t i = 0; i < ys.size(); ++i)
            {
                if (std::abs(ws[i] - (a + b * ys[i])) < max_residual)
                {
                    ys_clean.push_back(ys[i]);
                    ws_clean.push_back(ws[i]);
                }
            }
            if (ys_clean.size() >= 2)
            {
                sum_x = 0; sum_y = 0;
                for (size_t i = 0; i < ys_clean.size(); ++i)
                {
                    sum_x += ys_clean[i];
                    sum_y += ws_clean[i];
                }
                mean_x = sum_x / ys_clean.size();
                mean_y = sum_y / ys_clean.size();
                sxx = 0; sxy = 0;
                for (size_t i = 0; i < ys_clean.size(); ++i)
                {
                    double const dx = ys_clean[i] - mean_x;
                    double const dy = ws_clean[i] - mean_y;
                    sxx += dx * dx;
                    sxy += dx * dy;
                }
                b = (sxx > 1e-6) ? (sxy / sxx) : 0.0;
                a = mean_y - b * mean_x;
            }

            for (int r = 0; r < num_rows; ++r)
            {
                col_w[c][r] = static_cast<float>(a + b * r);
            }
        }
        else if (!ys.empty())
        {
            for (int r = 0; r < num_rows; ++r)
            {
                col_w[c][r] = static_cast<float>(ws[0]);
            }
        }
        else
        {
            for (int r = 0; r < num_rows; ++r)
            {
                col_w[c][r] = col_w[c + 1][r] + static_cast<float>(cell_side_length);
            }
        }
    }

    for (int r = 0; r < num_rows; ++r)
    {
        col_w[0][r] = 2.0f * col_w[1][r] - (num_clue_cols >= 2 ? col_w[2][r] : 0.0f);
    }

    cv::Mat cross_locs_left_mat(num_rows, num_left_cols, CV_32FC2);
    for (int r = 0; r < num_rows; ++r)
    {
        cv::Point2f const p0 = cross_locs_main_mat.at<cv::Point2f>(r, 0);
        cv::Point2f const u = row_u[r];

        for (int c = 0; c < num_left_cols - 1; ++c)
        {
            cross_locs_left_mat.at<cv::Point2f>(r, c) = p0 + col_w[c][r] * u;
        }
        cross_locs_left_mat.at<cv::Point2f>(r, num_left_cols - 1) = p0;
    }

    return cross_locs_left_mat;
}





cv::Mat CrossLocsDetector::scale_cross_locs_mat(
    cv::Mat const& cross_locs_mat,
    float const scale)
{
    if (cross_locs_mat.empty())
    {
        return cv::Mat();
    }

    cv::Mat out = cross_locs_mat.clone();
    for (int y = 0; y < out.rows; ++y)
    {
        for (int x = 0; x < out.cols; ++x)
        {
            auto& cross_loc = out.at<cv::Point2f>(y, x);
            if (cross_loc != cv::Point2f(-1.0f, -1.0f))
            {
                cross_loc = cv::Point2f(cross_loc.x / scale, cross_loc.y / scale);
            }
        }
    }
    return out;
}


cv::Mat CrossLocsDetector::draw(
    cv::Mat const& image,
    cv::Mat const& cross_locs_mat,
    int const radius,
    cv::Scalar const color)
{
    auto image_copy = image.clone();

    if (cross_locs_mat.empty())
    {
        return image_copy;
    }

    std::for_each(
        cross_locs_mat.begin<cv::Point2f>(),
        cross_locs_mat.end<cv::Point2f>(),
        [&](cv::Point2f const& cross_loc)
        {
            cv::circle(image_copy, cross_loc, radius, color, -1);
        });

    return image_copy;
}


}
