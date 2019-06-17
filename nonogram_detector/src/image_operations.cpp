#include <algorithm>
#include <queue>
#include <set>

#include "image_operations.hpp"
#include "masks.hpp"

namespace ng
{

cv::Mat resize(
    cv::Mat const& image,
    int const width_or_height_max,
    cv::InterpolationFlags const interpolation_flag)
{
    auto const image_width_or_height_max = static_cast<float>(std::max(image.rows, image.cols));

    auto const scale = static_cast<float>(width_or_height_max) / image_width_or_height_max;

    cv::Mat image_resized;
    cv::resize(image, image_resized, cv::Size(), scale, scale, interpolation_flag);

    return image_resized;
}

cv::Mat threshold(
    cv::Mat const& image_grayscale,
    int const block_size,
    double const c)
{
    auto const MAX_VALUE = 1;

    cv::Mat image_thresholded;
    cv::adaptiveThreshold(
        image_grayscale,
        image_thresholded,
        MAX_VALUE,
        cv::ADAPTIVE_THRESH_MEAN_C,
        cv::THRESH_BINARY_INV,
        block_size,
        c);

    return image_thresholded;
}

cv::Rect get_roi(cv::Point const& center, cv::Size const& roi_size)
{
    return cv::Rect(center - cv::Point(roi_size / 2), roi_size);
}

std::pair<bool, cv::Point2f> find_kernel_loc(
    cv::Mat const& image_thresholded,
    cv::Mat const& kernel,
    double const max,
    double const similarity_ratio_min,
    cv::Point const& anchor)
{
    // Convolve image with a <kernel> to get locations of the <kernel>
    cv::Mat image_filtered;
    cv::filter2D(
        image_thresholded,
        image_filtered,
        CV_32F,
        kernel,
        anchor,
        0.0,
        cv::BORDER_ISOLATED);

    // Normalize the image with a known max value
    image_filtered /= max;

    double peak_max;
    cv::Point peak_max_loc;
    cv::minMaxLoc(image_filtered, nullptr, &peak_max, nullptr, &peak_max_loc);

    return peak_max > similarity_ratio_min ?
        std::make_pair(true, peak_max_loc) :
        std::make_pair(false, cv::Point2f(-1, -1));
}

bool is_inside(cv::Rect const& rect, cv::Rect const& sub_rect)
{
    return (rect & sub_rect) == sub_rect;
}

std::pair<bool, cv::Point2f> find_kernel_loc(
    cv::Mat const& image_thresholded,
    cv::Rect const& roi,
    cv::Mat const& kernel,
    double const max,
    double const similarity_ratio_min,
    cv::Point const& anchor)
{
    cv::Rect const image_thresholded_roi(cv::Point(0, 0), image_thresholded.size());
    if (!is_inside(image_thresholded_roi, roi))
    {
        return std::make_pair(false, cv::Point2f(-1, -1));
    }

    bool kernel_loc_found;
    cv::Point2f kernel_loc;
    std::tie(kernel_loc_found, kernel_loc) = find_kernel_loc(
        image_thresholded(roi),
        kernel,
        max,
        similarity_ratio_min,
        anchor);

    return kernel_loc_found ?
        std::make_pair(true, kernel_loc + cv::Point2f(roi.tl())) :
        std::make_pair(false, cv::Point2f(-1, -1));
}

std::tuple<bool, int, cv::Point2f> find_cell_side_length_cell_loc(
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
        std::tie(mask_square, mask_square_perimeter) = ng::get_mask_square(cell_side_length);

        bool cell_loc_found;
        cv::Point2f cell_loc;
        std::tie(cell_loc_found, cell_loc) = ng::find_kernel_loc(
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

    return std::make_tuple(false, -1, cv::Point2f(-1, -1));
}

std::map<cv::Point, cv::Point2f, PointCompare> get_cross_locs_map(
    cv::Mat const& image_thresholded,
    std::vector<cv::Point2f> const& cross_locs_init,
    std::vector<cv::Point> const& indices_init,
    std::vector<cv::Point> const& indices_deltas,
    std::vector<cv::Point2f> const& cross_loc_deltas,
    int const cell_side_length,
    cv::Mat const& mask_cross,
    int const mask_cross_perimeter,
    double const similarity_ratio_min)
{

    std::queue<cv::Point> indices_queue;
    for (auto const& indices_init : indices_init)
    {
        indices_queue.push(indices_init);
    }

    std::map<cv::Point, cv::Point2f, PointCompare> cross_locs_init_map;
    for (int i = 0; i < indices_init.size(); ++i)
    {
        cross_locs_init_map[indices_init[i]] = cross_locs_init[i];
    }

    std::set<cv::Point, PointCompare> visited;
    for (auto const& indices_init : indices_init)
    {
        visited.insert(indices_init);
    }

    cv::Mat image_thresholded_copy = image_thresholded.clone();
    image_thresholded_copy *= 255;

    std::map<cv::Point, cv::Point2f, PointCompare> cross_locs_map;

    while (!indices_queue.empty())
    {
        auto const indices = indices_queue.front();
        indices_queue.pop();

        auto const& cross_loc_init = cross_locs_init_map[indices];

        bool cross_loc_found;
        cv::Point2f cross_loc;
        std::tie(cross_loc_found, cross_loc) = ng::find_kernel_loc(
            image_thresholded,
            ng::get_roi(cross_loc_init, cv::Size(2 * cell_side_length, 2 * cell_side_length)),
            mask_cross,
            mask_cross_perimeter,
            similarity_ratio_min);

        cv::circle(image_thresholded_copy, cross_loc, 5, 255, -1);
        cv::circle(image_thresholded_copy, cross_loc, 3, 0, -1);

        cv::imshow("image_thresholded_copy", image_thresholded_copy);
        cv::waitKey(1);

        if (cross_loc_found)
        {
            cross_locs_map[indices] = cross_loc;

            for (int i = 0; i < indices_deltas.size(); ++i)
            {
                auto const indices_neighbor = indices + indices_deltas[i];

                if (visited.find(indices_neighbor) == visited.end())
                {
                    indices_queue.push(indices_neighbor);
                    visited.insert(indices_neighbor);

                    auto const cross_loc_neighbor_init = cross_loc + cross_loc_deltas[i];
                    cross_locs_init_map[indices_neighbor] = cross_loc_neighbor_init;
                }
            }
        }
    }

    return cross_locs_map;
}

cv::Rect get_bounding_rectangle(std::map<cv::Point, cv::Point2f, PointCompare> const& cross_locs_map)
{
    auto const x_min_max_it = std::minmax_element(
        cross_locs_map.cbegin(),
        cross_locs_map.cend(),
        [](std::pair<cv::Point, cv::Point2f> const& p_1, std::pair<cv::Point, cv::Point2f> const& p_2)
        {
            auto const& x_1 = p_1.first.x;
            auto const& x_2 = p_2.first.x;

            return x_1 < x_2;
        });

    auto const y_min_max_it = std::minmax_element(
        cross_locs_map.cbegin(),
        cross_locs_map.cend(),
        [](std::pair<cv::Point, cv::Point2f> const& p_1, std::pair<cv::Point, cv::Point2f> const& p_2)
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

cv::Mat convert_to_mat(std::map<cv::Point, cv::Point2f, PointCompare> const& cross_locs_map)
{
    auto const bounding_rectangle = get_bounding_rectangle(cross_locs_map);
    auto const cross_loc_mat_size = bounding_rectangle.size() + cv::Size(3, 3);

    cv::Mat cross_locs_mat(cross_loc_mat_size, CV_32FC2, cv::Scalar(-1, -1));

    for (auto x_map = bounding_rectangle.tl().x, x_mat = 1; x_map <= bounding_rectangle.br().x; ++x_map, ++x_mat)
    {
        for (auto y_map = bounding_rectangle.tl().y, y_mat = 1; y_map <= bounding_rectangle.br().y; ++y_map, ++y_mat)
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

cv::Mat get_cross_locs_main_mat(
    cv::Mat const& image_thresholded,
    cv::Point2f const& cross_loc_init,
    int const cell_side_length,
    double const similarity_ratio_min)
{
    auto const cell_side_length_odd = cell_side_length / 2 * 2 + 1;
    auto const line_width = static_cast<int>(cell_side_length / 6.5);
    auto const line_width_half = line_width / 2;

    cv::Mat mask_cross;
    int mask_cross_perimeter;
    std::tie(mask_cross, mask_cross_perimeter) =
        ng::get_mask_cross(cell_side_length_odd, line_width_half);

    std::vector<cv::Point> const indices_deltas = {
        cv::Point(0, -1),
        cv::Point(1, 0),
        cv::Point(0, 1),
        cv::Point(-1, 0) };

    std::vector<cv::Point2f> const cross_loc_deltas = {
        cv::Point2f(0, -cell_side_length),
        cv::Point2f(cell_side_length, 0),
        cv::Point2f(0, cell_side_length),
        cv::Point2f(-cell_side_length, 0) };

    auto const cross_locs_main_map = ng::get_cross_locs_map(
        image_thresholded,
        { cross_loc_init },
        { cv::Point(0, 0) },
        indices_deltas,
        cross_loc_deltas,
        cell_side_length,
        mask_cross,
        mask_cross_perimeter,
        similarity_ratio_min);

    auto cross_locs_main_mat = ng::convert_to_mat(cross_locs_main_map);

    return cross_locs_main_mat;
}

cv::Mat get_cross_locs_top_mat(
    cv::Mat const& image_thresholded,
    cv::Mat const& cross_locs_main_mat,
    int const cell_side_length,
    double const similarity_ratio_min)
{
    std::vector<cv::Point2f> cross_locs_init;
    std::vector<cv::Point> indices_init;

    for (auto x = 1; x < cross_locs_main_mat.cols - 1; ++x)
    {
        if (cross_locs_main_mat.at<cv::Point2f>(1, x) != cv::Point2f(-1, -1))
        {
            auto const cross_loc_init =
                cross_locs_main_mat.at<cv::Point2f>(1, x) + cv::Point2f(0, -cell_side_length);

            cross_locs_init.push_back(cross_loc_init);
            indices_init.emplace_back(x, 0);
        }
    }

    std::vector<cv::Point> const indices_deltas = {
        cv::Point(0, -1),
        cv::Point(1, 0),
        cv::Point(-1, 0) };

    std::vector<cv::Point2f> const cross_loc_deltas = {
        cv::Point2f(0, -cell_side_length),
        cv::Point2f(cell_side_length, 0),
        cv::Point2f(-cell_side_length, 0) };

    auto const cell_side_length_odd = cell_side_length / 2 * 2 + 1;

    cv::Mat mask_cross;
    int mask_cross_perimeter;
    std::tie(mask_cross, mask_cross_perimeter) =
        ng::get_mask_cross(cell_side_length_odd);

    auto const cross_locs_top_map = ng::get_cross_locs_map(
        image_thresholded,
        cross_locs_init,
        indices_init,
        indices_deltas,
        cross_loc_deltas,
        cell_side_length,
        mask_cross,
        mask_cross_perimeter,
        similarity_ratio_min);

    auto cross_locs_top_mat = ng::convert_to_mat(cross_locs_top_map);

    return cross_locs_top_mat;
}

cv::Mat get_cross_locs_left_mat(
    cv::Mat const& image_thresholded,
    cv::Mat const& cross_locs_main_mat,
    int const cell_side_length,
    double const similarity_ratio_min)
{
    std::vector<cv::Point2f> cross_locs_init;
    std::vector<cv::Point> indices_init;
    for (auto y = 1; y < cross_locs_main_mat.rows - 1; ++y)
    {
        if (cross_locs_main_mat.at<cv::Point2f>(y, 1) != cv::Point2f(-1, -1))
        {
            auto const cross_loc_init =
                cross_locs_main_mat.at<cv::Point2f>(y, 1) + cv::Point2f(-cell_side_length, 0);

            cross_locs_init.push_back(cross_loc_init);
            indices_init.emplace_back(0, y);
        }
    }

    std::vector<cv::Point> const indices_deltas = {
        cv::Point(0, -1),
        cv::Point(0, 1),
        cv::Point(-1, 0) };

    std::vector<cv::Point2f> const cross_loc_deltas = {
        cv::Point2f(0, -cell_side_length),
        cv::Point2f(0, cell_side_length),
        cv::Point2f(-cell_side_length, 0) };

    auto const cell_side_length_odd = cell_side_length / 2 * 2 + 1;

    cv::Mat mask_cross;
    int mask_cross_perimeter;
    std::tie(mask_cross, mask_cross_perimeter) =
        ng::get_mask_cross(cell_side_length_odd);

    auto const cross_locs_top_map = ng::get_cross_locs_map(
        image_thresholded,
        cross_locs_init,
        indices_init,
        indices_deltas,
        cross_loc_deltas,
        cell_side_length,
        mask_cross,
        mask_cross_perimeter,
        similarity_ratio_min);

    auto cross_locs_left_mat = ng::convert_to_mat(cross_locs_top_map);

    return cross_locs_left_mat;
}

void print(cv::Mat const& cross_locs_mat)
{
    for (int y = 0; y < cross_locs_mat.rows; ++y)
    {
        for (int x = 0; x < cross_locs_mat.cols; ++x)
        {
            std::cout << (cross_locs_mat.at<cv::Point2f>(y, x) != cv::Point2f(-1.0f, -1.0f));
        }
        std::cout << std::endl;
    }
}

}