#include <algorithm>
#include <array>
#include <iterator>
#include <numeric>
#include <set>
#include <tuple>
#include <queue>

#include "grid_detector.hpp"
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


std::tuple<cv::Mat, cv::Mat, cv::Mat> CrossLocsDetector::detect(cv::Mat const& image)
{
    cv::Mat image_resized;
    float scale;
    std::tie(image_resized, scale) = resize(image, M_RESIZE_WIDTH_HEIGHT_MAX);

    std::cout << "scale: " << scale << std::endl;

    cv::Mat image_gray;
    cv::cvtColor(image_resized, image_gray, cv::COLOR_BGR2GRAY);

    auto const image_thresholded =
        threshold(image_gray, M_THRESHOLD_BLOCK_SIZE, M_THRESHOLD_C);

    bool cell_loc_found;
    int cell_side_length;
    cv::Point cell_loc;
    {
        cv::Point const image_center(image_thresholded.size() / 2);
        auto const cell_loc_roi = get_roi(image_center, { 150, 150 });

        std::tie(cell_loc_found, cell_side_length, cell_loc) =
            find_cell_side_length_cell_loc(
                image_thresholded,
                cell_loc_roi,
                M_FIND_CELL_SIDE_LENGTH_MIN,
                M_FIND_CELL_SIDE_LENGTH_MAX,
                M_SIMILARITY_RATIO_MIN);
    }

    std::cout << "cell_side_length: " << cell_side_length << std::endl;
    std::cout << "cell_loc: " << cell_loc << std::endl;

    auto cross_locs_main_mat = get_cross_locs_main_mat(
        image_thresholded,
        cell_loc,
        cell_side_length,
        M_SIMILARITY_RATIO_MIN);

    cv::Mat cross_locs_main_rescaled_mat = cross_locs_main_mat / scale;

    auto cross_locs_top_mat = get_cross_locs_top_mat(
        image_thresholded,
        cross_locs_main_mat,
        cell_side_length,
        M_SIMILARITY_RATIO_MIN);

    cv::Mat cross_locs_top_rescaled_mat = cross_locs_top_mat / scale;

    auto cross_locs_left_mat = get_cross_locs_left_mat(
        image_thresholded,
        cross_locs_main_mat,
        cell_side_length,
        M_SIMILARITY_RATIO_MIN);

    cv::Mat cross_locs_left_rescaled_mat = cross_locs_left_mat / scale;

    return std::make_tuple(
        cross_locs_main_rescaled_mat,
        cross_locs_top_rescaled_mat,
        cross_locs_left_rescaled_mat);
}


std::tuple<bool, int, cv::Point> CrossLocsDetector::find_cell_side_length_cell_loc(
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
        cv::Point cell_loc;
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

    return std::make_tuple(false, -1, cv::Point(-1, -1));
}


std::map<cv::Point, cv::Point, PointCompare> CrossLocsDetector::get_cross_locs_map(
    cv::Mat const& image_thresholded,
    std::vector<cv::Point> const& indices_init,
    std::vector<cv::Point> const& cross_locs_init,
    std::vector<cv::Point> const& indices_deltas,
    std::vector<cv::Point> const& cross_loc_deltas,
    cv::Size const roi_size,
    cv::Mat const& mask_cross,
    int const mask_cross_perimeter,
    double const similarity_ratio_min)
{
    std::queue<cv::Point> indices_queue;
    std::set<cv::Point, PointCompare> was_in_indices_queue_set;
    for (auto const& indices_init : indices_init)
    {
        indices_queue.push(indices_init);
        was_in_indices_queue_set.insert(indices_init);
    }

    // Stores predicted initial values
    std::map<cv::Point, cv::Point, PointCompare> cross_locs_init_map;
    for (int i = 0; i < indices_init.size(); ++i)
    {
        cross_locs_init_map[indices_init[i]] = cross_locs_init[i];
    }

    //cv::Mat image_thresholded_copy = image_thresholded.clone();
    //image_thresholded_copy *= 255;

    std::map<cv::Point, cv::Point, PointCompare> cross_locs_map;

    while (!indices_queue.empty())
    {
        auto const indices = indices_queue.front();
        indices_queue.pop();

        auto const& cross_loc_init = cross_locs_init_map[indices];

        bool cross_loc_found;
        cv::Point cross_loc;
        std::tie(cross_loc_found, cross_loc) = find_kernel_loc(
            image_thresholded,
            get_roi(cross_loc_init, roi_size),
            mask_cross,
            mask_cross_perimeter,
            similarity_ratio_min);

        if (cross_loc_found)
        {
            //// Draw
            //{
            //    cv::circle(image_thresholded_copy, cross_loc, 5, 255, -1);
            //    cv::circle(image_thresholded_copy, cross_loc, 3, 0, -1);

            //    cv::imshow("image_thresholded_copy", image_thresholded_copy);
            //    cv::waitKey(1);
            //}

            cross_locs_map[indices] = cross_loc;

            for (int i = 0; i < indices_deltas.size(); ++i)
            {
                auto const indices_neighbor = indices + indices_deltas[i];
                bool const was_in_indices_queue =
                    was_in_indices_queue_set.find(indices_neighbor) != was_in_indices_queue_set.end();

                if (!was_in_indices_queue)
                {
                    indices_queue.push(indices_neighbor);
                    was_in_indices_queue_set.insert(indices_neighbor);

                    auto const cross_loc_neighbor_init = cross_loc + cross_loc_deltas[i];
                    cross_locs_init_map[indices_neighbor] = cross_loc_neighbor_init;
                }
            }
        }
    }

    return cross_locs_map;
}


cv::Rect CrossLocsDetector::get_bounding_rectangle(
    std::map<cv::Point, cv::Point, PointCompare> const& cross_locs_map)
{
    auto const x_min_max_it = std::minmax_element(
        cross_locs_map.cbegin(),
        cross_locs_map.cend(),
        [](std::pair<cv::Point, cv::Point> const& p_1, std::pair<cv::Point, cv::Point> const& p_2)
        {
            auto const& x_1 = p_1.first.x;
            auto const& x_2 = p_2.first.x;

            return x_1 < x_2;
        });

    auto const y_min_max_it = std::minmax_element(
        cross_locs_map.cbegin(),
        cross_locs_map.cend(),
        [](std::pair<cv::Point, cv::Point> const& p_1, std::pair<cv::Point, cv::Point> const& p_2)
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
    std::map<cv::Point, cv::Point, PointCompare> const& cross_locs_map)
{
    auto const bounding_rectangle = get_bounding_rectangle(cross_locs_map);
    auto const cross_loc_mat_size = bounding_rectangle.size() + cv::Size(1, 1);

    cv::Mat cross_locs_mat(cross_loc_mat_size, CV_32SC2, cv::Scalar(-1, -1));

    for (auto x_map = bounding_rectangle.tl().x, x_mat = 0; x_map <= bounding_rectangle.br().x; ++x_map, ++x_mat)
    {
        for (auto y_map = bounding_rectangle.tl().y, y_mat = 0; y_map <= bounding_rectangle.br().y; ++y_map, ++y_mat)
        {
            cv::Point indices_map(x_map, y_map);
            cv::Point indices_mat(x_mat, y_mat);

            auto const indices_cross_loc_it = cross_locs_map.find(indices_map);
            if (indices_cross_loc_it != cross_locs_map.end())
            {
                cross_locs_mat.at<cv::Point>(indices_mat) = indices_cross_loc_it->second;
            }
        }
    }

    return cross_locs_mat;
}


cv::Mat CrossLocsDetector::augment(
    cv::Mat image_resized,
    cv::Mat const& cross_locs_mat,
    int const cell_side_length)
{
    std::set<cv::Point, PointCompare> indices_empty_set;

    for (int y = 0; y < cross_locs_mat.rows; ++y)
    {
        for (int x = 0; x < cross_locs_mat.cols; ++x)
        {
            cv::Point indices(x, y);

            if (cross_locs_mat.at<cv::Point>(indices) == cv::Point(-1, -1))
            {
                indices_empty_set.insert(indices);
            }
        }
    }

    cv::Rect const indices_roi(cv::Point(0, 0), cross_locs_mat.size());
    cv::Mat cross_locs_mat_augmented = cross_locs_mat.clone();

    while (!indices_empty_set.empty())
    {
        std::map<cv::Point, cv::Point, PointCompare> indices_cross_locs_interpolated_map;
        for (auto const& indices : indices_empty_set)
        {
            std::vector<cv::Point> cross_locs_interpolated;

            for (auto const& indices_delta : INDICES_DELTAS)
            {
                auto const indices_neighbor_1 = indices + indices_delta;
                auto const indices_neighbor_2 = indices_neighbor_1 + indices_delta;

                std::vector<cv::Point> const indices_neighbors = {
                    indices_neighbor_1,
                    /*indices_neighbor_2*/ };

                auto const indices_neighbors_are_in_range = std::all_of(
                    indices_neighbors.begin(),
                    indices_neighbors.end(),
                    [&indices_roi](cv::Point const& indices)
                    {
                        return indices_roi.contains(indices);
                    });

                if (indices_neighbors_are_in_range)
                {
                    auto const indices_neighbors_have_value = std::all_of(
                        indices_neighbors.begin(),
                        indices_neighbors.end(),
                        [&cross_locs_mat_augmented](cv::Point const& indices)
                        {
                            return cross_locs_mat_augmented.at<cv::Point>(indices) != cv::Point(-1, -1);
                        });

                    if (indices_neighbors_have_value)
                    {
                        std::vector<cv::Point> neighbors;
                        std::transform(
                            indices_neighbors.begin(),
                            indices_neighbors.end(),
                            std::back_inserter(neighbors),
                            [&cross_locs_mat_augmented](cv::Point const& indices)
                            {
                                return cross_locs_mat_augmented.at<cv::Point>(indices);
                            });

                        auto const direction = indices - indices_neighbors[0];
                        //auto const cross_loc_interpolated = neighbors[0] + (neighbors[0] - neighbors[1]);
                        auto const cross_loc_interpolated = neighbors[0] + cell_side_length * direction;

                        cross_locs_interpolated.push_back(cross_loc_interpolated);
                    }
                }
            }

            if (!cross_locs_interpolated.empty())
            {
                auto const cross_locs_interpolated_sum = std::accumulate(
                    cross_locs_interpolated.begin(),
                    cross_locs_interpolated.end(),
                    cv::Point());
                auto const cross_locs_interpolated_n = static_cast<int>(cross_locs_interpolated.size());
                auto const cross_loc_interpolated =
                    cross_locs_interpolated_sum / cross_locs_interpolated_n;

                indices_cross_locs_interpolated_map[indices] = cross_loc_interpolated;
            }
        }

        for (auto const& indices_cross_loc_interpolated : indices_cross_locs_interpolated_map)
        {
            cv::Point indices;
            cv::Point cross_loc_interpolated;
            std::tie(indices, cross_loc_interpolated) = indices_cross_loc_interpolated;

            cross_locs_mat_augmented.at<cv::Point>(indices) = cross_loc_interpolated;

            indices_empty_set.erase(indices);
        }

        //draw(image_resized, cross_locs_mat_augmented);
    }

    return cross_locs_mat_augmented;
}


cv::Mat CrossLocsDetector::get_cross_locs_main_mat(
    cv::Mat const& image_thresholded,
    cv::Point const& cross_loc_init,
    int const cell_side_length,
    double const similarity_ratio_min)
{
    auto const cell_side_length_odd = cell_side_length / 2 * 2 + 1;
    auto const line_width = static_cast<int>(cell_side_length / 4);
    auto const line_width_half = line_width / 2;

    std::cout << "line_width: " << line_width << std::endl;

    cv::Mat mask_cross;
    int mask_cross_perimeter;
    std::tie(mask_cross, mask_cross_perimeter) =
        get_mask_cross(cell_side_length_odd, line_width_half);

    std::vector<cv::Point> const cross_loc_deltas = {
        cv::Point(0, -cell_side_length),
        cv::Point(cell_side_length, 0),
        cv::Point(0, cell_side_length),
        cv::Point(-cell_side_length, 0) };

    auto const cross_locs_main_map = get_cross_locs_map(
        image_thresholded,
        { cv::Point(0, 0) },
        { cross_loc_init },
        INDICES_DELTAS,
        cross_loc_deltas,
        cv::Size(2 * cell_side_length, 2 * cell_side_length),
        mask_cross,
        mask_cross_perimeter,
        similarity_ratio_min);

    auto cross_locs_main_mat = convert_to_mat(cross_locs_main_map);

    // Add extra lines on perimeter
    auto const cross_locs_main_resized_mat_size = cross_locs_main_mat.size() + cv::Size(2, 2);

    cv::Mat cross_locs_main_resized_mat(
        cross_locs_main_resized_mat_size,
        cross_locs_main_mat.type(),
        cv::Scalar(-1, -1));

    cv::Rect const roi(cv::Point(1, 1), cross_locs_main_mat.size());
    cross_locs_main_mat.copyTo(cross_locs_main_resized_mat(roi));

    auto const cross_locs_main_resized_augmented_mat =
        augment(cv::Mat(), cross_locs_main_resized_mat, cell_side_length);

    return cross_locs_main_resized_augmented_mat;
}


cv::Mat CrossLocsDetector::get_cross_locs_top_mat(
    cv::Mat const& image_thresholded,
    cv::Mat const& cross_locs_main_mat,
    int const cell_side_length,
    double const similarity_ratio_min)
{
    std::vector<cv::Point> cross_locs_neighbors_init;
    std::vector<cv::Point> indices_neighbors_init;

    cv::Point const cross_loc_delta_top(0, -cell_side_length);

    for (auto x = 1; x < cross_locs_main_mat.cols; ++x)
    {
        cv::Point const indices(x, 1);
        auto const& cross_loc = cross_locs_main_mat.at<cv::Point>(indices);

        if (cross_loc != cv::Point(-1, -1))
        {
            auto const cross_loc_neighbor_init = cross_loc + cross_loc_delta_top;

            cross_locs_neighbors_init.push_back(cross_loc_neighbor_init);
            indices_neighbors_init.push_back(indices);
        }
    }

    std::vector<cv::Point> const indices_deltas = {
        cv::Point(0, -1),
        cv::Point(1, 0),
        cv::Point(-1, 0) };

    std::vector<cv::Point> const cross_loc_deltas = {
        cv::Point(0, -cell_side_length),
        cv::Point(cell_side_length, 0),
        cv::Point(-cell_side_length, 0) };

    auto const cell_side_length_odd = cell_side_length / 2 * 2 + 1;

    cv::Mat mask_cross;
    int mask_cross_perimeter;
    std::tie(mask_cross, mask_cross_perimeter) =
        get_mask_cross(cell_side_length_odd);

    auto const cross_locs_top_map = get_cross_locs_map(
        image_thresholded,
        indices_neighbors_init,
        cross_locs_neighbors_init,
        indices_deltas,
        cross_loc_deltas,
        cv::Size(2 * cell_side_length, 2 * cell_side_length),
        mask_cross,
        mask_cross_perimeter,
        similarity_ratio_min);

    auto const cross_locs_top_mat = convert_to_mat(cross_locs_top_map);

    // Add extra line to the top and extra column to the right
    auto const cross_locs_top_resized_mat_size = cross_locs_top_mat.size() + cv::Size(1, 1);

    cv::Mat cross_locs_top_resized_mat(
        cross_locs_top_resized_mat_size,
        cross_locs_top_mat.type(),
        cv::Scalar(-1, -1));

    cv::Rect const roi(cv::Point(0, 1), cross_locs_top_mat.size());
    cross_locs_top_mat.copyTo(cross_locs_top_resized_mat(roi));

    auto cross_locs_top_resized_augmented_mat =
        augment(cv::Mat(), cross_locs_top_resized_mat, cell_side_length);

    return cross_locs_top_resized_augmented_mat;
}


cv::Mat CrossLocsDetector::get_cross_locs_left_mat(
    cv::Mat const& image_thresholded,
    cv::Mat const& cross_locs_main_mat,
    int const cell_side_length,
    double const similarity_ratio_min)
{
    std::vector<cv::Point> cross_locs_neighbors_init;
    std::vector<cv::Point> indices_neighbors_init;

    cv::Point cross_loc_delta_left(-cell_side_length, 0);

    for (auto y = 1; y < cross_locs_main_mat.rows; ++y)
    {
        cv::Point const indices(1, y);
        auto const& cross_loc = cross_locs_main_mat.at<cv::Point>(indices);

        if (cross_loc != cv::Point(-1, -1))
        {
            auto const cross_loc_neighbor_init = cross_loc + cross_loc_delta_left;

            cross_locs_neighbors_init.push_back(cross_loc_neighbor_init);
            indices_neighbors_init.push_back(indices);
        }
    }

    std::vector<cv::Point> const indices_deltas = {
        cv::Point(0, -1),
        cv::Point(0, 1),
        cv::Point(-1, 0) };

    std::vector<cv::Point> const cross_loc_deltas = {
        cv::Point(0, -cell_side_length),
        cv::Point(0, cell_side_length),
        cv::Point(-cell_side_length, 0) };

    auto const cell_side_length_odd = cell_side_length / 2 * 2 + 1;

    cv::Mat mask_cross;
    int mask_cross_perimeter;
    std::tie(mask_cross, mask_cross_perimeter) =
        ng::get_mask_cross(cell_side_length_odd);

    auto const cross_locs_left_map = get_cross_locs_map(
        image_thresholded,
        indices_neighbors_init,
        cross_locs_neighbors_init,
        indices_deltas,
        cross_loc_deltas,
        cv::Size(2 * cell_side_length, 2 * cell_side_length),
        mask_cross,
        mask_cross_perimeter,
        similarity_ratio_min);

    auto cross_locs_left_mat = convert_to_mat(cross_locs_left_map);

    // Add extra line to the bottom and extra column to the left
    auto const cross_locs_left_resized_mat_size = cross_locs_left_mat.size() + cv::Size(1, 1);

    cv::Mat cross_locs_left_resized_mat(
        cross_locs_left_resized_mat_size,
        cross_locs_left_mat.type(),
        cv::Scalar(-1, -1));

    cv::Rect const roi(cv::Point(1, 0), cross_locs_left_mat.size());
    cross_locs_left_mat.copyTo(cross_locs_left_resized_mat(roi));

    auto cross_locs_left_resized_augmented_mat =
        augment(cv::Mat(), cross_locs_left_resized_mat, cell_side_length);

    return cross_locs_left_resized_augmented_mat;
}


void CrossLocsDetector::print(cv::Mat const& cross_locs_mat)
{
    for (int y = 0; y < cross_locs_mat.rows; ++y)
    {
        for (int x = 0; x < cross_locs_mat.cols; ++x)
        {
            if (cross_locs_mat.at<cv::Point>(y, x) == cv::Point(-1, -1))
            {
                std::cout << 0;
            }
            else
            {
                std::cout << 1;
            }
        }
        std::cout << std::endl;
    }
}


cv::Mat CrossLocsDetector::draw(
    cv::Mat const& image,
    cv::Mat const& cross_locs_mat,
    cv::Scalar const color)
{
    auto image_copy = image.clone();

    std::for_each(
        cross_locs_mat.begin<cv::Point>(),
        cross_locs_mat.end<cv::Point>(),
        [&image_copy, &color](cv::Point const& cross_loc)
        {
            cv::circle(image_copy, cross_loc, 3, color, -1);
        });

    return image_copy;
}


}
