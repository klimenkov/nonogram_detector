#pragma once

#include <array>
#include <map>
#include <utility>
#include <vector>

#include "detection.hpp"
#include "point_compare.hpp"

#include <opencv2/opencv.hpp>

namespace ng
{

class CrossLocsDetector
{
public:
    CrossLocsDetector(
        float const resize_width_height_max,
        int const threshold_block_size,
        double const threshold_c,
        int const find_cell_side_length_min,
        int const find_cell_side_length_max,
        double const similarity_ratio_min);

    Detection detect(cv::Mat const& image);

    static int estimate_cell_side_length(
        cv::Mat const& image_thresholded,
        cv::Rect const& roi,
        int min, int max);

    static cv::Mat draw(
        cv::Mat const& image,
        cv::Mat const& cross_locs_mat,
        int const radius,
        cv::Scalar const color);

private:
    float const M_RESIZE_WIDTH_HEIGHT_MAX;
    int const M_THRESHOLD_BLOCK_SIZE;
    double const M_THRESHOLD_C;
    int const M_FIND_CELL_SIDE_LENGTH_MIN;
    int const M_FIND_CELL_SIDE_LENGTH_MAX;
    double const M_SIMILARITY_RATIO_MIN;

    static cv::Point const INDICES_DELTA_UP;
    static cv::Point const INDICES_DELTA_RIGHT;
    static cv::Point const INDICES_DELTA_DOWN;
    static cv::Point const INDICES_DELTA_LEFT;

    static std::vector<cv::Point> const INDICES_DELTAS;


    static std::tuple<bool, int, cv::Point2f> find_cell_side_length_cell_loc(
        cv::Mat const& image_thresholded,
        cv::Rect const& image_thresholded_roi,
        int const cell_side_length_min,
        int const cell_side_length_max,
        double const similarity_ratio_min);


    // <indices_init> must correspond with <cross_locs_init>
    static std::map<cv::Point, cv::Point2f, PointCompare> get_cross_locs_map(
        cv::Mat const& image_thresholded,
        std::vector<cv::Point> const& indices_init,
        std::vector<cv::Point2f> const& cross_locs_init,
        std::vector<cv::Point> const& indices_deltas,
        std::vector<cv::Point2f> const& cross_loc_deltas,
        cv::Size const roi_size,
        cv::Mat const& mask_cross,
        int const mask_cross_perimeter,
        double const similarity_ratio_min);


    static cv::Size get_cross_loc_search_roi(int cell_side_length);


    static cv::Rect get_bounding_rectangle(
        std::map<cv::Point, cv::Point2f, PointCompare> const& cross_locs_map);


    static cv::Mat convert_to_mat(
        std::map<cv::Point, cv::Point2f, PointCompare> const& cross_locs_map);


    // Shared tail of the cross-loc matrix builders: converts the BFS map to a
    // dense matrix, embeds it into a larger sentinel-filled canvas at <offset>
    // (<pad> extra entries overall), then augments missing entries. Returns an
    // empty Mat when the map converted to nothing.
    static cv::Mat convert_pad_augment(
        std::map<cv::Point, cv::Point2f, PointCompare> const& cross_locs_map,
        cv::Point const& offset,
        cv::Size const& pad,
        int const cell_side_length);


    static cv::Mat augment(
        cv::Mat const& cross_locs_mat,
        int const cell_side_length);


    static cv::Mat get_cross_locs_main_mat(
        cv::Mat const& image_thresholded,
        cv::Point2f const& cross_loc_init,
        int const cell_side_length,
        double const similarity_ratio_min);


    static cv::Mat get_cross_locs_top_mat(
        cv::Mat const& image_thresholded,
        cv::Mat const& cross_locs_main_mat,
        int const cell_side_length,
        double const similarity_ratio_min);


    static cv::Mat get_cross_locs_left_mat(
        cv::Mat const& image_thresholded,
        cv::Mat const& cross_locs_main_mat,
        int const cell_side_length,
        double const similarity_ratio_min);


    static cv::Mat scale_cross_locs_mat(
        cv::Mat const& cross_locs_mat,
        float const scale);


};

}
