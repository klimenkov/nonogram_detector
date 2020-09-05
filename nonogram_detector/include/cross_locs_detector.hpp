#pragma once

#include <array>
#include <map>
#include <utility>
#include <vector>

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

    // First value means if something was detected
    std::tuple<bool, cv::Mat, cv::Mat, cv::Mat> detect(cv::Mat const& image);

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


    static std::tuple<bool, int, cv::Point> find_cell_side_length_cell_loc(
        cv::Mat const& image_thresholded,
        cv::Rect const& image_thresholded_roi,
        int const cell_side_length_min,
        int const cell_side_length_max,
        double const similarity_ratio_min);


    // <indices_init> must correspond with <cross_locs_init>
    static std::map<cv::Point, cv::Point, PointCompare> get_cross_locs_map(
        cv::Mat const& image_thresholded,
        std::vector<cv::Point> const& indices_init,
        std::vector<cv::Point> const& cross_locs_init,
        std::vector<cv::Point> const& indices_deltas,
        std::vector<cv::Point> const& cross_loc_deltas,
        cv::Size const roi_size,
        cv::Mat const& mask_cross,
        int const mask_cross_perimeter,
        double const similarity_ratio_min);


    static cv::Rect get_bounding_rectangle(
        std::map<cv::Point, cv::Point, PointCompare> const& cross_locs_map);


    static cv::Mat convert_to_mat(
        std::map<cv::Point, cv::Point, PointCompare> const& cross_locs_map);


    static cv::Mat augment(
        cv::Mat image_resized,
        cv::Mat const& cross_locs_mat,
        int const cell_side_length);


    static cv::Mat get_cross_locs_main_mat(
        cv::Mat const& image_thresholded,
        cv::Point const& cross_loc_init,
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


    static void print(cv::Mat const& cross_locs_mat);


};

}
