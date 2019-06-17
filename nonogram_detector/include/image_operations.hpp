#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "point_compare.hpp"

#include <opencv2/opencv.hpp>

namespace ng
{

// Resize an image to a fixed size, so threshold parameters are fixed for every image
cv::Mat resize(
    cv::Mat const& image,
    int const max_width_height,
    cv::InterpolationFlags const interpolation_flag = cv::InterpolationFlags::INTER_LINEAR);

// Thresholds an image (uses cv::adaptiveThreshold), sets pixels to 0 or 1
cv::Mat threshold(
    cv::Mat const& image_grayscale,
    int const block_size,
    double const c);

// If roi size is odd, center will be in the bottom right of 4 central pixels
cv::Rect get_roi(cv::Point const& center, cv::Size const& roi_size);

// The boolean flag in the return value shows if the search was successful
std::pair<bool, cv::Point> find_kernel_loc(
    cv::Mat const& image_thresholded,
    cv::Mat const& kernel,
    double const max,
    double const similarity_ratio_min,
    cv::Point const& anchor);

bool is_inside(cv::Rect const& rect, cv::Rect const& sub_rect);

std::pair<bool, cv::Point> find_kernel_loc(
    cv::Mat const& image,
    cv::Rect const& roi,
    cv::Mat const& kernel,
    double const max,
    double const similarity_ratio_min,
    cv::Point const& anchor = cv::Point(-1, -1));

std::tuple<bool, int, cv::Point> find_cell_side_length_cell_loc(
    cv::Mat const& image,
    cv::Rect const& image_roi,
    int const cell_side_length_min,
    int const cell_side_length_max,
    double const similarity_ratio_min);

// <cross_locs_init> must correspond with <indices_init>
std::map<cv::Point, cv::Point, PointCompare> get_cross_locs_map(
    cv::Mat const& image_thresholded,
    std::vector<cv::Point> const& cross_locs_init,
    std::vector<cv::Point> const& indices_init,
    std::vector<cv::Point> const& cross_loc_deltas,
    std::vector<cv::Point> const& indices_deltas,
    int const cell_side_length,
    cv::Mat const& mask_cross,
    int const mask_cross_perimeter,
    double const similarity_ratio_min);

cv::Rect get_bounding_rectangle(std::map<cv::Point, cv::Point, PointCompare> const& cross_locs_map);

cv::Mat convert_to_mat(std::map<cv::Point, cv::Point, PointCompare> const& cross_locs_map);

cv::Mat get_cross_locs_main_mat(
    cv::Mat const& image_thresholded,
    cv::Point const& cross_loc_init,
    int const cell_side_length,
    double const similarity_ratio_min);

cv::Mat get_cross_locs_top_mat(
    cv::Mat const& image_thresholded,
    cv::Mat const& cross_locs_main_mat,
    int const cell_side_length,
    double const similarity_ratio_min);

cv::Mat get_cross_locs_left_mat(
    cv::Mat const& image_thresholded,
    cv::Mat const& cross_locs_main_mat,
    int const cell_side_length,
    double const similarity_ratio_min);

void print(cv::Mat const& cross_locs_mat);

}
