#include <map>
#include <tuple>
#include <utility>

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
std::pair<bool, cv::Point2f> find_kernel_loc(
    cv::Mat const& image_thresholded,
    cv::Mat const& kernel,
    double const max,
    double const similarity_ratio_min,
    cv::Point const& anchor);

bool is_inside(cv::Rect const& rect, cv::Rect const& sub_rect);

std::pair<bool, cv::Point2f> find_kernel_loc(
    cv::Mat const& image,
    cv::Rect const& roi,
    cv::Mat const& kernel,
    double const max,
    double const similarity_ratio_min,
    cv::Point const& anchor = cv::Point(-1, -1));

std::tuple<bool, int, cv::Point2f> find_cell_side_length_cell_loc(
    cv::Mat const& image,
    cv::Rect const& image_roi,
    int const cell_side_length_min,
    int const cell_side_length_max,
    double const similarity_ratio_min);

std::map<cv::Point, cv::Point2f, PointCompare> get_cross_locs_map(
    cv::Mat const& image_thresholded,
    cv::Point2f const& cross_loc_init,
    int const cell_side_length,
    cv::Mat const& mask_cross,
    int const mask_cross_perimeter,
    double const similarity_ratio_min);

}
