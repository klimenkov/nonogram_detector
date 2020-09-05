#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "point_compare.hpp"

#include <opencv2/opencv.hpp>

namespace ng
{


// Resizes an image so that the longest side (width or height) becomes <width_height_max_destination>
std::pair<cv::Mat, float> resize(
    cv::Mat const& image,
    int const width_height_max_destination,
    cv::InterpolationFlags const interpolation_flag = cv::InterpolationFlags::INTER_LINEAR);


// Thresholds a grayscale image (cv::adaptiveThreshold), returns CV_8U image of 0 and 1
cv::Mat threshold(
    cv::Mat const& image_gray,
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


std::vector<std::vector<cv::Mat>> get_cell_warped_images_vector(cv::Mat const& image, cv::Mat const& cross_locs);


}
