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


// Fits a 1-D parabola along x and through the y-neighbors of the integer peak
// in the (single-channel, floating-point) filtered response, returning a
// subpixel peak location. Falls back to the integer peak when the peak sits at
// the response border or the response is flat.
cv::Point2f refine_peak_loc(cv::Mat const& image_filtered, cv::Point const& peak);


// Refines cross locations (CV_32FC2 mat of cv::Point2f, (-1,-1) sentinels) to
// the geometric center of the drawn line intersection: for each location, the
// darkness-weighted centroid of the grayscale ink in a <window_radius> window
// around it, computed per axis with the crossing line's uniform contribution
// subtracted (baseline = min of the 1-D profile) and only ink within a 3 px
// band of the current location counted, so contaminating ink further out
// (e.g. a clue digit) cannot pull the centroid. Anti-aliased line edges carry
// the subpixel information. An entry keeps its per-axis location when the
// window clips the image border or the band holds too little ink (empty
// paper, extrapolated padding).
void refine_cross_locs_ink(
    cv::Mat const& image_gray,
    cv::Mat& cross_locs,
    int const window_radius);


// The boolean flag in the return value shows if the search was successful.
// The returned location is subpixel (cv::Point2f).
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


std::vector<std::vector<cv::Mat>> get_cell_warped_images_vector(cv::Mat const& image, cv::Mat const& cross_locs);


}
