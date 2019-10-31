#include <algorithm>
#include <iterator>
#include <numeric>
#include <queue>
#include <set>

#include "image_operations.hpp"
#include "masks.hpp"

namespace ng
{


std::pair<cv::Mat, float> resize(
    cv::Mat const& image,
    int const width_height_max_destination,
    cv::InterpolationFlags const interpolation_flag)
{
    auto const width_height_max = static_cast<float>(std::max(image.rows, image.cols));
    auto const scale = width_height_max_destination / width_height_max;

    cv::Mat image_resized;
    cv::resize(image, image_resized, cv::Size(), scale, scale, interpolation_flag);

    return std::make_pair(image_resized, scale);
}


cv::Mat threshold(
    cv::Mat const& image_gray,
    int const block_size,
    double const c)
{
    auto const MAX_VALUE = 1;

    cv::Mat image_thresholded;
    cv::adaptiveThreshold(
        image_gray,
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


std::pair<bool, cv::Point> find_kernel_loc(
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
        std::make_pair(false, cv::Point(-1, -1));
}


bool is_inside(cv::Rect const& rect, cv::Rect const& sub_rect)
{
    return (rect & sub_rect) == sub_rect;
}


std::pair<bool, cv::Point> find_kernel_loc(
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
        return std::make_pair(false, cv::Point(-1, -1));
    }

    bool kernel_loc_found;
    cv::Point kernel_loc;
    std::tie(kernel_loc_found, kernel_loc) = find_kernel_loc(
        image_thresholded(roi),
        kernel,
        max,
        similarity_ratio_min,
        anchor);

    return kernel_loc_found ?
        std::make_pair(true, kernel_loc + roi.tl()) :
        std::make_pair(false, cv::Point(-1, -1));
}


}
