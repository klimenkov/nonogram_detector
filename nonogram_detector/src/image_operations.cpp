#include <algorithm>
#include <cmath>
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


cv::Point2f refine_peak_loc(cv::Mat const& image_filtered, cv::Point const& peak)
{
    auto const x = peak.x;
    auto const y = peak.y;

    // Need the four orthogonal neighbors present.
    if (x - 1 < 0 || x + 1 >= image_filtered.cols ||
        y - 1 < 0 || y + 1 >= image_filtered.rows)
    {
        return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
    }

    auto const f = [&image_filtered](int px, int py) {
        return image_filtered.at<float>(py, px);
    };

    // 1-D parabola fit along x: offset where the quadratic peaks.
    float x_offset = 0.0f;
    {
        auto const f_m1 = f(x - 1, y);
        auto const f_0  = f(x,     y);
        auto const f_p1 = f(x + 1, y);
        auto const denom = f_m1 - 2.0f * f_0 + f_p1;
        if (std::fabs(denom) > 1e-6f)
            x_offset = (f_m1 - f_p1) / (2.0f * denom);
    }

    float y_offset = 0.0f;
    {
        auto const f_m1 = f(x, y - 1);
        auto const f_0  = f(x, y);
        auto const f_p1 = f(x, y + 1);
        auto const denom = f_m1 - 2.0f * f_0 + f_p1;
        if (std::fabs(denom) > 1e-6f)
            y_offset = (f_m1 - f_p1) / (2.0f * denom);
    }

    return cv::Point2f(static_cast<float>(x) + x_offset,
                       static_cast<float>(y) + y_offset);
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


std::vector<std::vector<cv::Mat>> get_cell_warped_images_vector(cv::Mat const& image, cv::Mat const& cross_locs)
{
    // Warp each clue cell to 20x20, matching the resolution the real-photo
    // digit models were trained on (the marked 20x20 clue cells). Keeping the
    // warp consistent with the training data makes both the single-digit and
    // the counter model operate in-distribution.
    auto const cell_warped_side_length = 20;
    cv::Size const cell_warped_size(cell_warped_side_length, cell_warped_side_length);

    auto const cell_warped_images_vector_size = cross_locs.size() - cv::Size(1, 1);

    std::vector<std::vector<cv::Mat>> cell_warped_images_vector(
        cell_warped_images_vector_size.height,
        std::vector<cv::Mat>(cell_warped_images_vector_size.width));

    for (int tl_x = 0, br_x = 1; br_x < cross_locs.cols; ++tl_x, ++br_x)
    {
        for (int tl_y = 0, br_y = 1; br_y < cross_locs.rows; ++tl_y, ++br_y)
        {
            cv::Point const tl(tl_x, tl_y);
            cv::Point const tr(br_x, tl_y);
            cv::Point const br(br_x, br_y);
            cv::Point const bl(tl_x, br_y);

            cv::Point2f const cell_tl = cross_locs.at<cv::Point>(tl);
            cv::Point2f const cell_tr = cross_locs.at<cv::Point>(tr);
            cv::Point2f const cell_br = cross_locs.at<cv::Point>(br);
            cv::Point2f const cell_bl = cross_locs.at<cv::Point>(bl);

            std::vector<cv::Point2f> const cell_points = {
                cell_tl, cell_tr, cell_br, cell_bl };

            cv::Point2f const cell_warped_tl(0, 0);
            cv::Point2f const cell_warped_tr(cell_warped_side_length, 0);
            cv::Point2f const cell_warped_br(cell_warped_side_length, cell_warped_side_length);
            cv::Point2f const cell_warped_bl(0, cell_warped_side_length);

            std::vector<cv::Point2f> const cell_warped_points = {
                cell_warped_tl, cell_warped_tr, cell_warped_br, cell_warped_bl };

            auto const warp_matrix =
                cv::getPerspectiveTransform(cell_points, cell_warped_points);

            cv::Mat cell_warped_image;
            cv::warpPerspective(image, cell_warped_image, warp_matrix, cell_warped_size);

            cell_warped_images_vector[tl_y][tl_x] = cell_warped_image;
        }
    }

    return cell_warped_images_vector;
}


}
