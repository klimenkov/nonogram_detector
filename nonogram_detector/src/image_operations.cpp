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


void refine_cross_locs_ink(
    cv::Mat const& image_gray,
    cv::Mat& cross_locs,
    int const window_radius)
{
    if (cross_locs.empty())
    {
        return;
    }

    // Half-width (px) of the per-axis band whose ink is considered; also the
    // maximum distance one refinement pass can move a location, since the
    // centroid is computed over the band alone.
    int const BAND = 3;
    // Minimum baseline-subtracted ink mass for a trustworthy per-axis center;
    // below this the band holds no line (empty paper, extrapolated padding).
    double const INK_MASS_MIN = 400.0;
    // The refinement iterates: each pass re-centres the band on the refined
    // location, so a bold line that the band truncated on the first pass
    // (previous stage landed near the band edge) is fully visible on the next
    // and the centroid converges on the true center. Two to three passes
    // suffice; four is a safety cap.
    int const MAX_PASSES = 4;
    double const CONVERGED_MOVE = 0.05;

    for (int r = 0; r < cross_locs.rows; ++r)
    {
        for (int c = 0; c < cross_locs.cols; ++c)
        {
            cv::Point2f& cross_loc = cross_locs.at<cv::Point2f>(r, c);
            if (cross_loc == cv::Point2f(-1.0f, -1.0f))
            {
                continue;
            }

            for (int pass = 0; pass < MAX_PASSES; ++pass)
            {
                int const px = cvRound(cross_loc.x);
                int const py = cvRound(cross_loc.y);
                int const x0 = px - window_radius, x1 = px + window_radius;
                int const y0 = py - window_radius, y1 = py + window_radius;
                if (x0 < 0 || y0 < 0 || x1 >= image_gray.cols || y1 >= image_gray.rows)
                {
                    break;  // clipped at the image border; keep the location
                }

                // Window paper level: its brightest pixel.
                uchar paper = 0;
                for (int y = y0; y <= y1; ++y)
                {
                    for (int x = x0; x <= x1; ++x)
                        paper = std::max(paper, image_gray.at<uchar>(y, x));
                }

                // Per-axis ink profiles. The vertical line's column profile is
                // summed over only the rows near the current location, and the
                // horizontal line's row profile over only the columns near it,
                // so ink that is far away on the other axis (a digit, the
                // neighbouring line) does not contaminate the profile.
                int const bx0 = std::max(x0, px - BAND), bx1 = std::min(x1, px + BAND);
                int const by0 = std::max(y0, py - BAND), by1 = std::min(y1, py + BAND);

                std::vector<double> col_mass(x1 - x0 + 1, 0.0);
                std::vector<double> row_mass(y1 - y0 + 1, 0.0);
                for (int y = by0; y <= by1; ++y)
                {
                    for (int x = x0; x <= x1; ++x)
                        col_mass[x - x0] +=
                            std::max(0.0, static_cast<double>(paper) - image_gray.at<uchar>(y, x));
                }
                for (int y = y0; y <= y1; ++y)
                {
                    for (int x = bx0; x <= bx1; ++x)
                        row_mass[y - y0] +=
                            std::max(0.0, static_cast<double>(paper) - image_gray.at<uchar>(y, x));
                }

                // Centroid of each profile over the band, with the profile's
                // baseline (the crossing line's uniform contribution along
                // this axis) subtracted so only the profiled line's own ink
                // moves the center. The location keeps its per-axis value when
                // the band carries too little ink.
                auto const refine_axis = [](std::vector<double> const& mass,
                                            int const coord0,
                                            int const b_lo,
                                            int const b_hi,
                                            double const mass_min,
                                            float& coord) {
                    double baseline = mass.front();
                    for (double const m : mass)
                        baseline = std::min(baseline, m);

                    double s = 0.0, sw = 0.0;
                    for (int i = b_lo; i <= b_hi; ++i)
                    {
                        double const w = std::max(0.0, mass[i] - baseline);
                        s += (coord0 + i) * w;
                        sw += w;
                    }
                    if (sw > mass_min)
                        coord = static_cast<float>(s / sw);
                };

                float const x_before = cross_loc.x;
                float const y_before = cross_loc.y;
                refine_axis(col_mass, x0, bx0 - x0, bx1 - x0, INK_MASS_MIN, cross_loc.x);
                refine_axis(row_mass, y0, by0 - y0, by1 - y0, INK_MASS_MIN, cross_loc.y);

                bool const settled =
                    std::fabs(cross_loc.x - x_before) < CONVERGED_MOVE &&
                    std::fabs(cross_loc.y - y_before) < CONVERGED_MOVE;
                if (settled)
                {
                    break;
                }
            }
        }
    }
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

    if (peak_max > similarity_ratio_min)
    {
        return std::make_pair(true, refine_peak_loc(image_filtered, peak_max_loc));
    }

    return std::make_pair(false, cv::Point2f(-1.0f, -1.0f));
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
        return std::make_pair(false, cv::Point2f(-1.0f, -1.0f));
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
        std::make_pair(false, cv::Point2f(-1.0f, -1.0f));
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

            cv::Point2f const cell_tl = cross_locs.at<cv::Point2f>(tl);
            cv::Point2f const cell_tr = cross_locs.at<cv::Point2f>(tr);
            cv::Point2f const cell_br = cross_locs.at<cv::Point2f>(br);
            cv::Point2f const cell_bl = cross_locs.at<cv::Point2f>(bl);

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
