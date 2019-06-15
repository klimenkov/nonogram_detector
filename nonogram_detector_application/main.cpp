#include <chrono>
#include <iostream>

#include "image_operations.hpp"
#include "masks.hpp"

int main()
{
    std::string const image_path = R"(C:\Users\klimenkov\Desktop\pc_old\nonograms\nonogram.jpg)";

    auto image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    image = ng::resize(image, 800);
    auto const image_thresholded = ng::threshold(image, 15, 5.0);

    cv::Point const image_center(image_thresholded.size() / 2);
    auto const cell_loc_roi = ng::get_roi(image_center, { 50, 50 });

    bool cell_loc_found;
    int cell_side_length;
    cv::Point2f cell_loc;
    std::tie(cell_loc_found, cell_side_length, cell_loc) =
        ng::find_cell_side_length_cell_loc(image_thresholded, cell_loc_roi, 5, 50, 0.9);

    std::cout << "cell_side_length: " << cell_side_length << std::endl;
    std::cout << "cell_loc: " << cell_loc << std::endl;

    // -----------------

    auto const cell_side_length_odd = cell_side_length / 2 * 2 + 1;
    auto const line_width = static_cast<int>(cell_side_length / 6.5);
    auto const line_width_half = line_width / 2;
    cv::Mat mask_cross;
    int mask_cross_perimeter;
    std::tie(mask_cross, mask_cross_perimeter) =
        ng::get_mask_cross(cell_side_length_odd, line_width_half);

    auto begin = std::chrono::steady_clock().now();

    auto const cross_locs_map = ng::get_cross_locs_map(
        image_thresholded,
        cell_loc,
        cell_side_length,
        mask_cross,
        mask_cross_perimeter,
        0.9);

    auto end = std::chrono::steady_clock().now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms" << std::endl;

    std::cout << cross_locs_map.size() << std::endl;

    // ------------------

    auto image_copy = cv::imread(image_path);
    image_copy = ng::resize(image_copy, 800);

    image_thresholded *= 255;

    //cv::rectangle(
    //    image_thresholded,
    //    cv::Rect(
    //        cell_loc - cv::Point2f(1.0f, 1.0f),
    //        cell_loc + cv::Point2f(cell_side_length, cell_side_length) + cv::Point2f(1.0f, 1.0f)),
    //    128,
    //    1);

    //cv::imshow("image_thresholded_roi", image_thresholded(ng::get_roi(cell_loc, cv::Size(2 * cell_side_length, 2 * cell_side_length))));
    cv::imshow("image_thresholded", image_thresholded);
    cv::waitKey();

    return 0;
}
