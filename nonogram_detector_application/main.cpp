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

    auto const cross_locs_main_mat = ng::get_cross_locs_main_mat(
        image_thresholded,
        { cell_loc },
        cell_side_length,
        0.9);

    ng::print(cross_locs_main_mat);

    auto const cross_locs_top_mat = ng::get_cross_locs_top_mat(
        image_thresholded,
        cross_locs_main_mat,
        cell_side_length,
        0.9);

    ng::print(cross_locs_top_mat);

    auto const cross_locs_left_mat = ng::get_cross_locs_left_mat(
        image_thresholded,
        cross_locs_main_mat,
        cell_side_length,
        0.9);

    ng::print(cross_locs_left_mat);


    // -----------------------


    // ------------------

    image_thresholded *= 255;

    cv::imshow("image_thresholded", image_thresholded);
    cv::waitKey();

    return 0;
}
