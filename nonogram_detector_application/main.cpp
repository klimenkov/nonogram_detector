#include <chrono>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "grid_detector.hpp"

int main()
{
    std::string const image_path =
        R"(C:\Users\klimenkov\Desktop\pc_old\nonograms\nonogram.jpg)";
    //std::string const image_path =
    //    R"(C:\Users\klimenkov\Desktop\pc_old\nonograms\photo_2018-08-18_13-28-02.jpg)";
    //std::string const image_path =
    //    R"(C:\Users\klimenkov\Desktop\pc_old\nonograms\20180811_114632.jpg)";

    auto image = cv::imread(image_path);

    //cv::resize(image, image, {}, 0.2, 0.2);

    //cv::imshow("image", image);
    //cv::waitKey();

    ng::CrossLocsDetector cross_loc_detector(900, 15, 9.0, 5, 50, 0.9);

    cv::Mat cross_locs_main;
    cv::Mat cross_locs_top;
    cv::Mat cross_locs_left;
    std::tie(cross_locs_main, cross_locs_top, cross_locs_left) =
        cross_loc_detector.detect(image);

    auto image_draw =
        ng::CrossLocsDetector::draw(image, cross_locs_main, cv::Scalar(255, 0, 0));
    image_draw =
        ng::CrossLocsDetector::draw(image_draw, cross_locs_top, cv::Scalar(0, 255, 0));
    image_draw =
        ng::CrossLocsDetector::draw(image_draw, cross_locs_left, cv::Scalar(0, 0, 255));

    cv::imshow("image_draw", image_draw);
    cv::waitKey();

    return 0;
}
