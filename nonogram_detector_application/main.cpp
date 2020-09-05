#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "cross_locs_detector.hpp"
#include "image_operations.hpp"

// Returns cv::Mat(cross_locs.size() - cv::Size(1, 1), CV_32SC4)
cv::Mat get_cell_rois(cv::Mat const& cross_locs)
{
    std::cout << cross_locs.size() << std::endl;

    auto const cell_rois_size = cross_locs.size() - cv::Size(1, 1);
    cv::Mat cell_rois(cell_rois_size, CV_32SC4);

    for (int tl_x = 0, br_x = 1; br_x < cross_locs.cols; ++tl_x, ++br_x)
    {
        for (int tl_y = 0, br_y = 1; br_y < cross_locs.rows; ++tl_y, ++br_y)
        {
            cv::Point const tl(tl_x, tl_y);
            cv::Point const br(br_x, br_y);

            cv::Rect const cell_roi(
                cross_locs.at<cv::Point>(tl),
                cross_locs.at<cv::Point>(br));

            cell_rois.at<cv::Rect>(tl) = cell_roi;

            //if ((4 * tl_x * tl_x + 2 * tl_y * tl_x + 3 * tl_y * tl_y) % 5 == 0)
            //{
            //    cv::rectangle(image_draw, cell_roi, cv::Scalar(0, 255, 0), -1);
            //}
        }
    }

    //cv::Mat cell_rois_cropped(cell_rois.size(), cell_rois.type());
    //std::transform(
    //    cell_rois.begin<cv::Rect>(),
    //    cell_rois.end<cv::Rect>(),
    //    cell_rois_cropped.begin<cv::Rect>(),
    //    [](cv::Rect const roi)
    //    {
    //        cv::Rect const roi_cropped(
    //            roi.tl() + cv::Point(0.13f * roi.width, 0.2f * roi.height),
    //            roi.br() - cv::Point(0.085f * roi.width, 0.08f * roi.height));

    //        return roi_cropped;
    //    });

    return cell_rois;
}

std::vector<std::vector<cv::Mat>> get_cell_images(cv::Mat const& image, cv::Mat const& cell_rois)
{
    std::vector<std::vector<cv::Mat>> cell_images(
        cell_rois.rows,
        std::vector<cv::Mat>(cell_rois.cols));
    for (auto i = 0; i < cell_rois.rows; ++i)
    {
        for (auto j = 0; j < cell_rois.cols; ++j)
        {
            auto const& roi = cell_rois.at<cv::Rect>(i, j);

            cell_images[i][j] = image(roi).clone();
        }
    }

    return cell_images;
}


void save_images(std::string const& images_dir_path, std::vector<std::vector<cv::Mat>> const& cell_images)
{
    for (auto i = 0; i < cell_images.size(); ++i)
    {
        for (auto j = 0; j < cell_images.front().size(); ++j)
        {
            std::ostringstream image_path_stream;

            image_path_stream << images_dir_path << R"(\)";

            image_path_stream << std::setw(3) << std::setfill('0');
            image_path_stream << i;
            image_path_stream << "_";
            image_path_stream << std::setw(3) << std::setfill('0');
            image_path_stream << j << ".png";

            auto const image_path = image_path_stream.str();

            auto const image = cell_images[i][j];

            cv::imwrite(image_path, image);
        }
    }
}


int main()
{
    //std::string const image_path =
    //    R"(C:\Users\klimenkov\Desktop\nonograms\20191102_004052.jpg)";
    std::string const name = "nonogram";
    std::string const image_path =
        R"(C:\Users\klimenkov\Desktop\nonograms\)" + name + ".jpg";
    //std::string const image_path =
    //    R"(C:\Users\klimenkov\Desktop\nonograms\vqtsmfq7o3k21.jpg)";

    auto image = cv::imread(image_path);

    if (image.empty())
    {
        std::cout << "Image was not read" << std::endl;

        return 1;
    }

    //cv::imshow("image", image);
    //cv::waitKey();

    // for Yan nonogram
    //ng::CrossLocsDetector cross_loc_detector(2200, 15, 4.0, 5, 50, 0.9);

    ng::CrossLocsDetector cross_loc_detector(1200, 15, 10.0, 5, 50, 0.9);

    cv::Mat image_gray;
    cv::cvtColor(image, image_gray, cv::COLOR_BGR2GRAY);

    auto const image_thresholded =
        ng::threshold(image_gray, 15, 10.0);

    bool cell_loc_found;
    cv::Mat cross_locs_main;
    cv::Mat cross_locs_top;
    cv::Mat cross_locs_left;
    std::tie(cell_loc_found, cross_locs_main, cross_locs_top, cross_locs_left) =
        cross_loc_detector.detect(image);

    int const radius = 8;
    auto image_draw = ng::CrossLocsDetector::draw(image, cross_locs_main, radius, cv::Scalar(255, 0, 0));
    image_draw = ng::CrossLocsDetector::draw(image_draw, cross_locs_top, radius, cv::Scalar(0, 255, 0));
    image_draw = ng::CrossLocsDetector::draw(image_draw, cross_locs_left, radius, cv::Scalar(0, 0, 255));

    //cv::Mat image_thresholded_visible = image_thresholded * 255;

    //cv::resize(image_thresholded_visible, image_thresholded_visible, {}, 0.25, 0.25);
    cv::resize(image_draw, image_draw, {}, 0.25, 0.25);

    ////cv::imwrite("grid.png", image_draw);
    cv::imshow("image_draw", image_draw);
    //cv::imshow("image_thresholded_visible", image_thresholded_visible);
    cv::waitKey();

    //auto const cell_rois = get_cell_rois(cross_locs_left);
    //auto const cell_images = get_cell_images(image, cell_rois);
    auto const cell_images = ng::get_cell_warped_images_vector(image_thresholded, cross_locs_left);

    //std::string const images_dir_path =
    //    R"(C:\Users\klimenkov\Desktop\nonograms_digits\)" + name + R"(\left)";
    //save_images(images_dir_path, cell_images);

    return 0;
}
