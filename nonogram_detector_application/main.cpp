#include <cstdlib>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "cross_locs_detector.hpp"
#include "image_operations.hpp"


int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <image> [resize_max]\n";
        return 2;
    }

    std::string const image_path = argv[1];
    int const resize_max = argc > 2 ? std::atoi(argv[2]) : 1200;

    auto image = cv::imread(image_path);
    if (image.empty())
    {
        std::cerr << "Image was not read: " << image_path << "\n";
        return 1;
    }

    ng::CrossLocsDetector cross_loc_detector(resize_max, 15, 10.0, 5, 50, 0.9);

    auto const detection = cross_loc_detector.detect(image);

    std::cout << "found=" << (detection.found ? "true" : "false") << "\n";

    if (!detection.found)
    {
        return 0;
    }

    int const radius = 8;
    auto image_draw =
        ng::CrossLocsDetector::draw(image, detection.main, radius, cv::Scalar(255, 0, 0));
    image_draw =
        ng::CrossLocsDetector::draw(image_draw, detection.top, radius, cv::Scalar(0, 255, 0));
    image_draw =
        ng::CrossLocsDetector::draw(image_draw, detection.left, radius, cv::Scalar(0, 0, 255));

    // Show the result unless running headless.
    if (std::getenv("DISPLAY"))
    {
        cv::resize(image_draw, image_draw, {}, 0.25, 0.25);
        cv::imshow("image_draw", image_draw);
        cv::waitKey();
    }
    else
    {
        // Optionally save the overlay for headless inspection.
        if (std::getenv("NG_SAVE_OUTPUT"))
        {
            cv::imwrite("grid.png", image_draw);
        }
    }

    return 0;
}
