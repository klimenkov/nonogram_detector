#pragma once

#include <utility>

#include <opencv2/opencv.hpp>

namespace ng
{

std::pair<cv::Mat, int> get_mask_square(int const side_length);

std::pair<cv::Mat, int> get_mask_cross(int const length, int const margin);

}
