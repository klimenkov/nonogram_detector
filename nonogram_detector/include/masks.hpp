#pragma once

#include <utility>

#include <opencv2/opencv.hpp>

namespace ng
{

std::pair<cv::Mat, int> get_mask_square(int const side_length);


// This mask is used to detect empty cells
// Uses margin to ignore the width of the line
// Example: ng::get_mask_cross(9, 2) returns
// [-1, -1,  0,  0,  1,  0,  0, -1, -1;
//  -1, -1,  0,  0,  1,  0,  0, -1, -1;
//   0,  0,  0,  0,  1,  0,  0,  0,  0;
//   0,  0,  0,  0,  1,  0,  0,  0,  0;
//   1,  1,  1,  1,  1,  1,  1,  1,  1;
//   0,  0,  0,  0,  1,  0,  0,  0,  0;
//   0,  0,  0,  0,  1,  0,  0,  0,  0;
//  -1, -1,  0,  0,  1,  0,  0, -1, -1;
//  -1, -1,  0,  0,  1,  0,  0, -1, -1]
std::pair<cv::Mat, int> get_mask_cross(int const length, int const margin);


// This mask is used to detect both empty and cells with numbers
std::pair<cv::Mat, int> get_mask_cross(int const length);


}
