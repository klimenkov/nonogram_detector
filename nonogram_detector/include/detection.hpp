#pragma once

#include <opencv2/opencv.hpp>

namespace ng
{

// Result of a grid detection: a boolean found-flag plus the located grid
// intersection ("cross_loc") matrices for the main cells region and the
// top / left clue regions. Each matrix is CV_32FC2; element (x, y) holds the
// subpixel pixel position of the grid intersection at that index of the
// region. (-1, -1) marks a missing location.
struct Detection
{
    bool found = false;
    cv::Mat main;
    cv::Mat top;
    cv::Mat left;
};

}
