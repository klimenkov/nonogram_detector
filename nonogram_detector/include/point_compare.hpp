#pragma once

#include "opencv2/opencv.hpp"

namespace ng
{

struct PointCompare
{
    bool operator()(cv::Point const& p1, cv::Point const& p2) const;
};

}
