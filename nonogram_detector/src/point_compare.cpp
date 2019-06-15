#include "point_compare.hpp"

bool PointCompare::operator()(cv::Point const& p1, cv::Point const& p2) const
{
    return p1.x == p2.x ? p1.y < p2.y : p1.x < p2.x;
}
