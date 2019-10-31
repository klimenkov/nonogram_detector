#include "masks.hpp"

namespace ng
{

std::pair<cv::Mat, int> get_mask_square(int const side_length)
{
    cv::Mat mask_square = cv::Mat::ones(side_length, side_length, CV_32S);
    mask_square(cv::Rect(1, 1, side_length - 2, side_length - 2)) = -1;

    auto const mask_square_perimeter = 4 * (side_length - 1);

    return std::make_pair(mask_square, mask_square_perimeter);
}

std::pair<cv::Mat, int> get_mask_cross(int const length, int const margin)
{
    assert(length % 2 == 1);

    cv::Mat mask_cross = (-1) * cv::Mat::ones(length, length, CV_32S);

    auto const length_half = length / 2;
    mask_cross(cv::Rect(length_half - margin, 0, 2 * margin + 1, length)) = 0;
    mask_cross(cv::Rect(0, length_half - margin, length, 2 * margin + 1)) = 0;

    mask_cross(cv::Rect(length_half, 0, 1, length)) = 1;
    mask_cross(cv::Rect(0, length_half, length, 1)) = 1;

    auto const mask_cross_perimeter = 2 * length - 1;

    return std::make_pair(mask_cross, mask_cross_perimeter);
}

std::pair<cv::Mat, int> get_mask_cross(int const length)
{
    assert(length % 2 == 1);

    cv::Mat mask_cross = cv::Mat::zeros(length, length, CV_32S);

    auto const length_half = length / 2;

    mask_cross(cv::Rect(length_half, 0, 1, length)) = 1;
    mask_cross(cv::Rect(0, length_half, length, 1)) = 1;

    auto const mask_cross_perimeter = 2 * length - 1;

    return std::make_pair(mask_cross, mask_cross_perimeter);
}

}