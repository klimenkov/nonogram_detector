#include "decode.hpp"

#include "image_operations.hpp"

namespace ng
{

namespace
{

// Warps each clue cell of <cross_locs> into a fixed-size image and recognizes
// the digit, filling <out> (row-major [row][col]). Cells with no recognizer
// output become -1.
std::vector<std::vector<int>> decode_region(
    cv::Mat const& image,
    cv::Mat const& cross_locs,
    DigitRecognizer const& recognizer)
{
    auto const cells = get_cell_warped_images_vector(image, cross_locs);

    std::vector<std::vector<int>> result(cells.size());

    for (std::size_t row = 0; row < cells.size(); ++row)
    {
        result[row].reserve(cells[row].size());
        for (std::size_t col = 0; col < cells[row].size(); ++col)
            result[row].push_back(recognizer.recognize(cells[row][col]));
    }

    return result;
}

}

bool decode_clues(
    cv::Mat const& image,
    Detection const& detection,
    DigitRecognizer const& recognizer,
    ClueGrid& out)
{
    if (!detection.found)
        return false;

    bool ok = !detection.top.empty() || !detection.left.empty();
    if (!ok)
        return false;

    if (!detection.top.empty())
        out.top = decode_region(image, detection.top, recognizer);
    if (!detection.left.empty())
        out.left = decode_region(image, detection.left, recognizer);

    return true;
}

}
