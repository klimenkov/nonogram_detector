#include "decode.hpp"

#include "image_operations.hpp"

namespace ng
{

namespace
{

// Split-read halves must each clear this softmax confidence floor before the
// two-digit value is accepted; a half read below it (near-uniform softmax,
// ambiguous glyph) makes recognize_two_digits return -1 and decode falls back
// to the whole-cell read. Calibrated on the marked two-digit corpus: genuine
// halves score >= 0.48, ambiguous ones ~0.1.
constexpr double kSplitConfidenceMin = 0.3;

// Warps each clue cell of <cross_locs> into a fixed-size image and recognizes
// the digit, filling <out> (row-major [row][col]) and <out_count> (per-cell
// digit count 1/2, 0 for empty/unreliable). Cells with no recognizer output
// become -1 / 0.
void decode_region(
    cv::Mat const& image,
    cv::Mat const& cross_locs,
    DigitRecognizer const& recognizer,
    std::vector<std::vector<int>>& out,
    std::vector<std::vector<int>>& out_count)
{
    auto const cells = get_cell_warped_images_vector(image, cross_locs);

    out.resize(cells.size());
    out_count.resize(cells.size());

    for (std::size_t row = 0; row < cells.size(); ++row)
    {
        out[row].reserve(cells[row].size());
        out_count[row].reserve(cells[row].size());
        for (std::size_t col = 0; col < cells[row].size(); ++col)
        {
            int const count = recognizer.digit_count(cells[row][col]);
            int digit = -1;
            if (count == 2)
            {
                // Genuine two-digit clue cell: read it as two digits by
                // splitting the cell. Fall back to the whole-cell read when the
                // split fails (e.g. a counter false positive on a single digit
                // or a low-confidence half).
                digit = recognizer.recognize_two_digits(cells[row][col], count, 3, kSplitConfidenceMin);
                if (digit < 0)
                    digit = recognizer.recognize(cells[row][col]);
            }
            else
            {
                digit = recognizer.recognize(cells[row][col]);
            }
            out[row].push_back(digit);
            out_count[row].push_back(digit < 0 ? 0 : count);
        }
    }
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
        decode_region(image, detection.top, recognizer, out.top, out.top_count);
    if (!detection.left.empty())
        decode_region(image, detection.left, recognizer, out.left, out.left_count);

    return true;
}

}
