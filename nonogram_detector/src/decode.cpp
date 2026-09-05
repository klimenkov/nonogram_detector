#include "decode.hpp"

#include <utility>

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

// Whole-cell read must clear this softmax confidence to override a split
// (counter false-positive on a single digit): a high-conf whole read is a
// strong signal that the counter is wrong. Calibrated on the corpus: genuine
// singles read >= 0.9; the counter's two-digit FPs on singles score < 0.6.
constexpr double kWholeHighConfMin = 0.9;

}

// Guard decision for a counter-flagged two-digit cell. <split> is the composed
// split read (-1 if a half failed), <whole>/<whole_conf> the whole-cell read.
// Returns the chosen digit (or -1). Declared in decode.hpp so decode_region and
// the unit tests exercise the same decision logic.
int resolve_two_digit(int split, int whole, double whole_conf,
                      double conf_l, double conf_r,
                      int max_clue,
                      double split_conf_min, double whole_high_conf_min)
{
    bool const plausible = split >= 10 && split <= max_clue;
    bool const both_halves_confident =
        conf_l >= split_conf_min && conf_r >= split_conf_min;
    bool const prefer_whole =
        whole >= 1 && whole_conf >= whole_high_conf_min && !both_halves_confident;
    if (plausible && !prefer_whole)
        return split;
    return whole;
}

int sanitize_clue_digit(int digit)
{
    return (digit == 0) ? -1 : digit;
}

namespace
{

// Recognizes one warped clue cell and fills <info>. Mirrors the existing
// decision logic exactly (counter -> split/whole -> guard -> sanitize).
void recognize_cell(
    cv::Mat const& cell,
    DigitRecognizer const& recognizer,
    int max_clue_value,
    int& digit,
    int& count,
    double& whole_conf,
    double& conf_l,
    double& conf_r)
{
    count = recognizer.digit_count(cell);
    digit = -1;
    whole_conf = 0.0;
    conf_l = 0.0;
    conf_r = 0.0;
    if (count == 2)
    {
        int const split = recognizer.recognize_two_digits_ex(
            cell, count, 3, kSplitConfidenceMin, conf_l, conf_r);
        int const whole = recognizer.recognize_ex(cell, whole_conf);
        digit = resolve_two_digit(split, whole, whole_conf, conf_l, conf_r,
                                  max_clue_value, kSplitConfidenceMin,
                                  kWholeHighConfMin);
    }
    else
    {
        digit = recognizer.recognize(cell);
    }
    digit = sanitize_clue_digit(digit);
}

// Warps each clue cell of <cross_locs> into a fixed-size image, recognizes the
// digit, and fills <out> (row-major [row][col]), <out_count> (per-cell digit
// count 1/2, 0 for empty/unreliable), and <info> (per-cell ClueCellInfo).
void decode_region(
    cv::Mat const& image,
    cv::Mat const& cross_locs,
    DigitRecognizer const& recognizer,
    int max_clue_value,
    std::vector<std::vector<int>>& out,
    std::vector<std::vector<int>>& out_count,
    std::vector<std::vector<ClueCellInfo>>& info)
{
    auto const cells = get_cell_warped_images_vector(image, cross_locs);

    out.resize(cells.size());
    out_count.resize(cells.size());
    info.resize(cells.size());

    for (std::size_t row = 0; row < cells.size(); ++row)
    {
        out[row].reserve(cells[row].size());
        out_count[row].reserve(cells[row].size());
        info[row].reserve(cells[row].size());
        for (std::size_t col = 0; col < cells[row].size(); ++col)
        {
            ClueCellInfo cell_info;
            cell_info.cell = cells[row][col];
            int digit = -1, count = 0;
            recognize_cell(cell_info.cell, recognizer, max_clue_value,
                           digit, count,
                           cell_info.whole_conf, cell_info.conf_l, cell_info.conf_r);
            cell_info.digit = digit;
            cell_info.count = digit < 0 ? 0 : count;
            out[row].push_back(digit);
            out_count[row].push_back(cell_info.count);
            info[row].push_back(std::move(cell_info));
        }
    }
}

}

bool decode_clues_ex(
    cv::Mat const& image,
    Detection const& detection,
    DigitRecognizer const& recognizer,
    ClueGrid& out,
    std::vector<std::vector<ClueCellInfo>>& top_info,
    std::vector<std::vector<ClueCellInfo>>& left_info)
{
    if (!detection.found)
        return false;

    bool ok = !detection.top.empty() || !detection.left.empty();
    if (!ok)
        return false;

    int const max_clue_value = [&detection]() {
        if (detection.main.empty())
            return 99;
        return std::max(detection.main.rows, detection.main.cols) - 1;
    }();

    if (!detection.top.empty())
        decode_region(image, detection.top, recognizer, max_clue_value,
                      out.top, out.top_count, top_info);
    if (!detection.left.empty())
        decode_region(image, detection.left, recognizer, max_clue_value,
                      out.left, out.left_count, left_info);

    return true;
}

bool decode_clues(
    cv::Mat const& image,
    Detection const& detection,
    DigitRecognizer const& recognizer,
    ClueGrid& out)
{
    std::vector<std::vector<ClueCellInfo>> top_info, left_info;
    return decode_clues_ex(image, detection, recognizer, out, top_info, left_info);
}

}
