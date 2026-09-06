#include "digit_recognizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace ng
{

namespace
{

constexpr int kMnistSize = 28;
constexpr double kMean = 0.1307;
constexpr double kStd = 0.3081;

// Softmax-normalizes a flat 1xN logits row. Returns the argmax location and
// the full probability row (CV_32F, same shape), so callers can read the
// probability of any class, not just the top one.
std::pair<cv::Point, cv::Mat> softmax_probs(cv::Mat const& logits_flat)
{
    double max_val = 0.0;
    cv::Point max_loc;
    cv::minMaxLoc(logits_flat, nullptr, &max_val, nullptr, &max_loc);

    cv::Mat shifted = logits_flat - cv::Scalar(max_val);
    cv::exp(shifted, shifted);
    double const sum = cv::sum(shifted)[0];
    cv::Mat probs = shifted / sum;

    return std::make_pair(max_loc, probs);
}

}

DigitRecognizer::DigitRecognizer(std::filesystem::path const& model_path)
{
    net_ = cv::dnn::readNetFromONNX(model_path.string());
    if (net_.empty())
        throw std::runtime_error("DigitRecognizer: failed to load ONNX model from " + model_path.string());
}

void DigitRecognizer::set_counter_model(std::filesystem::path const& model_path)
{
    counter_net_ = cv::dnn::readNetFromONNX(model_path.string());
    if (counter_net_.empty())
        throw std::runtime_error("DigitRecognizer: failed to load counter ONNX model from " + model_path.string());
}

bool DigitRecognizer::prepare_input(cv::Mat const& cell, cv::Mat& out_blob)
{
    if (cell.empty())
        return false;

    cv::Mat gray;
    if (cell.channels() == 3)
        cv::cvtColor(cell, gray, cv::COLOR_BGR2GRAY);
    else
        gray = cell;

    // Crop inward past the surrounding grid-frame ring. The frame always sits
    // along the cell border, so removing this margin leaves only the digit's
    // ink inside the crop.
    int const margin = std::max(2, static_cast<int>(std::round(gray.cols * 0.2)));
    cv::Rect inner(margin, margin,
        gray.cols - 2 * margin, gray.rows - 2 * margin);
    if (inner.width < 3 || inner.height < 3)
        return false;

    // Minimum contrast gate: an empty paper cell has virtually flat brightness
    // (contrast < 20), while real printed digits exhibit contrast > 75. Reject
    // empty cells before thresholding so Otsu never amplifies paper grain/sensor
    // noise into spurious digit predictions.
    double in_min = 0.0, in_max = 0.0;
    cv::minMaxLoc(gray(inner), &in_min, &in_max);
    if (in_max - in_min < 35.0)
        return false;

    // Isolate the dark digit from the (lighter) paper: result is white digit on
    // black, which is the polarity MNIST expects.
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // Union of all foreground pixels: the margin ring already stripped the
    // grid frame, so remaining ink is the digit. A thin digit (a "1" with a
    // 2 px stem, possibly fragmented by the margin crop) is kept; only nearly
    // empty cells fail the area gate.
    std::vector<cv::Point> nz;
    cv::findNonZero(binary(inner), nz);
    if (nz.size() < 10)
        return false;
    cv::Rect bbox = cv::boundingRect(nz);
    if (bbox.width < 2 || bbox.height < 5)
        return false;

    // Translate back into full-cell coordinates and add a small pad.
    int const pad = 2;
    cv::Rect roi(
        std::max(0, inner.x + bbox.x - pad),
        std::max(0, inner.y + bbox.y - pad),
        std::min(binary.cols, inner.x + bbox.x + bbox.width + pad) - std::max(0, inner.x + bbox.x - pad),
        std::min(binary.rows, inner.y + bbox.y + bbox.height + pad) - std::max(0, inner.y + bbox.y - pad));

    cv::Mat digit = binary(roi);

    // Rescale preserving aspect ratio into a kMnistSize canvas (black bg).
    double const scale = static_cast<double>(kMnistSize - 4) /
        static_cast<double>(std::max(digit.cols, digit.rows));
    int const nw = static_cast<int>(std::round(digit.cols * scale));
    int const nh = static_cast<int>(std::round(digit.rows * scale));

    cv::Mat resized;
    cv::resize(digit, resized, cv::Size(nw, nh), 0, 0, cv::INTER_AREA);

    cv::Mat canvas(kMnistSize, kMnistSize, CV_8UC1, cv::Scalar(0));
    cv::Rect const dst((kMnistSize - nw) / 2, (kMnistSize - nh) / 2, nw, nh);
    resized.copyTo(canvas(dst));

    // blobFromImage produces the 1 x 1 x 28 x 28 blob and applies
    // (img * scalefactor - mean). Choosing scalefactor/mean to yield the model's
    // expected whitening (x/255 - 0.1307) / 0.3081.
    out_blob = cv::dnn::blobFromImage(
        canvas,
        1.0 / (255.0 * kStd),
        cv::Size(kMnistSize, kMnistSize),
        cv::Scalar(kMean / kStd),
        false /*swapRB*/,
        false /*crop*/);
    return true;
}

int DigitRecognizer::recognize(cv::Mat const& cell, double confidence_min) const
{
    double confidence = 0.0;
    int const digit = recognize_ex(cell, confidence);
    if (digit < 0)
        return -1;
    return confidence < confidence_min ? -1 : digit;
}

int DigitRecognizer::recognize_ex(cv::Mat const& cell, double& confidence) const
{
    confidence = 0.0;

    cv::Mat blob;
    if (!prepare_input(cell, blob))
        return -1;

    net_.setInput(blob);
    cv::Mat logits = net_.forward();

    if (logits.empty() || logits.total() != 10)
        return -1;

    cv::Point max_loc;
    cv::Mat probs;
    std::tie(max_loc, probs) = softmax_probs(logits.reshape(1, 1));
    confidence = probs.at<float>(0, max_loc.x);

    return max_loc.x;
}

int DigitRecognizer::digit_count(cv::Mat const& cell) const
{
    double prob_two = 0.0;
    return digit_count_ex(cell, prob_two);
}

int DigitRecognizer::digit_count_ex(cv::Mat const& cell, double& prob_two) const
{
    prob_two = 0.0;
    if (counter_net_.empty())
        return -1;

    cv::Mat blob;
    if (!prepare_input(cell, blob))
        return -1;

    counter_net_.setInput(blob);
    cv::Mat logits = counter_net_.forward();

    if (logits.empty() || logits.total() != 2)
        return -1;

    cv::Point max_loc;
    cv::Mat probs;
    std::tie(max_loc, probs) = softmax_probs(logits.reshape(1, 1));
    prob_two = probs.at<float>(0, 1);

    return max_loc.x + 1; // class 0 -> 1 digit, class 1 -> 2 digits
}

int DigitRecognizer::recognize_two_digits_ex(
    cv::Mat const& cell,
    int count,
    int upscale,
    double split_conf_min,
    double& conf_l,
    double& conf_r) const
{
    conf_l = 0.0;
    conf_r = 0.0;
    if (count != 2)
        return -1;

    int const w = cell.cols;
    int const hw = w / 2;
    if (hw <= 0)
        return -1;

    cv::Mat left = cell.colRange(0, hw);
    cv::Mat right = cell.colRange(hw, w);

    cv::Mat left_up, right_up;
    if (upscale > 1)
    {
        cv::resize(left, left_up, cv::Size(), upscale, upscale, cv::INTER_CUBIC);
        cv::resize(right, right_up, cv::Size(), upscale, upscale, cv::INTER_CUBIC);
    }
    else
    {
        left_up = left;
        right_up = right;
    }

    int l = -1, r = -1;
    double cl = 0.0, cr = 0.0;
    l = recognize_ex(left_up, cl);
    if (l >= 0 && cl < split_conf_min)
        l = -1;
    r = recognize_ex(right_up, cr);
    if (r >= 0 && cr < split_conf_min)
        r = -1;
    conf_l = cl;
    conf_r = cr;

    if (l < 0 || r < 0)
        return -1;

    int const value = l * 10 + r;
    if (value > 99)
        return -1;
    return value;
}

int DigitRecognizer::recognize_two_digits(
    cv::Mat const& cell,
    int count,
    int upscale,
    double confidence_min) const
{
    double cl = 0.0, cr = 0.0;
    return recognize_two_digits_ex(cell, count, upscale, confidence_min, cl, cr);
}

}
