#include "digit_recognizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ng
{

namespace
{

constexpr int kMnistSize = 28;
constexpr double kMean = 0.1307;
constexpr double kStd = 0.3081;

// Returns the bounding box of the largest contour in <binary> (whose foreground
// is white on black), or an empty rect if there is none. The caller should feed
// a region with the grid frame already removed so the digit is the dominant
// foreground object.
cv::Rect largest_contour_bbox(cv::Mat const& binary)
{
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double best_area = 0.0;
    cv::Rect best{0, 0, 0, 0};

    for (auto const& contour : contours)
    {
        double const area = cv::contourArea(contour);
        if (area > best_area)
        {
            best_area = area;
            best = cv::boundingRect(contour);
        }
    }

    return best;
}

}

DigitRecognizer::DigitRecognizer(std::filesystem::path const& model_path)
{
    net_ = cv::dnn::readNetFromONNX(model_path.string());
    if (net_.empty())
        throw std::runtime_error("DigitRecognizer: failed to load ONNX model from " + model_path.string());
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

    // Isolate the dark digit from the (lighter) paper: result is white digit on
    // black, which is the polarity MNIST expects.
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // Crop inward past the surrounding grid-frame ring. The frame always sits
    // along the cell border, so removing this margin lets findContours see only
    // the digit instead of the frame.
    int const margin = std::max(2, static_cast<int>(std::round(binary.cols * 0.2)));
    cv::Rect inner(margin, margin,
        binary.cols - 2 * margin, binary.rows - 2 * margin);
    if (inner.width < 3 || inner.height < 3)
        return false;

    cv::Mat inner_binary = binary(inner);
    cv::Rect bbox = largest_contour_bbox(inner_binary);

    if (bbox.width < 3 || bbox.height < 3)
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

    cv::Mat logits_flat = logits.reshape(1, 1);

    // Softmax-normalize the logits so the top value is a probability in [0, 1],
    // which is a meaningful per-class confidence for the confidence_min gate.
    double max_val = 0.0;
    cv::Point max_loc;
    cv::minMaxLoc(logits_flat, nullptr, &max_val, nullptr, &max_loc);

    cv::Mat shifted = logits_flat - cv::Scalar(max_val);
    cv::exp(shifted, shifted);
    double const sum = cv::sum(shifted)[0];
    confidence = shifted.at<float>(0, max_loc.x) / sum;

    return max_loc.x;
}

}
