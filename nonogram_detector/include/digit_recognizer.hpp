#pragma once

#include <filesystem>

#include <opencv2/opencv.hpp>

namespace ng
{

// Recognizes a single digit inside a warped nonogram clue cell using a
// pre-trained MNIST convolutional network in ONNX format, loaded through
// OpenCV's dnn module.
//
// The model is tiny (~26 KB) and committed under
// <repo>/nonogram_detector/models/digits.onnx, so recognition is fully offline
// and headless. The cell image is geometrically normalized (grayscale, Otsu
// foreground isolation, crop to the digit's bounding box, rescale to 28x28)
// before the forward pass, which lets the model tolerate blur, skew, noise and
// the printed/hand-drawn digits found in real photos.
class DigitRecognizer
{
public:
    // Loads the ONNX model from <model_path>. Throws std::runtime_error if the
    // file cannot be read or parsed.
    explicit DigitRecognizer(std::filesystem::path const& model_path);

    // Loads a second ONNX model that classifies each cell as holding 1 or 2
    // digits. Throws std::runtime_error if the file cannot be read. The counter
    // model shares the same input normalization as the digit model (28x28) and
    // emits 2 logits. Not required for recognition; digit_count() returns -1
    // when no counter model is set.
    void set_counter_model(std::filesystem::path const& model_path);

    // Returns the recognized digit in [0, 9], or -1 if the cell appears empty
    // (no foreground) or the top class confidence is below <confidence_min>.
    int recognize(cv::Mat const& cell, double confidence_min = 0.0) const;

    // Like recognize(), but also reports the softmax probability of the chosen
    // class in <confidence> (set to 0 on an empty/unreliable cell). Useful for
    // calibrating a confidence_min threshold on a corpus of cells.
    int recognize_ex(cv::Mat const& cell, double& confidence) const;

    // Returns the number of digits in the cell: 1 or 2, as classified by the
    // counter model (argmax of 2 logits -> 1 or 2). Returns -1 if no counter
    // model is configured or the cell has no reliable foreground. The returned
    // label is the raw argmax; it does not apply a confidence gate.
    int digit_count(cv::Mat const& cell) const;

    // Like digit_count(), but also reports the softmax probability of the
    // two-digit class in <prob_two>. Returns -1 when no counter model is set or
    // the cell has no reliable foreground.
    int digit_count_ex(cv::Mat const& cell, double& prob_two) const;

    // Reads a cell that holds two digits: splits the warped cell into
    // left/right halves, upscales each by <upscale> with INTER_CUBIC (so each
    // half is large enough for reliable single-digit recognition), reads each
    // half, and returns left*10+right. <count> must be the cell's digit_count()
    // result, which the caller is expected to have computed already (this
    // avoids a second counter-model forward pass); anything other than 2
    // returns -1. Each half must also clear <confidence_min> softmax
    // confidence. Returns -1 when either half cannot be read reliably or the
    // composed value leaves [0, 99].
    int recognize_two_digits(cv::Mat const& cell, int count, int upscale = 3,
                             double confidence_min = 0.0) const;

    // Like recognize_two_digits(), but reports the per-half softmax confidences
    // (conf_l / conf_r; 0.0 for a rejected/empty half) and each half must clear
    // <split_conf_min>. Returns the composed value (or -1 if either half fails).
    int recognize_two_digits_ex(cv::Mat const& cell, int count, int upscale,
                                double split_conf_min,
                                double& conf_l, double& conf_r) const;

    // Normalizes a raw cell image into the 1x1x28x28 whitened input blob the
    // MNIST model expects. White digit on black, normalized (x/255 - 0.1307)
    // / 0.3081. Returns false if the cell has no reliable foreground.
    static bool prepare_input(cv::Mat const& cell, cv::Mat& out_blob);

private:
    mutable cv::dnn::Net net_;
    mutable cv::dnn::Net counter_net_;
};

}
