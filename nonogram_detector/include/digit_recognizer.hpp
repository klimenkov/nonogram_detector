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

    // Returns the recognized digit in [0, 9], or -1 if the cell appears empty
    // (no foreground) or the top class confidence is below <confidence_min>.
    int recognize(cv::Mat const& cell, double confidence_min = 0.0) const;

    // Like recognize(), but also reports the softmax probability of the chosen
    // class in <confidence> (set to 0 on an empty/unreliable cell). Useful for
    // calibrating a confidence_min threshold on a corpus of cells.
    int recognize_ex(cv::Mat const& cell, double& confidence) const;

    // Normalizes a raw cell image into the 1x1x28x28 whitened input blob the
    // MNIST model expects. White digit on black, normalized (x/255 - 0.1307)
    // / 0.3081. Returns false if the cell has no reliable foreground.
    static bool prepare_input(cv::Mat const& cell, cv::Mat& out_blob);

private:
    mutable cv::dnn::Net net_;
};

}
