#include <cstring>
#include <exception>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "digit_recognizer.hpp"

namespace
{

// Builds a synthetic nonogram clue cell: a white square with a thin black grid
// frame along the border (as in a real cell) and a single dark digit rendered
// in the middle. <cell> is the side length in pixels; <label> is one of "0".."9".
cv::Mat make_digit_cell(int const cell, char const* label)
{
    cv::Mat img(cell, cell, CV_8UC1, cv::Scalar(255));
    cv::rectangle(img, cv::Rect(0, 0, cell, cell), cv::Scalar(0), 2);
    double const font_scale = cell * 0.6 / 40.0;
    int baseline = 0;
    cv::Size text = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, 2, &baseline);
    cv::Point origin((cell - text.width) / 2, (cell + text.height) / 2);
    cv::putText(img, label, origin, cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0), 2, cv::LINE_AA);
    return img;
}

bool test_all_digits(ng::DigitRecognizer const& recognizer)
{
    bool ok = true;
    for (int d = 0; d <= 9; ++d)
    {
        char label[2] = { static_cast<char>('0' + d), '\0' };
        auto const cell = make_digit_cell(60, label);
        int const got = recognizer.recognize(cell);
        if (got != d)
        {
            std::cerr << "  [FAIL] recognizer('" << label << "') = " << got
                      << ", expected " << d << "\n";
            ok = false;
        }
    }
    if (ok)
        std::cout << "  [ok] recognizer classified all digits 0-9 correctly\n";
    return ok;
}

bool test_empty_cell(ng::DigitRecognizer const& recognizer)
{
    // A cell with only its border frame and no digit must read as empty (-1).
    cv::Mat empty_cell(60, 60, CV_8UC1, cv::Scalar(255));
    cv::rectangle(empty_cell, cv::Rect(0, 0, 60, 60), cv::Scalar(0), 2);

    int const got = recognizer.recognize(empty_cell);
    if (got != -1)
    {
        std::cerr << "  [FAIL] empty cell recognized as " << got << ", expected -1\n";
        return false;
    }
    std::cout << "  [ok] empty cell read as empty (-1)\n";
    return true;
}

bool test_prepare_input(ng::DigitRecognizer const&)
{
    auto const cell = make_digit_cell(60, "3");
    cv::Mat blob;
    if (!ng::DigitRecognizer::prepare_input(cell, blob))
    {
        std::cerr << "  [FAIL] prepare_input returned false on a digit cell\n";
        return false;
    }
    if (blob.dims != 4 || blob.size[0] != 1 || blob.size[1] != 1 ||
        blob.size[2] != 28 || blob.size[3] != 28)
    {
        std::cerr << "  [FAIL] prepare_input blob shape wrong (dims=" << blob.dims << ")\n";
        return false;
    }
    std::cout << "  [ok] prepare_input produced a 1x1x28x28 blob\n";
    return true;
}

}

int run_digit_recognizer_tests()
{
    int failures = 0;

    std::cout << "case: DigitRecognizer on synthetic cells\n";
    try
    {
        ng::DigitRecognizer recognizer(NG_MODEL_PATH);

        if (!test_all_digits(recognizer)) ++failures;
        if (!test_empty_cell(recognizer)) ++failures;
        if (!test_prepare_input(recognizer)) ++failures;
    }
    catch (std::exception const& e)
    {
        std::cerr << "  [FAIL] exception: " << e.what() << "\n";
        ++failures;
    }

    return failures;
}
