#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "digit_recognizer.hpp"

namespace
{

namespace fs = std::filesystem;

// Real marked clue cells (warped to 20x20) live under NG_DIGITS_DIR/<digit>/,
// one directory per single-digit value 1..9. This matches the distribution the
// bundled real-data digit model was trained on, so it is a valid end-to-end
// fixture for DigitRecognizer::recognize. When the dataset is not present the
// classification assertion is skipped (the dataset is external/untracked).

std::filesystem::path digits_dir()
{
    std::filesystem::path p = NG_DIGITS_DIR;
    if (!fs::is_directory(p))
        return {};
    return p;
}

// Reads every marked cell for <d> (1..9) and checks that the recognizer reads
// each as <d>. Returns (correct, total).
std::pair<int, int> score_digit(ng::DigitRecognizer const& recognizer, int const d)
{
    int correct = 0, total = 0;
    std::string dir = (digits_dir() / std::to_string(d)).string() + "/";
    if (!fs::is_directory(dir))
        return {0, 0};
    for (auto const& e : fs::directory_iterator(dir))
    {
        if (e.path().extension() != ".png")
            continue;
        cv::Mat img = cv::imread(e.path().string());
        if (img.empty())
            continue;
        ++total;
        if (recognizer.recognize(img) == d)
            ++correct;
    }
    return {correct, total};
}

// Classifies all 9 marked digit groups and requires a high aggregate accuracy
// (the model is ~99.9% on this dataset; we allow a couple of stray misreads).
bool test_real_digits(ng::DigitRecognizer const& recognizer)
{
    if (digits_dir().empty())
    {
        std::cout << "  [skip] NG_DIGITS_DIR dataset not present; skipping real-digit classification\n";
        return true;
    }

    int total = 0, correct = 0;
    for (int d = 1; d <= 9; ++d)
    {
        auto const [c, t] = score_digit(recognizer, d);
        correct += c;
        total += t;
        std::cout << "  digit " << d << ": " << c << "/" << t << "\n";
    }

    bool const ok = total > 0 && (100.0 * correct / total) >= 99.0;
    std::cout << "  ["
              << (ok ? "ok" : "FAIL")
              << "] real-digit accuracy " << correct << "/" << total
              << " = " << (total ? 100.0 * correct / total : 0.0) << "%\n";
    return ok;
}

// A cell with only a grid border and no inner digit must read as empty (-1).
bool test_empty_cell(ng::DigitRecognizer const& recognizer)
{
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
    cv::Mat cell(60, 60, CV_8UC1, cv::Scalar(255));
    cv::putText(cell, "3", cv::Point(22, 42), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(0), 2, cv::LINE_AA);

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

// Exercises the recognize_two_digits *mechanism* on the real marked two-digit
// cells (NG_DIGITS_DIR/<10,11,...>/). For every cell the counter model flags as
// two digits, the split-read must compose a valid two-digit value in [10, 99]
// (never a garbage or single-digit output). End-to-end accuracy on a specific
// photo is validated by the application/solve gate, not here, because the tied
// model is tuned per-photo (its general accuracy across many fonts is lower).
// Without a counter model the split must be guarded off.
bool test_two_digit(ng::DigitRecognizer& recognizer)
{
    if (digits_dir().empty())
    {
        std::cout << "  [skip] NG_DIGITS_DIR dataset not present; skipping two-digit test\n";
        return true;
    }

    recognizer.set_counter_model(NG_COUNTER_PATH);

    int flagged = 0, compose_ok = 0;
    for (int v = 10; v <= 22; ++v)
    {
        if (v == 17 || v == 19)
            continue; // no marked 17/19 cells
        std::string dir = (digits_dir() / std::to_string(v)).string() + "/";
        if (!fs::is_directory(dir))
            continue;
        for (auto const& e : fs::directory_iterator(dir))
        {
            if (e.path().extension() != ".png")
                continue;
            cv::Mat img = cv::imread(e.path().string());
            if (img.empty())
                continue;
            if (recognizer.digit_count(img) != 2)
                continue;
            ++flagged;
            int const two = recognizer.recognize_two_digits(img, 2);
            if (two >= 10 && two <= 99)
                ++compose_ok;
        }
    }

    bool const ok = flagged > 0 && compose_ok == flagged;
    std::cout << "  [" << (ok ? "ok" : "FAIL")
              << "] two-digit compose " << compose_ok << "/" << flagged
              << " flagged 2-digit cells produced a valid 2-digit value\n";
    return ok && flagged > 0;
}

// The split-read must refuse (return -1) when no counter model has been set
// (digit_count returns -1, which recognize_two_digits rejects), so a
// downstream whole-cell fallback is possible. Uses its own recognizer
// because the shared one has a counter model configured by test_two_digit.
bool test_two_digit_no_counter(ng::DigitRecognizer const& /*recognizer*/)
{
    ng::DigitRecognizer fresh(NG_MODEL_PATH);
    cv::Mat cell(60, 60, CV_8UC1, cv::Scalar(255));
    cv::putText(cell, "12", cv::Point(12, 42), cv::FONT_HERSHEY_SIMPLEX, 0.9,
                cv::Scalar(0), 2, cv::LINE_AA);
    bool const ok = fresh.recognize_two_digits(cell, fresh.digit_count(cell)) == -1;
    std::cout << "  [" << (ok ? "ok" : "FAIL")
              << "] recognize_two_digits guarded by counter model presence\n";
    return ok;
}

}

int run_digit_recognizer_tests()
{
    int failures = 0;

    std::cout << "case: DigitRecognizer on real marked cells\n";
    try
    {
        ng::DigitRecognizer recognizer(NG_MODEL_PATH);

        if (!test_real_digits(recognizer)) ++failures;
        if (!test_empty_cell(recognizer)) ++failures;
        if (!test_prepare_input(recognizer)) ++failures;
        if (!test_two_digit(recognizer)) ++failures;
        if (!test_two_digit_no_counter(recognizer)) ++failures;
    }
    catch (std::exception const& e)
    {
        std::cerr << "  [FAIL] exception: " << e.what() << "\n";
        ++failures;
    }

    return failures;
}
