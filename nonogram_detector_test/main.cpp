#include <chrono>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "cross_locs_detector.hpp"


class WindowTrackbarDetector
{
public:
    WindowTrackbarDetector(
        std::string const& window_name,
        cv::Mat const& image)
        : m_window_name(window_name)
        , m_image(image)
        , m_resize_width_height_max(1000)
        , m_threshold_block_size(3)
        , m_threshold_c(0.0)
    {
    }

    void show()
    {
        cv::namedWindow(m_window_name);

        int trackbar_position_resize_width_height_max = 50;
        int trackbar_position_block_size = 0;
        int trackbar_position_c = 50;

        cv::createTrackbar(
            "Resize width (height) max",
            m_window_name,
            &trackbar_position_resize_width_height_max,
            M_TRACKBAR_POSITION_MAX,
            on_change_resize_width_height_max,
            this);

        cv::createTrackbar(
            "Block size",
            m_window_name,
            &trackbar_position_block_size,
            M_TRACKBAR_POSITION_MAX,
            on_change_block_size,
            this);

        cv::createTrackbar(
            "C",
            m_window_name,
            &trackbar_position_c,
            M_TRACKBAR_POSITION_MAX,
            on_change_c,
            this);

        draw();

        cv::waitKey();
    }

    static void on_change_resize_width_height_max(int trackbar_position, void* object_p)
    {
        auto& object = *static_cast<WindowTrackbarDetector*>(object_p);

        auto const resize_width_height_max = 10 * trackbar_position + 500;
        object.m_resize_width_height_max = resize_width_height_max;

        object.draw();
    }

    static void on_change_block_size(int trackbar_position, void* object_p)
    {
        auto& object = *static_cast<WindowTrackbarDetector*>(object_p);

        auto const block_size = 2 * trackbar_position + 3;
        object.m_threshold_block_size = block_size;

        object.draw();
    }

    static void on_change_c(int trackbar_position, void* object_p)
    {
        auto& object = *static_cast<WindowTrackbarDetector*>(object_p);

        auto const c = trackbar_position - 50.0;
        object.m_threshold_c = c;

        object.draw();
    }

    void draw()
    {
        ng::CrossLocsDetector cross_loc_detector(
            m_resize_width_height_max,
            m_threshold_block_size,
            m_threshold_c,
            5,
            50,
            0.9);

        bool cell_loc_found;
        cv::Mat cross_locs_main;
        cv::Mat cross_locs_top;
        cv::Mat cross_locs_left;
        std::tie(cell_loc_found, cross_locs_main, cross_locs_top, cross_locs_left) =
            cross_loc_detector.detect(m_image);

        int const radius = 2;
        auto image_draw = ng::CrossLocsDetector::draw(m_image, cross_locs_main, radius, cv::Scalar(255, 0, 0));
        image_draw = ng::CrossLocsDetector::draw(image_draw, cross_locs_top, radius, cv::Scalar(0, 255, 0));
        image_draw = ng::CrossLocsDetector::draw(image_draw, cross_locs_left, radius, cv::Scalar(0, 0, 255));

        cv::imshow(m_window_name, image_draw);
    }

private:
    static int const M_TRACKBAR_POSITION_MAX = 100;

    std::string m_window_name;
    cv::Mat m_image;

    float m_resize_width_height_max;
    int m_threshold_block_size;
    double m_threshold_c;
};


int main()
{
    std::string const image_path =
        R"(C:\Users\klimenkov\Desktop\nonograms\nonogram.jpg)";

    auto image = cv::imread(image_path);

    WindowTrackbarDetector window_trackbar_detector("My window", image);
    window_trackbar_detector.show();

    return 0;
}
