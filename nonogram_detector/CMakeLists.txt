set(HEADERS
	"include/detection.hpp"
	"include/image_operations.hpp"
	"include/cross_locs_detector.hpp"
	"include/masks.hpp"
	"include/point_compare.hpp"
	"include/digit_recognizer.hpp"
	"include/decode.hpp"
	"include/grid_smooth_fit.hpp")

set(SOURCES
	"src/image_operations.cpp"
	"src/cross_locs_detector.cpp"
	"src/masks.cpp"
	"src/point_compare.cpp"
	"src/digit_recognizer.cpp"
	"src/decode.cpp"
	"src/grid_smooth_fit.cpp")

add_library(nonogram_detector ${HEADERS} ${SOURCES})
target_include_directories(nonogram_detector PUBLIC include)

set_target_properties(nonogram_detector PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

set(OPENCV_COMPONENTS core imgproc imgcodecs dnn)
find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_COMPONENTS})

target_include_directories(nonogram_detector PUBLIC ${OpenCV_INCLUDE_DIRS})
foreach(module ${OPENCV_COMPONENTS})
	target_link_libraries(nonogram_detector PUBLIC opencv_${module})
endforeach()
