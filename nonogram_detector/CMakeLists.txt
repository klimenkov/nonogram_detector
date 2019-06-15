set(HEADERS
	"include/image_operations.hpp"
    "include/masks.hpp"
	"include/point_compare.hpp")

set(SOURCES
	"src/image_operations.cpp"
    "src/masks.cpp"
	"src/point_compare.cpp")

add_library(nonogram_detector ${HEADERS} ${SOURCES})
target_include_directories(nonogram_detector PUBLIC include)

find_package(OpenCV REQUIRED)
target_link_libraries(nonogram_detector ${OpenCV_LIBS})
