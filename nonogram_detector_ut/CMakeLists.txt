set(SOURCES
	"main.cpp")

add_executable(nonogram_detector_ut ${SOURCES})
target_link_libraries(nonogram_detector_ut nonogram_detector)
set_target_properties(nonogram_detector_ut PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
