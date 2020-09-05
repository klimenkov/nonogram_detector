set(SOURCES
	"main.cpp")

add_executable(nonogram_detector_test ${SOURCES})
target_link_libraries(nonogram_detector_test nonogram_detector)
