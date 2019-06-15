set(SOURCES
	"main.cpp")

add_executable(nonogram_detector_application ${SOURCES})
target_link_libraries(nonogram_detector_application nonogram_detector)
