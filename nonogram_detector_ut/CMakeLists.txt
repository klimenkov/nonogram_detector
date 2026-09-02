set(SOURCES
	"main.cpp"
	"digit_recognizer_test.cpp")

add_executable(nonogram_detector_ut ${SOURCES})
target_link_libraries(nonogram_detector_ut nonogram_detector)
target_compile_definitions(nonogram_detector_ut PRIVATE
	NG_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../nonogram_detector/models/digits.onnx")
set_target_properties(nonogram_detector_ut PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
