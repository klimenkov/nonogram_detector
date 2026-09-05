set(SOURCES
	"main.cpp")

add_executable(nonogram_detector_application ${SOURCES})
target_link_libraries(nonogram_detector_application nonogram_detector)
if(NG_ENABLE_SOLVER)
	target_link_libraries(nonogram_detector_application nonogram_solver)
	target_compile_definitions(nonogram_detector_application PRIVATE NG_ENABLE_SOLVER=1)
endif()
set_target_properties(nonogram_detector_application PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

# Ensure the bundled digit-recognition models sit next to the executable so the
# default model-path resolution works regardless of the working directory.
add_custom_command(TARGET nonogram_detector_application POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory
		"$<TARGET_FILE_DIR:nonogram_detector_application>/models"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${CMAKE_CURRENT_SOURCE_DIR}/../nonogram_detector/models/digits.onnx"
		"$<TARGET_FILE_DIR:nonogram_detector_application>/models/digits.onnx"
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		"${CMAKE_CURRENT_SOURCE_DIR}/../nonogram_detector/models/digits_counter.onnx"
		"$<TARGET_FILE_DIR:nonogram_detector_application>/models/digits_counter.onnx")
