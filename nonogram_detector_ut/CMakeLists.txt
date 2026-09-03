set(SOURCES
	"main.cpp"
	"digit_recognizer_test.cpp")
if(NG_ENABLE_SOLVER)
	list(APPEND SOURCES "solver_test.cpp")
endif()

add_executable(nonogram_detector_ut ${SOURCES})
target_link_libraries(nonogram_detector_ut nonogram_detector)
if(NG_ENABLE_SOLVER)
	target_link_libraries(nonogram_detector_ut nonogram_solver)
endif()
target_compile_definitions(nonogram_detector_ut PRIVATE
	NG_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../nonogram_detector/models/digits.onnx")
if(NG_ENABLE_SOLVER)
	target_compile_definitions(nonogram_detector_ut PRIVATE NG_ENABLE_SOLVER=1)
endif()
set_target_properties(nonogram_detector_ut PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
