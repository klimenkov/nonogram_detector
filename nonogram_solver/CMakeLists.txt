add_library(nonogram_solver
	include/solver.hpp
	src/solver.cpp
	include/clue_corrector.hpp
	src/clue_corrector.cpp
	include/non_file.hpp
	src/non_file.cpp)

target_include_directories(nonogram_solver PUBLIC include)

find_package(OpenCV REQUIRED COMPONENTS core)
target_include_directories(nonogram_solver PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(nonogram_solver PUBLIC picross::picross opencv_core)

set_target_properties(nonogram_solver PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
