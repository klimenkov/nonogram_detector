add_library(nonogram_solver
	include/solver.hpp
	src/solver.cpp)

target_include_directories(nonogram_solver PUBLIC include)
target_link_libraries(nonogram_solver PUBLIC picross::picross)

set_target_properties(nonogram_solver PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
