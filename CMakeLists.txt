cmake_minimum_required(VERSION 3.14)

project(nonogram_detector LANGUAGES CXX)

# Optional integration of the third-party picross-solver library, used to solve
# the decoded nonogram clues. Disable to drop the configure-time network fetch.
option(NG_ENABLE_SOLVER "Integrate the third-party picross solver" ON)

if(NG_ENABLE_SOLVER)
	# pierre-dejoue/picross-solver (MIT): C++17 nonogram solver library. Only the
	# dependency-free core `picross::picross` target is used; the app/CLI/test
	# targets shipped with the library are all disabled.
	#
	# Pinned to `main`: the released 0.3.0 tag fails to link on GCC because its
	# templated OutputGrid::get_line<Type> is never explicitly instantiated
	# (unresolved symbols). `main` carries the fix; pinned to a concrete commit
	# (current main) for reproducibility.
	include(FetchContent)
	FetchContent_Declare(
		picross_solver
		GIT_REPOSITORY https://github.com/pierre-dejoue/picross-solver.git
		GIT_TAG        a24d940b6351e83e390de6ab300acef0e769acef
		GIT_SHALLOW    TRUE)
	set(PICROSS_BUILD_APP OFF CACHE BOOL "" FORCE)
	set(PICROSS_BUILD_CLI OFF CACHE BOOL "" FORCE)
	set(PICROSS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
	set(PICROSS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(picross_solver)

	# picross-solver compiles itself with -Werror, which fails against newer
	# GCC toolchains on a false-positive null-dereference warning. Strip the
	# -Werror flag from the third-party target (keeping its other warnings) so
	# the vendored library builds without us modifying its source.
	get_target_property(_picross_opts picross COMPILE_OPTIONS)
	if(_picross_opts)
		list(FILTER _picross_opts EXCLUDE REGEX "^-Werror$")
		set_target_properties(picross PROPERTIES COMPILE_OPTIONS "${_picross_opts}")
	endif()
	add_subdirectory(nonogram_solver)
endif()

add_subdirectory(nonogram_detector)
add_subdirectory(nonogram_detector_application)
add_subdirectory(nonogram_detector_ut)
