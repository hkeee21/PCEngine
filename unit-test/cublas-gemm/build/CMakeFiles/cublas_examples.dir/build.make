# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hongke21/nfs/code/spconv/hashgemm-v4/unit-test/cublas-gemm

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hongke21/nfs/code/spconv/hashgemm-v4/unit-test/cublas-gemm/build

# Utility rule file for cublas_examples.

# Include the progress variables for this target.
include CMakeFiles/cublas_examples.dir/progress.make

cublas_examples: CMakeFiles/cublas_examples.dir/build.make

.PHONY : cublas_examples

# Rule to build all files generated by this target.
CMakeFiles/cublas_examples.dir/build: cublas_examples

.PHONY : CMakeFiles/cublas_examples.dir/build

CMakeFiles/cublas_examples.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cublas_examples.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cublas_examples.dir/clean

CMakeFiles/cublas_examples.dir/depend:
	cd /home/hongke21/nfs/code/spconv/hashgemm-v4/unit-test/cublas-gemm/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hongke21/nfs/code/spconv/hashgemm-v4/unit-test/cublas-gemm /home/hongke21/nfs/code/spconv/hashgemm-v4/unit-test/cublas-gemm /home/hongke21/nfs/code/spconv/hashgemm-v4/unit-test/cublas-gemm/build /home/hongke21/nfs/code/spconv/hashgemm-v4/unit-test/cublas-gemm/build /home/hongke21/nfs/code/spconv/hashgemm-v4/unit-test/cublas-gemm/build/CMakeFiles/cublas_examples.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cublas_examples.dir/depend

