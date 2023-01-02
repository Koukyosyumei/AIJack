cmake_minimum_required(VERSION 3.13...3.18)
project(aijack_cpp_core VERSION 0.0.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native -march=native")

find_package(Threads REQUIRED)
find_package(Boost REQUIRED)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
pybind11_add_module(aijack_cpp_core src/main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_link_libraries(aijack_cpp_core PRIVATE boost_random)
