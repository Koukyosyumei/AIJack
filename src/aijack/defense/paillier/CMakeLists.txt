cmake_minimum_required(VERSION 3.13...3.18)
project(aijack_paillier VERSION 0.0.0)

set(Boost_DEBUG 1)

find_package(Threads REQUIRED)

#find_package(boost REQUIRED)
add_subdirectory(boost)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG)
pybind11_add_module(aijack_paillier src/main.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_link_libraries(aijack_paillier PRIVATE boost_math boost_random boost_multiprecision)
