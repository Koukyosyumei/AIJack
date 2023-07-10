cmake_minimum_required(VERSION 3.13)

file(GLOB projectSRC "*/*.h")

add_library(libaivalut INTERFACE)

message(STATUS "CMake inatall directory: " ${CMAKE_INSTALL_INCLUDEDIR})

target_include_directories(libaivalut
    INTERFACE 
        $<BUILD_INTERFACE:${PROJECT_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
