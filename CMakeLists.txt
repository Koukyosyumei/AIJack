cmake_minimum_required(VERSION 3.13)
project("libmyplan" LANGUAGES C CXX)

# find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native -march=native")

set(SOURCE_DIR  "mydb")
set(SCRIPT_DIR "script")
set(TEST_DIR  "test")

include(GNUInstallDirs)

include_directories("${PROJECT_SOURCE_DIR}/src")

add_subdirectory(${SOURCE_DIR})
add_subdirectory(${SCRIPT_DIR})

enable_testing()
add_subdirectory(${TEST_DIR})
