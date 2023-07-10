cmake_minimum_required(VERSION 3.13)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.0.zip
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})
# add_library(tupledata STATIC ../mydb/storage/data.pb.cc)

set(TEST_NAME aivalut-test)

file(GLOB testSRC
    "*.cpp"
)
set(TEST_RUNNER main.cpp ${testSRC})
add_executable(${TEST_NAME} ${TEST_RUNNER})

target_link_libraries(${TEST_NAME} gtest gtest_main pthread tupledata ${PROTOBUF_LIBRARY} libaivalut)

gtest_discover_tests(${TEST_NAME})
