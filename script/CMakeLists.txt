cmake_minimum_required(VERSION 3.13)

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

add_library(tupledata STATIC ../src/storage/data.pb.cc)

add_executable(mydbapp ../src/main.cpp)

target_link_libraries(mydbapp pthread tupledata ${PROTOBUF_LIBRARY} libmydb)
