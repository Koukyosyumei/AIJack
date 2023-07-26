cmake_minimum_required(VERSION 3.13)

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

add_library(tupledata STATIC ../src/storage/data.pb.cc)

add_executable(aivalut ../src/main.cpp)

target_link_libraries(aivalut pthread tupledata ${PROTOBUF_LIBRARY} libaivalut)
