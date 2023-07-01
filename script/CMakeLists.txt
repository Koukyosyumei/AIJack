cmake_minimum_required(VERSION 3.13)

add_executable(mydb main.cpp)

target_link_libraries(mydb pthread libmydb)
