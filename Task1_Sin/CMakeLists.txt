cmake_minimum_required(VERSION 3.27)
project(Task1_Sin)

set(CMAKE_CXX_STANDARD 17)

add_executable(double main.cpp)
add_executable(float main.cpp)

target_compile_definitions(float PUBLIC FLOAT)
