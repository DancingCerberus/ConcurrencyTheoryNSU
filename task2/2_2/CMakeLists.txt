cmake_minimum_required(VERSION 3.22.1)
project(2_2)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP)

add_executable(2_2 main.cpp)

if (OpenMP_CXX_FOUND)
    target_link_libraries(2_2 PUBLIC OpenMP::OpenMP_CXX)
endif ()