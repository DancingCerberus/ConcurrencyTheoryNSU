cmake_minimum_required(VERSION 3.28)
project(2_3)

set(CMAKE_CXX_STANDARD 20)

add_executable(2_3
        section_for_each.cpp
        one_section.cpp)

set(CMAKE_CXX_FLAGS "-fopenmp -O2")

add_executable(one_section one_section.cpp)

add_executable(static section_for_each.cpp)
add_executable(dynamic section_for_each.cpp)
target_compile_definitions(dynamic PRIVATE SCHEDULE_DYNAMIC)
add_executable(guided section_for_each.cpp)
target_compile_definitions(guided PRIVATE SCHEDULE_GUIDED)
add_executable(auto section_for_each.cpp)
target_compile_definitions(auto PRIVATE SCHEDULE_AUTO)