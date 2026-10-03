cmake_minimum_required(VERSION 3.14)
project(moeengine_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PALM_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../vendor/Palm-Infra")
add_library(moeengine_optimizer STATIC
    "${PALM_SOURCE}/kernels/moeengine_optimizer.cpp")
target_include_directories(moeengine_optimizer PUBLIC "${PALM_SOURCE}")

enable_testing()
add_executable(test_moeengine_optimizer
    "${PALM_SOURCE}/tests/test_moeengine_optimizer.cpp")
target_link_libraries(test_moeengine_optimizer PRIVATE moeengine_optimizer)
add_test(NAME moeengine_optimizer COMMAND test_moeengine_optimizer)
