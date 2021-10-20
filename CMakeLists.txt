cmake_minimum_required(VERSION 3.18)
project(clifford VERSION 0.1.0
DESCRIPTION "Clifford Attractors. Contact orbita@orbital.me.uk"
LANGUAGES CXX)

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(src)
