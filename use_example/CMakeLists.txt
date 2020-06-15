cmake_minimum_required(VERSION 3.16)
project(use_example)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -msse4.1)

add_executable(use_example main.cpp NN.cpp NN.h Matrix.cpp Matrix.h)