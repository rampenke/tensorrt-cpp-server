cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

project(tensrort-cpp-server)
#include(cmake/ccache.cmake)

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.4)
find_package(CUDA REQUIRED)

add_executable(tensorrt_cpp_server src/server.cpp src/mnist.cpp)

target_include_directories(tensorrt_cpp_server PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(tensorrt_cpp_server PUBLIC ${CUDA_LIBRARIES})


target_link_libraries(tensorrt_cpp_server PUBLIC nvonnxparser nvinfer pthread)