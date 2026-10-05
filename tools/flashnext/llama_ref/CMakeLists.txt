# vision_ref: llama.cpp's side of the Flash-Next vision checks (see vision_ref.cpp). A separate project, built
# against a llama.cpp checkout on the CPU (static, no GPU), never part of the NInfer build:
#
#   cmake -S tools/flashnext/llama_ref -B build-vision-ref -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=<llama.cpp>
#   cmake --build build-vision-ref --target vision_ref
#
# Adapted from Strata's tools/vision/CMakeLists.txt (MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors).
cmake_minimum_required(VERSION 3.21)
project(flashnext_vision_ref C CXX)

set(LLAMA_DIR "" CACHE PATH "llama.cpp checkout")
if(NOT LLAMA_DIR)
  message(FATAL_ERROR "set LLAMA_DIR to a llama.cpp checkout")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD ON CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ON CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_DIR} llama EXCLUDE_FROM_ALL)

add_executable(vision_ref vision_ref.cpp)
target_link_libraries(vision_ref PRIVATE mtmd llama)
target_compile_features(vision_ref PRIVATE cxx_std_17)
set_target_properties(vision_ref PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
