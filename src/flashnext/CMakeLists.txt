# Flash-Next engine for Qwen3.8-Flash-Next GGUF models (architecture qwen4exp): GPU dense path with
# FP32 activations and cuBLAS GEMMs for prompts, routed experts on the AVX-512 CPU and a VRAM cache.
add_library(ninfer_flashnext STATIC
  gguf.cpp
  shards.cpp
  quants.cpp
  cpu_experts.cpp
  reference.cpp
  engine.cpp
  prefill.cpp
  cuda/gemv.cu
  cuda/ops.cu
  cuda/experts.cu
  cuda/gemm.cu
  cuda/gemm_tc.cu
  cuda/qsa.cu
  cuda/experts_batch.cu
  cuda/expert_stream.cu)
target_include_directories(ninfer_flashnext PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ninfer_flashnext PUBLIC CUDA::cublas ${NINFER_CUDART_TARGET} Threads::Threads)
ninfer_cuda_non_rdc_archive(ninfer_flashnext)
if(WIN32)
  set_target_properties(ninfer_flashnext PROPERTIES CUDA_RUNTIME_LIBRARY Static)
endif()
# The host kernels are written for AVX-512; the CUDA sources get no host ISA flags.
if(MSVC)
  target_compile_options(ninfer_flashnext PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
else()
  target_compile_options(ninfer_flashnext PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-march=x86-64-v4 -mavx512vnni -mf16c>)
endif()

# Processes that link the engine import cuBLAS but call it only once a Flash-Next model is loaded,
# so the DLL is delay-loaded on Windows: executables that never serve Flash-Next start without it.
if(WIN32)
  set(NINFER_CUBLAS_DLL "cublas64_${CUDAToolkit_VERSION_MAJOR}.dll")
  target_link_options(ninfer_flashnext INTERFACE "/DELAYLOAD:${NINFER_CUBLAS_DLL}")
  target_link_libraries(ninfer_flashnext INTERFACE delayimp)
  file(GLOB NINFER_CUBLAS_RUNTIME
    "${CUDAToolkit_BIN_DIR}/cublas64_*.dll" "${CUDAToolkit_BIN_DIR}/cublasLt64_*.dll"
    "${CUDAToolkit_BIN_DIR}/x64/cublas64_*.dll" "${CUDAToolkit_BIN_DIR}/x64/cublasLt64_*.dll")
  if(NOT NINFER_CUBLAS_RUNTIME)
    message(FATAL_ERROR "Flash-Next needs the cuBLAS DLLs; none found under ${CUDAToolkit_BIN_DIR}")
  endif()
  set(NINFER_CUBLAS_RUNTIME "${NINFER_CUBLAS_RUNTIME}" CACHE INTERNAL "cuBLAS DLLs shipped with Flash-Next")
endif()

# Copies the cuBLAS DLLs next to an executable that serves Flash-Next models.
function(ninfer_flashnext_runtime target)
  if(WIN32)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${NINFER_CUBLAS_RUNTIME} $<TARGET_FILE_DIR:${target}>
      VERBATIM)
  endif()
endfunction()
