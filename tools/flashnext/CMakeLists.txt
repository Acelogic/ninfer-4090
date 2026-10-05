# Flash-Next developer tools: the generator/benchmark, the FP32 reference generator and the kernel tests.
# Each runs against the real model files; see docs/maintainer/flash-next-engine.md.
set(NINFER_FLASHNEXT_TOOLS
  fn_generate:fn_generate.cpp
  fn_vision:fn_vision.cpp
  ref_generate:ref_generate.cpp
  test_cpu_experts:test_cpu_experts.cpp
  test_gpu_gemv:test_gpu_gemv.cu
  test_gpu_experts:test_gpu_experts.cu
  test_gpu_experts_batch:test_gpu_experts_batch.cu
  test_expert_stream:test_expert_stream.cu
  test_gemm_tc:test_gemm_tc.cu
  test_qsa:test_qsa.cu)
foreach(entry IN LISTS NINFER_FLASHNEXT_TOOLS)
  string(REPLACE ":" ";" parts "${entry}")
  list(GET parts 0 tool)
  list(GET parts 1 source)
  add_executable(${tool} ${source})
  target_link_libraries(${tool} PRIVATE ninfer_flashnext)
  if(MSVC)
    target_compile_options(${tool} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX512>)
    set_target_properties(${tool} PROPERTIES CUDA_RUNTIME_LIBRARY Static)
  endif()
  set_target_properties(${tool} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/flashnext)
  ninfer_flashnext_runtime(${tool})
endforeach()
