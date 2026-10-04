add_library(ninfer_runtime_support STATIC
  engine/admission_policy.cpp
  engine/host_sampler.cpp
  engine/context_cache/context_cost.cpp
  engine/context_cache/context_cost_defaults.cpp
  engine/kv_capacity.cpp
  engine/public_types.cpp
)
ninfer_internal_includes(ninfer_runtime_support)
target_link_libraries(ninfer_runtime_support PUBLIC ninfer_core PRIVATE ninfer::json)

# Public Engine and common contracts; model execution remains in the model layer.
add_library(ninfer_engine STATIC
  engine/model_instance.cpp
  contract/sampling.cpp
  engine/engine.cpp
)
ninfer_internal_includes(ninfer_engine)
target_include_directories(ninfer_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ninfer_engine PRIVATE
  ninfer_model_runtime
  ninfer_artifact
  ninfer_core
  ninfer_runtime_support
  ninfer_ops
  ninfer_text
  ninfer_media_decode)
if(NINFER_WITH_FLASHNEXT)
  target_sources(ninfer_engine PRIVATE
    engine/flashnext_core.cpp
    engine/flashnext_frontend.cpp)
  target_compile_definitions(ninfer_engine PRIVATE NINFER_WITH_FLASHNEXT=1)
  target_link_libraries(ninfer_engine PRIVATE ninfer_flashnext ninfer::json)
endif()
add_library(ninfer::engine ALIAS ninfer_engine)
