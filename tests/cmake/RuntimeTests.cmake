ninfer_add_test(ninfer_admission_policy_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_admission_policy.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_context_cost_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_context_cost.cpp"
  LIBRARIES ninfer_runtime_support ninfer::json)

ninfer_add_test(ninfer_resource_manager_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_resource_manager.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_kv_capacity_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_kv_capacity.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_sampling_defaults_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_sampling_defaults.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_host_sampler_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_sampler.cpp"
  LIBRARIES ninfer_runtime_support)

if(NINFER_WITH_FLASHNEXT)
  # Needs NINFER_FLASHNEXT_GGUF and NINFER_FLASHNEXT_ORACLE; skips otherwise.
  ninfer_add_test(ninfer_flashnext_frontend_real_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_flashnext_frontend_real.cpp"
    LIBRARIES ninfer_engine ninfer_model_runtime ninfer_flashnext)
  set_tests_properties(ninfer_flashnext_frontend_real_test PROPERTIES SKIP_RETURN_CODE 77)
endif()
