if(EXISTS "/home/usr/Documents/TelAvivU/build-term/run_unit_tests")
  if(NOT EXISTS "/home/usr/Documents/TelAvivU/build-term/run_unit_tests[1]_tests.cmake" OR
     NOT "/home/usr/Documents/TelAvivU/build-term/run_unit_tests[1]_tests.cmake" IS_NEWER_THAN "/home/usr/Documents/TelAvivU/build-term/run_unit_tests" OR
     NOT "/home/usr/Documents/TelAvivU/build-term/run_unit_tests[1]_tests.cmake" IS_NEWER_THAN "${CMAKE_CURRENT_LIST_FILE}")
    include("/usr/share/cmake-3.28/Modules/GoogleTestAddTests.cmake")
    gtest_discover_tests_impl(
      TEST_EXECUTABLE [==[/home/usr/Documents/TelAvivU/build-term/run_unit_tests]==]
      TEST_EXECUTOR [==[]==]
      TEST_WORKING_DIR [==[/home/usr/Documents/TelAvivU/build-term]==]
      TEST_EXTRA_ARGS [==[]==]
      TEST_PROPERTIES [==[ENVIRONMENT;SHOW_LORENZ_PORTRAIT=0;LORENZ_WINDOW_SECONDS=1;LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/tcm/1.5/lib:/opt/intel/oneapi/umf/1.1/lib:/opt/intel/oneapi/tcm/1.5/env/../lib:/opt/intel/oneapi/tbb/2023.0/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/mpi/2021.18/opt/mpi/libfabric/lib:/opt/intel/oneapi/mpi/2021.18/lib:/opt/intel/oneapi/mkl/2026.0/lib:/opt/intel/oneapi/ippcp/2026.0/lib/:/opt/intel/oneapi/ipp/2026.0/lib:/opt/intel/oneapi/dnnl/2026.0/lib:/opt/intel/oneapi/debugger/2026.0/opt/debugger/lib:/opt/intel/oneapi/compiler/2026.0/opt/compiler/lib:/opt/intel/oneapi/compiler/2026.0/lib:/opt/intel/oneapi/ccl/2022.0/lib/:/usr/local/cuda/lib64:/usr/local/cuda/lib64;LABELS;unit]==]
      TEST_PREFIX [==[]==]
      TEST_SUFFIX [==[]==]
      TEST_FILTER [==[]==]
      NO_PRETTY_TYPES [==[FALSE]==]
      NO_PRETTY_VALUES [==[FALSE]==]
      TEST_LIST [==[run_unit_tests_TESTS]==]
      CTEST_FILE [==[/home/usr/Documents/TelAvivU/build-term/run_unit_tests[1]_tests.cmake]==]
      TEST_DISCOVERY_TIMEOUT [==[5]==]
      TEST_XML_OUTPUT_DIR [==[]==]
    )
  endif()
  include("/home/usr/Documents/TelAvivU/build-term/run_unit_tests[1]_tests.cmake")
else()
  add_test(run_unit_tests_NOT_BUILT run_unit_tests_NOT_BUILT)
endif()
