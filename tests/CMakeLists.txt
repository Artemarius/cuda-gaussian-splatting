add_executable(test_cuda_utils test_cuda_utils.cpp)

target_link_libraries(test_cuda_utils PRIVATE
    cugs_utils
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_cuda_utils DISCOVERY_MODE PRE_TEST)
