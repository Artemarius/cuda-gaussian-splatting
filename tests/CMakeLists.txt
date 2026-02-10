add_executable(test_cuda_utils test_cuda_utils.cpp)

target_link_libraries(test_cuda_utils PRIVATE
    cugs_utils
    GTest::gtest_main
)

add_executable(test_colmap_loader test_colmap_loader.cpp)

target_link_libraries(test_colmap_loader PRIVATE
    cugs_data
    GTest::gtest_main
)

add_executable(test_dataset test_dataset.cpp)

target_link_libraries(test_dataset PRIVATE
    cugs_data
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_cuda_utils DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_colmap_loader DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_dataset DISCOVERY_MODE PRE_TEST)
