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

add_executable(test_ply_io test_ply_io.cpp)

target_link_libraries(test_ply_io PRIVATE
    cugs_data
    GTest::gtest_main
)

add_executable(test_gaussian_model test_gaussian_model.cpp)

target_link_libraries(test_gaussian_model PRIVATE
    cugs_data
    GTest::gtest_main
)

add_executable(test_sh test_sh.cpp)

target_link_libraries(test_sh PRIVATE
    cugs_data
    cugs_utils
    GTest::gtest_main
)

add_executable(test_gaussian_init test_gaussian_init.cpp)

target_link_libraries(test_gaussian_init PRIVATE
    cugs_data
    GTest::gtest_main
)

add_executable(test_projection test_projection.cpp)

target_link_libraries(test_projection PRIVATE
    cugs_rasterizer
    GTest::gtest_main
)

add_executable(test_rasterizer test_rasterizer.cpp)

target_link_libraries(test_rasterizer PRIVATE
    cugs_rasterizer
    GTest::gtest_main
)

add_executable(test_loss test_loss.cpp)

target_link_libraries(test_loss PRIVATE
    cugs_training
    GTest::gtest_main
)

add_executable(test_backward test_backward.cpp)

target_link_libraries(test_backward PRIVATE
    cugs_rasterizer
    cugs_training
    GTest::gtest_main
)

add_executable(test_training test_training.cpp)

target_link_libraries(test_training PRIVATE
    cugs_training
    GTest::gtest_main
)

add_executable(test_densification test_densification.cpp)

target_link_libraries(test_densification PRIVATE
    cugs_training
    GTest::gtest_main
)

add_executable(test_fused_adam test_fused_adam.cpp)

target_link_libraries(test_fused_adam PRIVATE
    cugs_training
    GTest::gtest_main
)

add_executable(test_metrics test_metrics.cpp)

target_link_libraries(test_metrics PRIVATE
    cugs_training
    GTest::gtest_main
)

add_executable(test_mcmc test_mcmc.cpp)

target_link_libraries(test_mcmc PRIVATE
    cugs_training
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_training DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_backward DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_loss DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_cuda_utils DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_colmap_loader DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_dataset DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_ply_io DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_gaussian_model DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_sh DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_gaussian_init DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_projection DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_rasterizer DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_densification DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_fused_adam DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_metrics DISCOVERY_MODE PRE_TEST)
gtest_discover_tests(test_mcmc DISCOVERY_MODE PRE_TEST)
