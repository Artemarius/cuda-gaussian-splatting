#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "utils/cuda_utils.cuh"
#include "utils/cuda_info.hpp"

TEST(CudaUtils, GpuIsAccessible) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    EXPECT_GT(device_count, 0) << "No CUDA-capable GPU found";
}

TEST(CudaUtils, CanQueryDeviceProperties) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    EXPECT_GT(prop.major, 0);
    EXPECT_GT(prop.multiProcessorCount, 0);
}

TEST(CudaUtils, VramFreeIsPositive) {
    float free_mb = cugs::vram_free_mb();
    EXPECT_GT(free_mb, 0.0f);
}

TEST(CudaUtils, VramTotalIsReasonable) {
    float total_mb = cugs::vram_total_mb();
    // Any real GPU should have at least 1 GB
    EXPECT_GT(total_mb, 1000.0f);
}

TEST(CudaUtils, TestKernelSucceeds) {
    EXPECT_TRUE(cugs::print_gpu_info_and_test());
}
