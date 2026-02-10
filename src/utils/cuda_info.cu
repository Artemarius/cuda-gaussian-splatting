#include "utils/cuda_utils.cuh"

#include <spdlog/spdlog.h>
#include <cuda_runtime.h>

namespace cugs {

/// @brief Trivial kernel that writes thread indices to output — used to verify
///        CUDA compilation and kernel launch.
__global__ void k_hello(int* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = idx * idx;
    }
}

/// @brief Query the GPU, log its properties, and launch a trivial kernel to
///        verify the CUDA toolchain is working end to end.
/// @return true if a CUDA-capable GPU was found and the test kernel succeeded.
bool print_gpu_info_and_test() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        spdlog::error("No CUDA-capable GPU found.");
        return false;
    }

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));

        spdlog::info("GPU {}: {}", i, prop.name);
        spdlog::info("  Compute capability : {}.{}", prop.major, prop.minor);
        spdlog::info("  VRAM               : {:.0f} MB total, {:.0f} MB free",
                      vram_total_mb(), vram_free_mb());
        spdlog::info("  SM count           : {}", prop.multiProcessorCount);
        spdlog::info("  Max threads/block  : {}", prop.maxThreadsPerBlock);
        spdlog::info("  Warp size          : {}", prop.warpSize);
    }

    // Launch a trivial kernel to verify everything links and runs.
    constexpr int n = 32;
    int* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(int)));

    k_hello<<<1, n>>>(d_out, n);
    CUDA_SYNC_CHECK();

    int h_out[n];
    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));

    // Quick sanity check: 5^2 == 25
    if (h_out[5] != 25) {
        spdlog::error("Kernel output mismatch: expected 25, got {}", h_out[5]);
        return false;
    }

    spdlog::info("CUDA test kernel passed.");
    return true;
}

} // namespace cugs
