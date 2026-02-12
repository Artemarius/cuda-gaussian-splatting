#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace cugs {

/// @brief Check a CUDA runtime call and throw on error.
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(                                           \
                std::string("CUDA error at ") + __FILE__ + ":" +                \
                std::to_string(__LINE__) + " — " + cudaGetErrorString(err));     \
        }                                                                       \
    } while (0)

/// @brief Synchronize and check for errors. Only active in debug builds.
#ifdef NDEBUG
#define CUDA_SYNC_CHECK() ((void)0)
#else
#define CUDA_SYNC_CHECK()                                                      \
    do {                                                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                    \
        CUDA_CHECK(cudaGetLastError());                                         \
    } while (0)
#endif

/// @brief Returns the amount of free VRAM in megabytes.
inline float vram_free_mb() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return static_cast<float>(free_bytes) / (1024.0f * 1024.0f);
}

/// @brief Returns the amount of used VRAM in megabytes.
inline float vram_used_mb() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return static_cast<float>(total_bytes - free_bytes) / (1024.0f * 1024.0f);
}

/// @brief Returns total VRAM in megabytes.
inline float vram_total_mb() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return static_cast<float>(total_bytes) / (1024.0f * 1024.0f);
}

/// @brief Non-throwing VRAM query result.
struct VramInfo {
    float free_mb  = -1.0f;
    float total_mb = -1.0f;
    bool valid() const { return free_mb >= 0.0f; }
    float used_mb() const { return total_mb - free_mb; }
};

/// @brief Non-throwing VRAM query — returns {-1, -1} on error.
inline VramInfo vram_info_mb() {
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return {};
    return { static_cast<float>(free_bytes) / (1024.f * 1024.f),
             static_cast<float>(total_bytes) / (1024.f * 1024.f) };
}

} // namespace cugs
