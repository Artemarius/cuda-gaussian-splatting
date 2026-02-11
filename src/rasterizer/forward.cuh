#pragma once

/// @file forward.cuh
/// @brief Shared-memory data structure and constants for the forward rasterizer.

#include <cuda_runtime.h>

namespace cugs {

/// @brief Per-Gaussian data loaded into shared memory for tile-based rasterization.
///
/// Each tile block loads batches of BLOCK_SIZE Gaussians into shared memory.
/// Threads then evaluate all shared Gaussians against their pixel.
///
/// Size: 10 floats × 4 bytes = 40 bytes per entry.
/// For BLOCK_SIZE=256: 256 × 40 = 10,240 bytes (10 KB) shared memory per block.
struct SharedGaussian {
    float xy[2];         ///< Screen-space position (px, py)
    float cov_2d_inv[3]; ///< Inverse 2D covariance (a, b, c)
    float rgb[3];        ///< View-dependent RGB color
    float opacity;       ///< Activated opacity (sigmoid)
};

/// @brief Rasterizer block size: 16×16 = 256 threads per tile.
constexpr int kRastBlockX = 16;
constexpr int kRastBlockY = 16;
constexpr int kRastBlockSize = kRastBlockX * kRastBlockY;

/// @brief Transmittance threshold for early termination.
/// When a pixel's accumulated transmittance drops below this, stop compositing.
constexpr float kTransmittanceThreshold = 1.0f / 255.0f;

} // namespace cugs
