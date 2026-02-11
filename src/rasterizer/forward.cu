/// @file forward.cu
/// @brief CUDA kernel for tile-based alpha-compositing of sorted Gaussians.
///
/// The forward rasterizer processes each 16×16 pixel tile with one CUDA block.
/// Gaussians assigned to the tile (from the sorting stage) are loaded in batches
/// into shared memory, then each thread evaluates all shared Gaussians against
/// its pixel, accumulating color via front-to-back alpha compositing.
///
/// Reference: Kerbl et al. "3D Gaussian Splatting" (SIGGRAPH 2023), Section 4.

#include "rasterizer/forward.cuh"
#include "rasterizer/forward.hpp"
#include "rasterizer/sorting.hpp"
#include "utils/cuda_utils.cuh"

#include <cuda_runtime.h>
#include <torch/torch.h>

namespace cugs {

// ---------------------------------------------------------------------------
// Forward rasterization kernel
// ---------------------------------------------------------------------------

/// @brief Per-tile alpha-compositing kernel.
///
/// Grid:  dim3(num_tiles_x, num_tiles_y) — one block per tile.
/// Block: dim3(kRastBlockX, kRastBlockY) = 16×16 = 256 threads.
///
/// Each block processes Gaussians in batches of kRastBlockSize:
///   1. Load batch of Gaussians into shared memory (cooperative load)
///   2. Each thread evaluates all shared Gaussians against its pixel
///   3. Accumulate color: C += alpha * T * rgb; T *= (1 - alpha)
///   4. Early-exit when all threads in the block have T < threshold
///
/// @param num_tiles_x  Number of tiles in x direction.
/// @param img_w, img_h Image dimensions.
/// @param bg_r, bg_g, bg_b Background color.
/// @param tile_ranges  Per-tile [start, end) in sorted array [num_tiles, 2].
/// @param gaussian_idx Sorted Gaussian indices [P].
/// @param means_2d     2D positions [N, 2].
/// @param cov_2d_inv   Inverse 2D covariance [N, 3].
/// @param rgb          RGB colors [N, 3].
/// @param opacities    Activated opacities [N].
/// @param out_color    Output image [H, W, 3].
/// @param out_final_T  Output final transmittance [H, W].
/// @param out_n_contrib Output number of contributing Gaussians [H, W].
__global__ void k_rasterize_forward(
    int num_tiles_x,
    int img_w, int img_h,
    float bg_r, float bg_g, float bg_b,
    const int* __restrict__ tile_ranges,      // [num_tiles, 2]
    const int* __restrict__ gaussian_idx,      // [P]
    const float* __restrict__ means_2d,        // [N, 2]
    const float* __restrict__ cov_2d_inv,      // [N, 3]
    const float* __restrict__ rgb,             // [N, 3]
    const float* __restrict__ opacities,       // [N]
    float* __restrict__ out_color,             // [H, W, 3]
    float* __restrict__ out_final_T,           // [H, W]
    int* __restrict__ out_n_contrib)           // [H, W]
{
    // Tile and pixel coordinates
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int tile_id = tile_y * num_tiles_x + tile_x;

    int px = tile_x * kRastBlockX + threadIdx.x;
    int py = tile_y * kRastBlockY + threadIdx.y;
    int thread_id = threadIdx.y * kRastBlockX + threadIdx.x;

    bool inside = (px < img_w) && (py < img_h);
    float pxf = static_cast<float>(px) + 0.5f; // Pixel center
    float pyf = static_cast<float>(py) + 0.5f;

    // Load tile range
    int range_start = tile_ranges[tile_id * 2 + 0];
    int range_end   = tile_ranges[tile_id * 2 + 1];

    // Per-thread accumulation
    float T = 1.0f;
    float C[3] = {0.0f, 0.0f, 0.0f};
    int contributor_count = 0;
    bool done = !inside;

    // Shared memory for batched Gaussian loading
    __shared__ SharedGaussian shared_gaussians[kRastBlockSize];

    // Shared flag: are all threads done?
    __shared__ int block_done;

    // Process Gaussians in batches of kRastBlockSize
    int num_in_range = range_end - range_start;
    int num_batches = (num_in_range + kRastBlockSize - 1) / kRastBlockSize;

    for (int batch = 0; batch < num_batches; ++batch) {
        // ----- Early termination check for entire block -----
        if (thread_id == 0) block_done = 1;
        __syncthreads();
        if (!done) atomicMin(&block_done, 0);
        __syncthreads();
        if (block_done) break;

        // ----- Cooperative load into shared memory -----
        int load_idx = range_start + batch * kRastBlockSize + thread_id;
        if (load_idx < range_end) {
            int g_idx = gaussian_idx[load_idx];
            shared_gaussians[thread_id].xy[0]         = means_2d[g_idx * 2 + 0];
            shared_gaussians[thread_id].xy[1]         = means_2d[g_idx * 2 + 1];
            shared_gaussians[thread_id].cov_2d_inv[0]  = cov_2d_inv[g_idx * 3 + 0];
            shared_gaussians[thread_id].cov_2d_inv[1]  = cov_2d_inv[g_idx * 3 + 1];
            shared_gaussians[thread_id].cov_2d_inv[2]  = cov_2d_inv[g_idx * 3 + 2];
            shared_gaussians[thread_id].rgb[0]         = rgb[g_idx * 3 + 0];
            shared_gaussians[thread_id].rgb[1]         = rgb[g_idx * 3 + 1];
            shared_gaussians[thread_id].rgb[2]         = rgb[g_idx * 3 + 2];
            shared_gaussians[thread_id].opacity        = opacities[g_idx];
        }
        __syncthreads();

        // ----- Evaluate all loaded Gaussians against this pixel -----
        if (!done) {
            int batch_count = min(kRastBlockSize, num_in_range - batch * kRastBlockSize);
            for (int j = 0; j < batch_count; ++j) {
                float dx = pxf - shared_gaussians[j].xy[0];
                float dy = pyf - shared_gaussians[j].xy[1];

                float a = shared_gaussians[j].cov_2d_inv[0];
                float b = shared_gaussians[j].cov_2d_inv[1];
                float c = shared_gaussians[j].cov_2d_inv[2];

                // power = -0.5 * [dx, dy] * Σ⁻¹ * [dx, dy]ᵀ
                float power = -0.5f * (dx * (a * dx + b * dy) +
                                       dy * (b * dx + c * dy));

                // Skip if too far (Gaussian negligible at this pixel)
                if (power > 0.0f) continue;

                float alpha = shared_gaussians[j].opacity * expf(power);

                // Clamp alpha to valid range
                alpha = fminf(alpha, 0.99f);
                if (alpha < 1.0f / 255.0f) continue;

                float weight = alpha * T;

                C[0] += weight * shared_gaussians[j].rgb[0];
                C[1] += weight * shared_gaussians[j].rgb[1];
                C[2] += weight * shared_gaussians[j].rgb[2];

                T *= (1.0f - alpha);
                contributor_count++;

                // Early termination for this pixel
                if (T < kTransmittanceThreshold) {
                    done = true;
                    break;
                }
            }
        }
        __syncthreads();
    }

    // ----- Write output -----
    if (inside) {
        int pixel_idx = py * img_w + px;

        // Blend remaining transmittance with background
        out_color[pixel_idx * 3 + 0] = C[0] + T * bg_r;
        out_color[pixel_idx * 3 + 1] = C[1] + T * bg_g;
        out_color[pixel_idx * 3 + 2] = C[2] + T * bg_b;

        out_final_T[pixel_idx] = T;
        out_n_contrib[pixel_idx] = contributor_count;
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

ForwardOutput rasterize_forward(
    const torch::Tensor& means_2d,
    const torch::Tensor& cov_2d_inv,
    const torch::Tensor& rgb,
    const torch::Tensor& opacities,
    const torch::Tensor& tile_ranges,
    const torch::Tensor& gaussian_indices,
    int img_w, int img_h,
    const float background[3])
{
    TORCH_CHECK(means_2d.is_cuda(), "means_2d must be on CUDA");

    auto device = means_2d.device();
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(device);

    // Allocate output buffers
    auto out_color   = torch::zeros({img_h, img_w, 3}, opts_f);
    auto out_final_T = torch::ones({img_h, img_w}, opts_f);
    auto out_n_contrib = torch::zeros({img_h, img_w}, opts_i);

    int num_tiles_x = (img_w + kTileSize - 1) / kTileSize;
    int num_tiles_y = (img_h + kTileSize - 1) / kTileSize;

    if (num_tiles_x == 0 || num_tiles_y == 0) {
        // Fill with background
        out_color.select(2, 0).fill_(background[0]);
        out_color.select(2, 1).fill_(background[1]);
        out_color.select(2, 2).fill_(background[2]);
        return ForwardOutput{out_color, out_final_T, out_n_contrib};
    }

    // Ensure inputs are contiguous
    auto means_2d_c     = means_2d.contiguous();
    auto cov_2d_inv_c   = cov_2d_inv.contiguous();
    auto rgb_c          = rgb.contiguous();
    auto opacities_c    = opacities.contiguous();
    auto tile_ranges_c  = tile_ranges.contiguous();
    auto gaussian_idx_c = gaussian_indices.contiguous();

    dim3 grid(num_tiles_x, num_tiles_y);
    dim3 block(kRastBlockX, kRastBlockY);

    k_rasterize_forward<<<grid, block>>>(
        num_tiles_x,
        img_w, img_h,
        background[0], background[1], background[2],
        tile_ranges_c.data_ptr<int>(),
        gaussian_idx_c.data_ptr<int>(),
        means_2d_c.data_ptr<float>(),
        cov_2d_inv_c.data_ptr<float>(),
        rgb_c.data_ptr<float>(),
        opacities_c.data_ptr<float>(),
        out_color.data_ptr<float>(),
        out_final_T.data_ptr<float>(),
        out_n_contrib.data_ptr<int>());

    CUDA_CHECK(cudaGetLastError());

    return ForwardOutput{out_color, out_final_T, out_n_contrib};
}

} // namespace cugs
