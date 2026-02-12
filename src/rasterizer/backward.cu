/// @file backward.cu
/// @brief CUDA kernel for the backward pass of tile-based alpha-compositing.
///
/// Mirrors the forward rasterizer structure: same 16×16 tile grid, same
/// shared memory batching of Gaussians. Traverses Gaussians back-to-front
/// (reverse order) to reconstruct transmittance and compute per-pixel
/// gradients, which are then scattered to per-Gaussian accumulators via
/// atomicAdd.
///
/// Reference: Kerbl et al. "3D Gaussian Splatting" (SIGGRAPH 2023), Section 5.

#include "rasterizer/backward.hpp"
#include "rasterizer/forward.cuh"
#include "utils/cuda_utils.cuh"

#include <cuda_runtime.h>
#include <torch/torch.h>

namespace cugs {

/// @brief Backward rasterization kernel.
///
/// Grid:  dim3(num_tiles_x, num_tiles_y) — one block per tile.
/// Block: dim3(kRastBlockX, kRastBlockY) = 16×16 = 256 threads.
///
/// Each block processes its tile's Gaussians in reverse order (back-to-front):
///   1. Load batch into shared memory (same SharedGaussian struct as forward)
///   2. Reconstruct transmittance T by dividing out (1 - alpha) going backwards
///   3. Compute per-pixel gradient contributions
///   4. Scatter to per-Gaussian accumulators via atomicAdd
__global__ void k_rasterize_backward(
    int num_tiles_x,
    int img_w, int img_h,
    float bg_r, float bg_g, float bg_b,
    const int* __restrict__ tile_ranges,
    const int* __restrict__ gaussian_idx,
    const float* __restrict__ means_2d,
    const float* __restrict__ cov_2d_inv,
    const float* __restrict__ rgb,
    const float* __restrict__ opacities,
    const float* __restrict__ dL_dcolor,     // [H, W, 3]
    const float* __restrict__ final_T_buf,   // [H, W]
    const int* __restrict__ n_contrib_buf,   // [H, W]
    float* __restrict__ dL_drgb_out,         // [N, 3]
    float* __restrict__ dL_dopacity_act_out, // [N]
    float* __restrict__ dL_dmeans_2d_out,    // [N, 2]
    float* __restrict__ dL_dcov_2d_inv_out,  // [N, 3]
    int n_gaussians)
{
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int tile_id = tile_y * num_tiles_x + tile_x;

    int px = tile_x * kRastBlockX + threadIdx.x;
    int py = tile_y * kRastBlockY + threadIdx.y;
    int thread_id = threadIdx.y * kRastBlockX + threadIdx.x;

    bool inside = (px < img_w) && (py < img_h);
    float pxf = static_cast<float>(px) + 0.5f;
    float pyf = static_cast<float>(py) + 0.5f;

    int range_start = tile_ranges[tile_id * 2 + 0];
    int range_end   = tile_ranges[tile_id * 2 + 1];
    int num_in_range = range_end - range_start;

    // Load per-pixel forward outputs
    int pixel_idx = py * img_w + px;
    float T = inside ? final_T_buf[pixel_idx] : 0.0f;
    int max_contrib = inside ? n_contrib_buf[pixel_idx] : 0;

    // Load dL/dC for this pixel
    float dL_dC[3] = {0.0f, 0.0f, 0.0f};
    if (inside) {
        dL_dC[0] = dL_dcolor[pixel_idx * 3 + 0];
        dL_dC[1] = dL_dcolor[pixel_idx * 3 + 1];
        dL_dC[2] = dL_dcolor[pixel_idx * 3 + 2];
    }

    // Running accumulator for the "contribution after" this Gaussian:
    // S_after = sum_{j > i} alpha_j * T_j * rgb_j (per channel)
    // Initialized to T_final * background (the background contribution)
    float S_after[3] = {0.0f, 0.0f, 0.0f};
    if (inside) {
        S_after[0] = T * bg_r;
        S_after[1] = T * bg_g;
        S_after[2] = T * bg_b;
    }

    // Shared memory for batched Gaussian loading
    __shared__ SharedGaussian shared_gaussians[kRastBlockSize];

    // Track how many actual contributors we've found going backwards.
    // When this reaches max_contrib, we've processed all forward contributors.
    int contributors_found = 0;
    bool done = !inside;

    // Process Gaussians in reverse: from range_end-1 down to range_start,
    // in batches of kRastBlockSize
    int num_batches = (num_in_range + kRastBlockSize - 1) / kRastBlockSize;

    for (int batch = num_batches - 1; batch >= 0; --batch) {
        // Cooperative load into shared memory
        int load_idx = range_start + batch * kRastBlockSize + thread_id;
        if (load_idx < range_end) {
            int g_idx = gaussian_idx[load_idx];
            shared_gaussians[thread_id].xy[0]         = means_2d[g_idx * 2 + 0];
            shared_gaussians[thread_id].xy[1]         = means_2d[g_idx * 2 + 1];
            shared_gaussians[thread_id].cov_2d_inv[0] = cov_2d_inv[g_idx * 3 + 0];
            shared_gaussians[thread_id].cov_2d_inv[1] = cov_2d_inv[g_idx * 3 + 1];
            shared_gaussians[thread_id].cov_2d_inv[2] = cov_2d_inv[g_idx * 3 + 2];
            shared_gaussians[thread_id].rgb[0]        = rgb[g_idx * 3 + 0];
            shared_gaussians[thread_id].rgb[1]        = rgb[g_idx * 3 + 1];
            shared_gaussians[thread_id].rgb[2]        = rgb[g_idx * 3 + 2];
            shared_gaussians[thread_id].opacity       = opacities[g_idx];
        }
        __syncthreads();

        // Process shared Gaussians in reverse within this batch
        if (!done) {
            int batch_count = min(kRastBlockSize, num_in_range - batch * kRastBlockSize);
            for (int j = batch_count - 1; j >= 0; --j) {
                // Global index for this Gaussian in the sorted array
                int sorted_idx = range_start + batch * kRastBlockSize + j;

                float dx = pxf - shared_gaussians[j].xy[0];
                float dy = pyf - shared_gaussians[j].xy[1];

                float a = shared_gaussians[j].cov_2d_inv[0];
                float b = shared_gaussians[j].cov_2d_inv[1];
                float c = shared_gaussians[j].cov_2d_inv[2];

                float power = -0.5f * (dx * (a * dx + b * dy) +
                                        dy * (b * dx + c * dy));
                if (power > 0.0f) continue;

                float alpha = shared_gaussians[j].opacity * expf(power);
                alpha = fminf(alpha, 0.99f);
                if (alpha < 1.0f / 255.0f) continue;

                // This Gaussian actually contributed in forward.
                contributors_found++;
                if (contributors_found > max_contrib) {
                    done = true;
                    break;
                }

                // Reconstruct T_i (transmittance BEFORE this Gaussian)
                // T_after = T_before * (1 - alpha)
                // T_before = T_after / (1 - alpha)
                float one_minus_alpha = fmaxf(1.0f - alpha, 1e-5f);
                T /= one_minus_alpha;

                float weight = alpha * T;

                // ---- dL/d(rgb_i) ----
                // C += alpha_i * T_i * rgb_i → dL/d(rgb_i) = dL/dC * alpha_i * T_i
                float dL_drgb_r = dL_dC[0] * weight;
                float dL_drgb_g = dL_dC[1] * weight;
                float dL_drgb_b = dL_dC[2] * weight;

                // ---- dL/d(alpha_i) ----
                // From the compositing equation:
                // dL/d(alpha_i) = dot(dL/dC, T_i * rgb_i) - dot(dL/dC, S_after) / (1 - alpha_i)
                // where S_after = sum of contributions after i (including background)
                float dL_dalpha = 0.0f;
                dL_dalpha += dL_dC[0] * (T * shared_gaussians[j].rgb[0] - S_after[0] / one_minus_alpha);
                dL_dalpha += dL_dC[1] * (T * shared_gaussians[j].rgb[1] - S_after[1] / one_minus_alpha);
                dL_dalpha += dL_dC[2] * (T * shared_gaussians[j].rgb[2] - S_after[2] / one_minus_alpha);

                // Update S_after: add current Gaussian's contribution
                S_after[0] += weight * shared_gaussians[j].rgb[0];
                S_after[1] += weight * shared_gaussians[j].rgb[1];
                S_after[2] += weight * shared_gaussians[j].rgb[2];

                // ---- dL/d(opacity_act) ----
                // alpha = opacity_act * exp(power)
                // dL/d(opacity_act) = dL/d(alpha) * exp(power)
                float exp_power = expf(power);
                // But if alpha was clamped to 0.99, gradient is zero
                float dL_dopacity_act = dL_dalpha * exp_power;
                if (shared_gaussians[j].opacity * exp_power >= 0.99f) {
                    dL_dopacity_act = 0.0f;
                }

                // ---- dL/d(power) ----
                // alpha = opacity_act * exp(power)
                // dL/d(power) = dL/d(alpha) * alpha (since d(exp(p))/dp = exp(p))
                float dL_dpower = dL_dalpha * alpha;
                if (shared_gaussians[j].opacity * exp_power >= 0.99f) {
                    dL_dpower = 0.0f;
                }

                // ---- dL/d(means_2d) ----
                // power = -0.5 * (dx*(a*dx+b*dy) + dy*(b*dx+c*dy))
                // d(power)/d(px_mean) = -d(power)/d(dx)
                //   = -(-0.5) * (a*dx+b*dy + dx*a + dy*b)  ... wait, let me redo:
                // d(power)/d(dx) = -0.5 * (2*a*dx + 2*b*dy) = -(a*dx + b*dy)
                // d(power)/d(dy) = -0.5 * (2*b*dx + 2*c*dy) = -(b*dx + c*dy)
                // dx = pxf - mean_x, so d(dx)/d(mean_x) = -1
                // d(power)/d(mean_x) = (a*dx + b*dy)
                // d(power)/d(mean_y) = (b*dx + c*dy)
                float dL_dmean_x = dL_dpower * (a * dx + b * dy);
                float dL_dmean_y = dL_dpower * (b * dx + c * dy);

                // ---- dL/d(cov_2d_inv) ----
                // power = -0.5 * (a*dx² + 2*b*dx*dy + c*dy²)
                // d(power)/d(a) = -0.5 * dx²
                // d(power)/d(b) = -0.5 * 2*dx*dy = -dx*dy
                // d(power)/d(c) = -0.5 * dy²
                float dL_da = dL_dpower * (-0.5f * dx * dx);
                float dL_db = dL_dpower * (-dx * dy);
                float dL_dc = dL_dpower * (-0.5f * dy * dy);

                // Scatter to global buffers via atomicAdd
                int g_idx = gaussian_idx[sorted_idx];

                atomicAdd(&dL_drgb_out[g_idx * 3 + 0], dL_drgb_r);
                atomicAdd(&dL_drgb_out[g_idx * 3 + 1], dL_drgb_g);
                atomicAdd(&dL_drgb_out[g_idx * 3 + 2], dL_drgb_b);

                atomicAdd(&dL_dopacity_act_out[g_idx], dL_dopacity_act);

                atomicAdd(&dL_dmeans_2d_out[g_idx * 2 + 0], dL_dmean_x);
                atomicAdd(&dL_dmeans_2d_out[g_idx * 2 + 1], dL_dmean_y);

                atomicAdd(&dL_dcov_2d_inv_out[g_idx * 3 + 0], dL_da);
                atomicAdd(&dL_dcov_2d_inv_out[g_idx * 3 + 1], dL_db);
                atomicAdd(&dL_dcov_2d_inv_out[g_idx * 3 + 2], dL_dc);
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

RasterizeBackwardOutput rasterize_backward(
    const torch::Tensor& dL_dcolor,
    const torch::Tensor& means_2d,
    const torch::Tensor& cov_2d_inv,
    const torch::Tensor& rgb,
    const torch::Tensor& opacities,
    const torch::Tensor& tile_ranges,
    const torch::Tensor& gaussian_indices,
    const torch::Tensor& final_T,
    const torch::Tensor& n_contrib,
    int img_w, int img_h,
    const float background[3],
    int n_gaussians)
{
    TORCH_CHECK(dL_dcolor.is_cuda(), "dL_dcolor must be on CUDA");

    auto device = dL_dcolor.device();
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // Allocate output gradient accumulators (zeroed)
    auto dL_drgb         = torch::zeros({n_gaussians, 3}, opts_f);
    auto dL_dopacity_act = torch::zeros({n_gaussians}, opts_f);
    auto dL_dmeans_2d    = torch::zeros({n_gaussians, 2}, opts_f);
    auto dL_dcov_2d_inv  = torch::zeros({n_gaussians, 3}, opts_f);

    int num_tiles_x = (img_w + 16 - 1) / 16;
    int num_tiles_y = (img_h + 16 - 1) / 16;

    if (num_tiles_x == 0 || num_tiles_y == 0) {
        return {dL_drgb, dL_dopacity_act, dL_dmeans_2d, dL_dcov_2d_inv};
    }

    auto dL_dcolor_c      = dL_dcolor.contiguous();
    auto means_2d_c       = means_2d.contiguous();
    auto cov_2d_inv_c     = cov_2d_inv.contiguous();
    auto rgb_c            = rgb.contiguous();
    auto opacities_c      = opacities.contiguous();
    auto tile_ranges_c    = tile_ranges.contiguous();
    auto gaussian_idx_c   = gaussian_indices.contiguous();
    auto final_T_c        = final_T.contiguous();
    auto n_contrib_c      = n_contrib.contiguous();

    dim3 grid(num_tiles_x, num_tiles_y);
    dim3 block(kRastBlockX, kRastBlockY);

    k_rasterize_backward<<<grid, block>>>(
        num_tiles_x,
        img_w, img_h,
        background[0], background[1], background[2],
        tile_ranges_c.data_ptr<int>(),
        gaussian_idx_c.data_ptr<int>(),
        means_2d_c.data_ptr<float>(),
        cov_2d_inv_c.data_ptr<float>(),
        rgb_c.data_ptr<float>(),
        opacities_c.data_ptr<float>(),
        dL_dcolor_c.data_ptr<float>(),
        final_T_c.data_ptr<float>(),
        n_contrib_c.data_ptr<int>(),
        dL_drgb.data_ptr<float>(),
        dL_dopacity_act.data_ptr<float>(),
        dL_dmeans_2d.data_ptr<float>(),
        dL_dcov_2d_inv.data_ptr<float>(),
        n_gaussians);

    CUDA_CHECK(cudaGetLastError());

    return {dL_drgb, dL_dopacity_act, dL_dmeans_2d, dL_dcov_2d_inv};
}

} // namespace cugs
