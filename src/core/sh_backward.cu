/// @file sh_backward.cu
/// @brief CUDA kernel for the backward pass of spherical harmonics evaluation.
///
/// SH evaluation is linear in coefficients: color = Σ c_k * Y_k(dir) + 0.5.
/// Gradient: dL/d(c_k) = dL/d(color) * Y_k(dir), gated by the forward ReLU clamp.
///
/// Grid: one thread per Gaussian (1D grid), same as forward SH kernel.

#include "core/sh_backward.hpp"
#include "core/sh.hpp"
#include "utils/cuda_utils.cuh"

#include <torch/torch.h>

namespace cugs {

/// @brief Backward kernel for SH evaluation.
///
/// Grid:  ((N + 255) / 256) blocks × 256 threads.
/// Each thread computes dL/d(sh_coeffs) for one Gaussian, all 3 channels.
///
/// @param degree     Active SH degree (0..3).
/// @param sh_coeffs  [N, 3, C] SH coefficients (row-major).
/// @param directions [N, 3] unit direction vectors.
/// @param dL_dcolor  [N, 3] gradient of loss w.r.t. clamped color.
/// @param dL_dsh     [N, 3, C] output gradients for SH coefficients.
/// @param n          Number of Gaussians.
/// @param num_coeffs Number of SH coefficients per channel (C).
__global__ void k_evaluate_sh_backward(
    int degree,
    const float* __restrict__ sh_coeffs,
    const float* __restrict__ directions,
    const float* __restrict__ dL_dcolor,
    float* __restrict__ dL_dsh,
    int n,
    int num_coeffs)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float x = directions[idx * 3 + 0];
    const float y = directions[idx * 3 + 1];
    const float z = directions[idx * 3 + 2];

    // Precompute SH basis values for all active degrees
    float Y[16]; // max 16 coefficients for degree 3
    int num_active = 0;

    // Degree 0
    Y[0] = 0.28209479177387814f;
    num_active = 1;

    if (degree >= 1) {
        Y[1] = -0.4886025119029199f * y;
        Y[2] =  0.4886025119029199f * z;
        Y[3] = -0.4886025119029199f * x;
        num_active = 4;
    }

    if (degree >= 2) {
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, xz = x * z, yz = y * z;

        Y[4] = 1.0925484305920792f * xy;
        Y[5] = 1.0925484305920792f * yz;
        Y[6] = 0.31539156525252005f * (2 * zz - xx - yy);
        Y[7] = 1.0925484305920792f * xz;
        Y[8] = 0.5462742152960396f * (xx - yy);
        num_active = 9;
    }

    if (degree >= 3) {
        const float xx = x * x, yy = y * y, zz = z * z;

        Y[9]  = 0.5900435899266435f * y * (3 * xx - yy);
        Y[10] = 2.890611442640554f  * x * y * z;
        Y[11] = 0.4570457994644658f * y * (4 * zz - xx - yy);
        Y[12] = 0.3731763325901154f * z * (2 * zz - 3 * xx - 3 * yy);
        Y[13] = 0.4570457994644658f * x * (4 * zz - xx - yy);
        Y[14] = 1.4453057213202769f * z * (xx - yy);
        Y[15] = 0.5900435899266435f * x * (xx - 3 * yy);
        num_active = 16;
    }

    // For each color channel, compute raw SH output to check ReLU gate
    for (int ch = 0; ch < 3; ++ch) {
        const float* c = sh_coeffs + idx * 3 * num_coeffs + ch * num_coeffs;
        float* dsh = dL_dsh + idx * 3 * num_coeffs + ch * num_coeffs;
        float dL_dc = dL_dcolor[idx * 3 + ch];

        // Recompute raw color to check ReLU clamp
        float raw_color = 0.0f;
        for (int k = 0; k < num_active; ++k) {
            raw_color += c[k] * Y[k];
        }
        raw_color += 0.5f;

        // ReLU gate: gradient is zero if forward output was clamped to 0
        float gate = (raw_color > 0.0f) ? 1.0f : 0.0f;
        float dL_dc_gated = dL_dc * gate;

        // dL/d(c_k) = dL/d(color) * Y_k(dir)
        for (int k = 0; k < num_active; ++k) {
            dsh[k] = dL_dc_gated * Y[k];
        }

        // Zero out unused coefficients
        for (int k = num_active; k < num_coeffs; ++k) {
            dsh[k] = 0.0f;
        }
    }
}

torch::Tensor evaluate_sh_backward_cuda(
    int degree,
    const torch::Tensor& sh_coeffs,
    const torch::Tensor& directions,
    const torch::Tensor& dL_dcolor)
{
    TORCH_CHECK(degree >= 0 && degree <= 3, "SH degree must be 0..3, got ", degree);
    TORCH_CHECK(sh_coeffs.is_cuda(), "sh_coeffs must be on CUDA device");
    TORCH_CHECK(directions.is_cuda(), "directions must be on CUDA device");
    TORCH_CHECK(dL_dcolor.is_cuda(), "dL_dcolor must be on CUDA device");
    TORCH_CHECK(sh_coeffs.dim() == 3 && sh_coeffs.size(1) == 3,
                "sh_coeffs must be [N, 3, C]");
    TORCH_CHECK(directions.dim() == 2 && directions.size(1) == 3,
                "directions must be [N, 3]");
    TORCH_CHECK(dL_dcolor.dim() == 2 && dL_dcolor.size(1) == 3,
                "dL_dcolor must be [N, 3]");

    const int64_t n = sh_coeffs.size(0);
    const int num_coeffs = static_cast<int>(sh_coeffs.size(2));

    auto coeffs = sh_coeffs.to(torch::kFloat32).contiguous();
    auto dirs = directions.to(torch::kFloat32).contiguous();
    auto dl_dc = dL_dcolor.to(torch::kFloat32).contiguous();

    auto dL_dsh = torch::zeros_like(coeffs);

    if (n == 0) return dL_dsh;

    const int threads = 256;
    const int blocks = (static_cast<int>(n) + threads - 1) / threads;

    k_evaluate_sh_backward<<<blocks, threads>>>(
        degree,
        coeffs.data_ptr<float>(),
        dirs.data_ptr<float>(),
        dl_dc.data_ptr<float>(),
        dL_dsh.data_ptr<float>(),
        static_cast<int>(n),
        num_coeffs);
    CUDA_CHECK(cudaGetLastError());

    return dL_dsh;
}

} // namespace cugs
