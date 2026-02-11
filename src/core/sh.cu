#include "core/sh.hpp"
#include "utils/cuda_utils.cuh"

#include <torch/torch.h>

namespace cugs {

/// @brief CUDA kernel: evaluate SH for each Gaussian.
///
/// Grid: one thread per Gaussian (1D grid).
/// Each thread evaluates SH for all 3 color channels.
///
/// @param degree Active SH degree (0..3).
/// @param sh_coeffs [N, 3, C] SH coefficients (row-major).
/// @param directions [N, 3] unit direction vectors.
/// @param output [N, 3] output RGB colors.
/// @param n Number of Gaussians.
/// @param num_coeffs Number of SH coefficients per channel (C).
__global__ void k_evaluate_sh(
    int degree,
    const float* __restrict__ sh_coeffs,
    const float* __restrict__ directions,
    float* __restrict__ output,
    int n,
    int num_coeffs)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Direction for this Gaussian
    const float x = directions[idx * 3 + 0];
    const float y = directions[idx * 3 + 1];
    const float z = directions[idx * 3 + 2];

    // sh_coeffs layout: [N, 3, C] row-major
    // For Gaussian idx, channel ch, coeff k:
    //   sh_coeffs[idx * 3 * num_coeffs + ch * num_coeffs + k]

    for (int ch = 0; ch < 3; ++ch) {
        const float* c = sh_coeffs + idx * 3 * num_coeffs + ch * num_coeffs;
        float color = 0.0f;

        // Degree 0
        color += 0.28209479177387814f * c[0];

        if (degree >= 1) {
            color += 0.4886025119029199f * (
                -c[1] * y +
                 c[2] * z +
                -c[3] * x
            );
        }

        if (degree >= 2) {
            const float xx = x * x, yy = y * y, zz = z * z;
            const float xy = x * y, xz = x * z, yz = y * z;

            color += 1.0925484305920792f * c[4] * xy;
            color += 1.0925484305920792f * c[5] * yz;
            color += 0.31539156525252005f * c[6] * (2*zz - xx - yy);
            color += 1.0925484305920792f * c[7] * xz;
            color += 0.5462742152960396f * c[8] * (xx - yy);
        }

        if (degree >= 3) {
            const float xx = x * x, yy = y * y, zz = z * z;

            color += 0.5900435899266435f * c[9]  * y * (3*xx - yy);
            color += 2.890611442640554f  * c[10] * x * y * z;
            color += 0.4570457994644658f * c[11] * y * (4*zz - xx - yy);
            color += 0.3731763325901154f * c[12] * z * (2*zz - 3*xx - 3*yy);
            color += 0.4570457994644658f * c[13] * x * (4*zz - xx - yy);
            color += 1.4453057213202769f * c[14] * z * (xx - yy);
            color += 0.5900435899266435f * c[15] * x * (xx - 3*yy);
        }

        output[idx * 3 + ch] = color + 0.5f;
    }
}

torch::Tensor evaluate_sh_cuda(int degree,
                                const torch::Tensor& sh_coeffs,
                                const torch::Tensor& directions) {
    TORCH_CHECK(degree >= 0 && degree <= 3, "SH degree must be 0..3, got ", degree);
    TORCH_CHECK(sh_coeffs.is_cuda(), "sh_coeffs must be on CUDA device");
    TORCH_CHECK(directions.is_cuda(), "directions must be on CUDA device");
    TORCH_CHECK(sh_coeffs.dim() == 3 && sh_coeffs.size(1) == 3,
                "sh_coeffs must be [N, 3, C]");
    TORCH_CHECK(directions.dim() == 2 && directions.size(1) == 3,
                "directions must be [N, 3]");
    TORCH_CHECK(sh_coeffs.size(0) == directions.size(0), "Batch size mismatch");

    const int required_coeffs = (degree + 1) * (degree + 1);
    TORCH_CHECK(sh_coeffs.size(2) >= required_coeffs,
                "Need at least ", required_coeffs, " coefficients for degree ", degree);

    const int64_t n = sh_coeffs.size(0);
    const int num_coeffs = static_cast<int>(sh_coeffs.size(2));

    auto coeffs = sh_coeffs.to(torch::kFloat32).contiguous();
    auto dirs = directions.to(torch::kFloat32).contiguous();

    auto output = torch::zeros({n, 3}, torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(sh_coeffs.device()));

    if (n == 0) return output;

    const int threads = 256;
    const int blocks = (static_cast<int>(n) + threads - 1) / threads;

    k_evaluate_sh<<<blocks, threads>>>(
        degree,
        coeffs.data_ptr<float>(),
        dirs.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(n),
        num_coeffs
    );
    CUDA_CHECK(cudaGetLastError());

    return output;
}

} // namespace cugs
