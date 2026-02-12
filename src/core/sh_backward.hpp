#pragma once

/// @file sh_backward.hpp
/// @brief Backward pass for spherical harmonics evaluation.

#include <torch/torch.h>

namespace cugs {

/// @brief Compute gradients of SH coefficients given dL/d(color).
///
/// Since SH evaluation is linear in coefficients:
///   color_ch = sum_k(c_k * Y_k(dir)) + 0.5
/// the gradient is:
///   dL/d(c_k) = dL/d(color_ch) * Y_k(dir)
///
/// Handles the ReLU clamp from the forward pass: gradient is zero when
/// the raw SH output (before clamp) was negative.
///
/// @param degree    Active SH degree (0..3).
/// @param sh_coeffs SH coefficients [N, 3, C], float32, CUDA.
/// @param directions Unit direction vectors [N, 3], float32, CUDA.
/// @param dL_dcolor Gradient of loss w.r.t. clamped RGB colors [N, 3], float32, CUDA.
/// @return dL/d(sh_coeffs) [N, 3, C], float32, same device.
torch::Tensor evaluate_sh_backward_cuda(
    int degree,
    const torch::Tensor& sh_coeffs,
    const torch::Tensor& directions,
    const torch::Tensor& dL_dcolor);

} // namespace cugs
