#pragma once

#include <torch/torch.h>

namespace cugs {

/// @brief Evaluate spherical harmonics for a batch of directions and
///        per-Gaussian SH coefficients.
///
/// This is the CPU reference implementation. For the GPU path, see
/// evaluate_sh_cuda() in sh.cu.
///
/// @param degree Active SH degree (0..3). Only coefficients up to this
///        degree are evaluated; higher-order coefficients are ignored.
/// @param sh_coeffs SH coefficients tensor [N, 3, C] where C >= (degree+1)^2.
/// @param directions Unit direction vectors [N, 3] (camera-to-point, normalised).
/// @return RGB colors [N, 3] in linear space. Values may be negative or >1
///         due to SH representation; caller should clamp.
torch::Tensor evaluate_sh_cpu(int degree,
                              const torch::Tensor& sh_coeffs,
                              const torch::Tensor& directions);

/// @brief Evaluate spherical harmonics on GPU.
///
/// @param degree Active SH degree (0..3).
/// @param sh_coeffs SH coefficients tensor [N, 3, C] on CUDA device.
/// @param directions Unit direction vectors [N, 3] on CUDA device.
/// @return RGB colors [N, 3] on the same CUDA device.
torch::Tensor evaluate_sh_cuda(int degree,
                               const torch::Tensor& sh_coeffs,
                               const torch::Tensor& directions);

/// @brief Evaluate SH, dispatching to CPU or CUDA based on tensor device.
inline torch::Tensor evaluate_sh(int degree,
                                 const torch::Tensor& sh_coeffs,
                                 const torch::Tensor& directions) {
    if (sh_coeffs.is_cuda()) {
        return evaluate_sh_cuda(degree, sh_coeffs, directions);
    }
    return evaluate_sh_cpu(degree, sh_coeffs, directions);
}

// ---------------------------------------------------------------------------
// SH constants — the Y_l^m basis function coefficients.
// These are the real-valued SH basis functions used in computer graphics,
// following the convention from "An Efficient Representation for Irradiance
// Environment Maps" (Ramamoorthi & Hanrahan, 2001).
// ---------------------------------------------------------------------------

// Degree 0
constexpr float kSH_C0 = 0.28209479177387814f;  // 1/(2*sqrt(pi))

// Degree 1
constexpr float kSH_C1 = 0.4886025119029199f;   // sqrt(3/(4*pi))

// Degree 2
constexpr float kSH_C2_0 = 1.0925484305920792f;  // sqrt(15/(4*pi))
constexpr float kSH_C2_1 = 1.0925484305920792f;  // sqrt(15/(4*pi))
constexpr float kSH_C2_2 = 0.31539156525252005f;  // sqrt(5/(16*pi))
constexpr float kSH_C2_3 = 1.0925484305920792f;  // sqrt(15/(4*pi))
constexpr float kSH_C2_4 = 0.5462742152960396f;  // sqrt(15/(16*pi))

// Degree 3
constexpr float kSH_C3_0 = 0.5900435899266435f;
constexpr float kSH_C3_1 = 2.890611442640554f;
constexpr float kSH_C3_2 = 0.4570457994644658f;
constexpr float kSH_C3_3 = 0.3731763325901154f;
constexpr float kSH_C3_4 = 0.4570457994644658f;
constexpr float kSH_C3_5 = 1.4453057213202769f;
constexpr float kSH_C3_6 = 0.5900435899266435f;

} // namespace cugs
