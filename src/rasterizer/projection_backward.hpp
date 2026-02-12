#pragma once

/// @file projection_backward.hpp
/// @brief Host-side declaration for the backward projection pass.

#include <torch/torch.h>

#include "core/types.hpp"

namespace cugs {

/// @brief Output of the backward projection pass.
///
/// Gradients w.r.t. all learnable Gaussian parameters.
struct ProjectionBackwardOutput {
    torch::Tensor dL_dpositions;  ///< dL/d(positions) [N, 3], float32
    torch::Tensor dL_drotations;  ///< dL/d(rotations) [N, 4], float32
    torch::Tensor dL_dscales;     ///< dL/d(scales) [N, 3], float32 (log-space)
    torch::Tensor dL_dopacities;  ///< dL/d(opacities) [N, 1], float32 (logit-space)
    torch::Tensor dL_dsh_coeffs;  ///< dL/d(sh_coeffs) [N, 3, C], float32
};

/// @brief Backward pass through the projection stage.
///
/// One thread per Gaussian. Recomputes forward intermediates and applies
/// the chain rule from rasterizer-level gradients (dL/d(means_2d),
/// dL/d(cov_2d_inv), dL/d(rgb), dL/d(opacity_act)) through to the
/// learnable parameters (positions, rotations, scales, opacities, SH coeffs).
///
/// @param dL_dmeans_2d     dL/d(means_2d) [N, 2] from rasterize_backward.
/// @param dL_dcov_2d_inv   dL/d(cov_2d_inv) [N, 3] from rasterize_backward.
/// @param dL_drgb          dL/d(rgb) [N, 3] from rasterize_backward.
/// @param dL_dopacity_act  dL/d(opacity_act) [N] from rasterize_backward.
/// @param positions        World-space positions [N, 3].
/// @param rotations        Quaternions (w,x,y,z) [N, 4].
/// @param scales           Log-space scales [N, 3].
/// @param opacities        Logit-space opacities [N, 1].
/// @param sh_coeffs        SH coefficients [N, 3, C].
/// @param radii            Pixel radii from forward [N] (0 = culled).
/// @param camera           Camera parameters.
/// @param active_sh_degree Active SH degree.
/// @param scale_modifier   Global scale modifier.
/// @return Gradients w.r.t. all learnable parameters.
ProjectionBackwardOutput project_backward(
    const torch::Tensor& dL_dmeans_2d,
    const torch::Tensor& dL_dcov_2d_inv,
    const torch::Tensor& dL_drgb,
    const torch::Tensor& dL_dopacity_act,
    const torch::Tensor& positions,
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_coeffs,
    const torch::Tensor& radii,
    const CameraInfo& camera,
    int active_sh_degree,
    float scale_modifier = 1.0f);

} // namespace cugs
