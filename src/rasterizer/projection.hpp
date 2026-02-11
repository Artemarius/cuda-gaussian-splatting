#pragma once

/// @file projection.hpp
/// @brief Host-side declarations for projecting 3D Gaussians to 2D.

#include <torch/torch.h>

#include "core/types.hpp"

namespace cugs {

/// @brief Output of the projection stage.
///
/// All tensors are on the same CUDA device as the inputs.
struct ProjectionOutput {
    torch::Tensor means_2d;       ///< Projected 2D means [N, 2] (pixel coords)
    torch::Tensor depths;         ///< View-space depth (z) [N]
    torch::Tensor cov_2d_inv;     ///< Inverse 2D covariance [N, 3] (a, b, c)
    torch::Tensor radii;          ///< Pixel radius per Gaussian [N] (int32)
    torch::Tensor tiles_touched;  ///< Number of tiles touched [N] (int32)
    torch::Tensor rgb;            ///< View-dependent color [N, 3] (from SH)
    torch::Tensor opacities_act;  ///< Activated (sigmoid) opacity [N]
};

/// @brief Project 3D Gaussians to 2D screen space, compute view-dependent
///        colors via SH evaluation, and determine tile coverage.
///
/// This is the first stage of the rasterization pipeline.
///
/// @param positions   World-space positions [N, 3], float32, CUDA.
/// @param rotations   Quaternions (w,x,y,z) [N, 4], float32, CUDA.
/// @param scales      Log-space scales [N, 3], float32, CUDA.
/// @param opacities   Logit-space opacities [N, 1], float32, CUDA.
/// @param sh_coeffs   SH coefficients [N, 3, C], float32, CUDA.
/// @param camera      Camera info (intrinsics + extrinsics).
/// @param active_sh_degree Active SH degree for color evaluation (0..3).
/// @param scale_modifier Global multiplier applied to scales (default 1.0).
/// @return Projection results for all N Gaussians.
ProjectionOutput project_gaussians(
    const torch::Tensor& positions,
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_coeffs,
    const CameraInfo& camera,
    int active_sh_degree,
    float scale_modifier = 1.0f);

} // namespace cugs
