#pragma once

/// @file rasterizer.hpp
/// @brief Public API for the forward Gaussian splatting rasterizer.
///
/// This is the main entry point for rendering. It orchestrates the full
/// pipeline: projection → sorting → rasterization.

#include <torch/torch.h>

#include "core/gaussian.hpp"
#include "core/types.hpp"

namespace cugs {

/// @brief Settings controlling the rasterization pipeline.
struct RenderSettings {
    float background[3] = {0.0f, 0.0f, 0.0f}; ///< Background color (RGB, 0-1)
    int active_sh_degree = 3;                    ///< Active SH degree for color evaluation
    float scale_modifier = 1.0f;                 ///< Global scale multiplier for Gaussians
};

/// @brief Output of the full render pipeline.
///
/// Contains the rendered image plus all intermediate buffers needed for the
/// backward pass and diagnostics.
struct RenderOutput {
    // Final image
    torch::Tensor color;          ///< Rendered image [H, W, 3], float32

    // Per-pixel auxiliary data
    torch::Tensor final_T;        ///< Final transmittance [H, W], float32
    torch::Tensor n_contrib;      ///< Number of contributing Gaussians [H, W], int32

    // Projection intermediates (retained for backward pass)
    torch::Tensor means_2d;       ///< 2D positions [N, 2], float32
    torch::Tensor depths;         ///< View-space depth [N], float32
    torch::Tensor cov_2d_inv;     ///< Inverse 2D covariance [N, 3], float32
    torch::Tensor radii;          ///< Pixel radii [N], int32
    torch::Tensor rgb;            ///< SH-evaluated colors [N, 3], float32
    torch::Tensor opacities_act;  ///< Activated opacities [N], float32

    // Sorting intermediates (retained for backward pass)
    torch::Tensor gaussian_indices; ///< Sorted Gaussian indices [P], int32
    torch::Tensor tile_ranges;      ///< Per-tile ranges [num_tiles, 2], int32
};

/// @brief Render a set of 3D Gaussians to an image.
///
/// Full pipeline: projection → tile sorting → alpha compositing.
///
/// @param model    Gaussian model (positions, rotations, scales, opacities, SH).
///                 Must be on a CUDA device.
/// @param camera   Camera parameters (intrinsics + world-to-camera extrinsics).
/// @param settings Render settings (background, SH degree, scale modifier).
/// @return RenderOutput with the rendered image and all intermediates.
RenderOutput render(
    const GaussianModel& model,
    const CameraInfo& camera,
    const RenderSettings& settings);

} // namespace cugs
