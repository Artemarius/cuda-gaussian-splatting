#pragma once

/// @file backward.hpp
/// @brief Host-side declaration for the backward rasterization pass.

#include <torch/torch.h>

namespace cugs {

/// @brief Output of the backward rasterization pass.
///
/// Contains per-Gaussian gradients accumulated from all contributing pixels.
struct RasterizeBackwardOutput {
    torch::Tensor dL_drgb;          ///< dL/d(rgb) [N, 3], float32
    torch::Tensor dL_dopacity_act;  ///< dL/d(opacity_activated) [N], float32
    torch::Tensor dL_dmeans_2d;     ///< dL/d(means_2d) [N, 2], float32
    torch::Tensor dL_dcov_2d_inv;   ///< dL/d(cov_2d_inv) [N, 3], float32
};

/// @brief Backward pass of the tile-based alpha-compositing rasterizer.
///
/// Mirrors the forward pass structure: same tile grid, same shared memory
/// batching, but traverses Gaussians back-to-front and accumulates gradients
/// via atomicAdd.
///
/// @param dL_dcolor      Gradient of loss w.r.t. rendered image [H, W, 3].
/// @param means_2d       2D positions from forward [N, 2].
/// @param cov_2d_inv     Inverse 2D covariance from forward [N, 3].
/// @param rgb            RGB colors from forward [N, 3].
/// @param opacities      Activated opacities from forward [N].
/// @param tile_ranges    Per-tile [start, end) from sorting [num_tiles, 2].
/// @param gaussian_indices Sorted Gaussian indices from sorting [P].
/// @param final_T        Final transmittance per pixel from forward [H, W].
/// @param n_contrib      Number of contributors per pixel from forward [H, W].
/// @param img_w, img_h   Image dimensions.
/// @param background     Background color [3].
/// @param n_gaussians    Total number of Gaussians N.
/// @return Per-Gaussian gradients.
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
    int n_gaussians);

} // namespace cugs
