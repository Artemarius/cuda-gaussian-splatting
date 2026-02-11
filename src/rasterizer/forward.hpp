#pragma once

/// @file forward.hpp
/// @brief Host-side declaration for the forward rasterization pass.

#include <torch/torch.h>

namespace cugs {

/// @brief Output of the forward rasterization pass.
struct ForwardOutput {
    torch::Tensor color;            ///< Rendered image [H, W, 3], float32
    torch::Tensor final_T;          ///< Final transmittance per pixel [H, W], float32
    torch::Tensor n_contrib;        ///< Number of Gaussians contributing per pixel [H, W], int32
};

/// @brief Alpha-composite sorted Gaussians into a per-tile image.
///
/// Implements the forward pass of differentiable Gaussian splatting.
/// Each tile (16×16 pixels) processes its assigned Gaussians in front-to-back
/// depth order using shared memory batching.
///
/// Alpha for each Gaussian at pixel (px, py):
///   power = -0.5 * (dx * (a*dx + b*dy) + dy * (b*dx + c*dy))
///   alpha = opacity * exp(power)
///   color += alpha * T * rgb
///   T *= (1 - alpha)
///
/// Where (a, b, c) = inverse 2D covariance, (dx, dy) = pixel - gaussian_center.
///
/// @param means_2d         2D positions [N, 2], float32, CUDA.
/// @param cov_2d_inv       Inverse 2D covariance [N, 3], float32, CUDA.
/// @param rgb              RGB colors [N, 3], float32, CUDA.
/// @param opacities        Activated opacities [N], float32, CUDA.
/// @param tile_ranges      Per-tile [start, end) indices [num_tiles, 2], int32, CUDA.
/// @param gaussian_indices Sorted Gaussian indices [P], int32, CUDA.
/// @param img_w            Image width in pixels.
/// @param img_h            Image height in pixels.
/// @param background       Background color [3], float32 (on CPU is fine).
/// @return ForwardOutput with rendered image and auxiliary buffers.
ForwardOutput rasterize_forward(
    const torch::Tensor& means_2d,
    const torch::Tensor& cov_2d_inv,
    const torch::Tensor& rgb,
    const torch::Tensor& opacities,
    const torch::Tensor& tile_ranges,
    const torch::Tensor& gaussian_indices,
    int img_w, int img_h,
    const float background[3]);

} // namespace cugs
