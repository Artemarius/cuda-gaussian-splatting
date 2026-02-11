#pragma once

/// @file loss.hpp
/// @brief Loss functions for 3D Gaussian Splatting training.
///
/// Implements the combined loss from Kerbl et al. (SIGGRAPH 2023):
///   L = (1 - lambda) * L1 + lambda * (1 - SSIM)
///
/// All functions accept [H, W, 3] float32 CUDA tensors matching the
/// RenderOutput::color layout.

#include <torch/torch.h>

namespace cugs {

/// @brief Compute L1 (mean absolute error) loss between two images.
/// @param rendered Rendered image [H, W, 3], float32, CUDA.
/// @param target   Ground truth image [H, W, 3], float32, CUDA.
/// @return Scalar tensor (mean absolute difference).
torch::Tensor l1_loss(const torch::Tensor& rendered, const torch::Tensor& target);

/// @brief Compute the per-pixel SSIM map between two images.
///
/// Implements Wang et al. (2004) structural similarity with a Gaussian
/// weighting window. Returns a 2D map so it can be reused for evaluation
/// metrics (Phase 9).
///
/// @param rendered    Rendered image [H, W, 3], float32, CUDA.
/// @param target      Ground truth image [H, W, 3], float32, CUDA.
/// @param window_size Side length of the Gaussian weighting window (must be odd).
/// @return SSIM map [H, W], float32 (mean across RGB channels).
torch::Tensor ssim(const torch::Tensor& rendered, const torch::Tensor& target,
                   int window_size = 11);

/// @brief Compute SSIM loss: 1 - mean(SSIM map).
/// @param rendered    Rendered image [H, W, 3], float32, CUDA.
/// @param target      Ground truth image [H, W, 3], float32, CUDA.
/// @param window_size Side length of the Gaussian weighting window (must be odd).
/// @return Scalar tensor.
torch::Tensor ssim_loss(const torch::Tensor& rendered, const torch::Tensor& target,
                        int window_size = 11);

/// @brief Compute the combined training loss from Kerbl et al.
///
/// L = (1 - lambda) * L1(rendered, target) + lambda * (1 - mean SSIM(rendered, target))
///
/// @param rendered Rendered image [H, W, 3], float32, CUDA.
/// @param target   Ground truth image [H, W, 3], float32, CUDA.
/// @param lambda_  Weight for the SSIM term (default 0.2 per the paper).
/// @return Scalar tensor.
torch::Tensor combined_loss(const torch::Tensor& rendered, const torch::Tensor& target,
                            float lambda_ = 0.2f);

} // namespace cugs
