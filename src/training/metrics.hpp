#pragma once

/// @file metrics.hpp
/// @brief Evaluation metrics for 3D Gaussian Splatting (PSNR, SSIM).
///
/// Computes standard quality metrics on held-out test views to compare
/// against published results (Kerbl et al. SIGGRAPH 2023, Table 1).

#include "core/gaussian.hpp"
#include "data/dataset.hpp"
#include "rasterizer/rasterizer.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cugs {

/// @brief Compute PSNR between two [H, W, 3] float32 CUDA tensors.
///
/// PSNR = 10 * log10(1 / MSE), assuming pixel values in [0, 1].
/// Identical images are clamped at 100 dB to avoid infinity.
///
/// @param rendered Rendered image [H, W, 3], float32, CUDA.
/// @param target   Ground truth image [H, W, 3], float32, CUDA.
/// @return Scalar float (dB).
float compute_psnr(const torch::Tensor& rendered, const torch::Tensor& target);

/// @brief Compute mean SSIM between two [H, W, 3] float32 CUDA tensors.
///
/// Reuses cugs::ssim() from loss.hpp internally.
///
/// @param rendered Rendered image [H, W, 3], float32, CUDA.
/// @param target   Ground truth image [H, W, 3], float32, CUDA.
/// @return Scalar float in [-1, 1].
float compute_ssim(const torch::Tensor& rendered, const torch::Tensor& target);

/// @brief Per-image evaluation results.
struct ImageMetrics {
    std::string image_name;
    float psnr = 0.0f;
    float ssim = 0.0f;
};

/// @brief Aggregate evaluation results across all test views.
struct EvalResults {
    float mean_psnr = 0.0f;
    float mean_ssim = 0.0f;
    std::vector<ImageMetrics> per_image;
    int num_gaussians = 0;
    int sh_degree = 0;
    float eval_time_seconds = 0.0f;

    /// @brief Serialize to JSON string (pretty-printed).
    std::string to_json() const;

    /// @brief Write JSON to file.
    void save_json(const std::filesystem::path& path) const;
};

/// @brief Evaluate a trained model against a dataset's test set.
///
/// Renders all test views with NoGradGuard, computes PSNR and SSIM
/// for each image, and returns aggregate results.
///
/// @param model    Trained Gaussian model (on CUDA).
/// @param dataset  Dataset with test cameras and images.
/// @param settings Render settings (SH degree, background color).
/// @return EvalResults with per-image and mean metrics.
EvalResults evaluate(
    const GaussianModel& model,
    const Dataset& dataset,
    const RenderSettings& settings);

} // namespace cugs
