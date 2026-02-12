#pragma once

/// @file densification.hpp
/// @brief Adaptive density control for 3D Gaussian Splatting.
///
/// Implements the clone/split/prune densification system from Kerbl et al.
/// (SIGGRAPH 2023, Section 5). Gaussians with high position gradients are
/// cloned (if small) or split (if large); low-opacity Gaussians are pruned.
///
/// Key references:
///   - Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field
///     Rendering", Section 5: Optimization with Adaptive Density Control

#include "core/gaussian.hpp"

#include <torch/torch.h>

#include <cstdint>

namespace cugs {

/// @brief Configuration for the densification schedule and thresholds.
struct DensificationConfig {
    // Schedule
    int densify_from  = 500;    ///< First iteration to densify
    int densify_until = 15000;  ///< Last iteration to densify
    int densify_every = 100;    ///< Densification frequency (iterations)
    int opacity_reset_every = 3000; ///< Opacity reset frequency

    // Thresholds
    float grad_threshold    = 0.0002f; ///< Position gradient norm threshold
    float opacity_threshold = 0.005f;  ///< Min sigmoid opacity to survive pruning
    float percent_dense     = 0.01f;   ///< Fraction of scene extent for clone/split
    int   max_screen_size   = 20;      ///< Max pixel radius before pruning

    // Capacity
    int   max_gaussians       = 0;     ///< Hard cap (0 = no limit)
    float min_vram_headroom_mb = 512.0f; ///< Skip densification if free VRAM below this
};

/// @brief Statistics from a single densification step.
struct DensificationStats {
    int num_cloned  = 0; ///< Number of Gaussians cloned
    int num_split   = 0; ///< Number of Gaussians split (originals replaced)
    int num_pruned  = 0; ///< Number of Gaussians pruned
    int num_before  = 0; ///< Gaussian count before densification
    int num_after   = 0; ///< Gaussian count after densification
    bool skipped_vram = false; ///< True if skipped due to low VRAM
};

/// @brief Adaptive density controller for Gaussian Splatting training.
///
/// Accumulates per-Gaussian position gradient norms across iterations.
/// When triggered (every `densify_every` steps), performs:
///   1. Clone: duplicate small Gaussians with high average gradient
///   2. Split: replace large Gaussians with 2 smaller children
///   3. Prune: remove low-opacity or oversized Gaussians
///
/// The controller also handles periodic opacity resets, which prevents
/// Gaussians from becoming permanently transparent.
class DensificationController {
public:
    /// @brief Construct the controller.
    /// @param config Densification schedule and threshold parameters.
    /// @param scene_extent Scene bounding sphere radius (from Dataset::scene_bounds().extent).
    DensificationController(const DensificationConfig& config, float scene_extent);

    /// @brief Accumulate screen-space position gradient norms for visible Gaussians.
    ///
    /// Called every iteration after the backward pass. Only Gaussians with
    /// radii > 0 (visible in the current view) contribute to the average.
    ///
    /// Following the reference implementation, the densification metric is
    /// ||dL/d(screen_xy)||_2, the norm of the 2D screen-space position
    /// gradient. This measures how much the projected position needs to
    /// change to reduce the loss.
    ///
    /// @param dL_dmeans_2d Screen-space position gradients [N, 2] from BackwardOutput.
    /// @param radii Per-Gaussian pixel radii [N] from RenderOutput.
    void accumulate_gradients(const torch::Tensor& dL_dmeans_2d,
                              const torch::Tensor& radii);

    /// @brief Check if densification should run at this step.
    bool should_densify(int step) const;

    /// @brief Check if opacity reset should run at this step.
    bool should_reset_opacity(int step) const;

    /// @brief Run the full clone/split/prune densification cycle.
    ///
    /// Modifies the model in-place: appends cloned/split Gaussians and
    /// removes pruned ones. After this call, all model tensors may have
    /// a different leading dimension.
    ///
    /// @param model Gaussian model to densify (modified in-place).
    /// @param step Current training iteration.
    /// @return Statistics about the densification step.
    DensificationStats densify(GaussianModel& model, int step);

    /// @brief Reset all opacities to inverse_sigmoid(0.01) = -4.595.
    ///
    /// This prevents Gaussians from becoming permanently transparent and
    /// gives them a chance to contribute again during training.
    ///
    /// @param model Gaussian model (modified in-place).
    void reset_opacity(GaussianModel& model);

private:
    DensificationConfig config_;
    float scene_extent_;

    // Accumulation buffers (lazy-initialized, reset after densification)
    torch::Tensor grad_accum_;   ///< Running sum of gradient norms [N]
    torch::Tensor grad_count_;   ///< Number of observations per Gaussian [N]
    torch::Tensor max_radii_2d_; ///< Max observed screen radius [N]

    /// @brief Reset or re-allocate accumulation buffers for n Gaussians.
    void reset_accumulators(int64_t n);

    /// @brief Compute mask of Gaussians to clone (small + high gradient).
    /// @return Boolean tensor [N], true = should clone.
    torch::Tensor compute_clone_mask(const GaussianModel& model);

    /// @brief Compute mask of Gaussians to split (large + high gradient).
    /// @return Boolean tensor [N], true = should split.
    torch::Tensor compute_split_mask(const GaussianModel& model);

    /// @brief Compute mask of Gaussians to keep (inverse of prune).
    ///
    /// A Gaussian is kept if:
    ///   - sigmoid(opacity) >= opacity_threshold, AND
    ///   - (only after first opacity reset): max screen radius <= max_screen_size
    ///   - (only after first opacity reset): max world-space scale <= 0.1 * scene_extent
    ///
    /// Following the reference implementation, screen-size and world-space-size
    /// pruning only applies after the first opacity reset (step > opacity_reset_every).
    ///
    /// @param model The Gaussian model.
    /// @param step Current training iteration.
    /// @return Boolean tensor [N], true = KEEP.
    torch::Tensor compute_keep_mask(const GaussianModel& model, int step);

    /// @brief Append new Gaussians to the model by boolean mask.
    ///
    /// For each true entry in mask, copies the corresponding Gaussian from
    /// the model (or uses override tensors if provided).
    ///
    /// @param model Gaussian model (modified in-place).
    /// @param mask Boolean selection mask [N].
    /// @param new_positions Override positions [M, 3] (optional).
    /// @param new_scales Override scales [M, 3] (optional).
    void append_gaussians(GaussianModel& model,
                          const torch::Tensor& mask,
                          const torch::Tensor& new_positions = {},
                          const torch::Tensor& new_scales = {});

    /// @brief Remove Gaussians where keep_mask is false.
    ///
    /// Indexes, detaches, clones, and makes contiguous all model tensors.
    ///
    /// @param model Gaussian model (modified in-place).
    /// @param keep_mask Boolean tensor [N], true = keep.
    void prune_gaussians(GaussianModel& model, const torch::Tensor& keep_mask);

    /// @brief Get free VRAM in megabytes.
    static float vram_free_mb();
};

} // namespace cugs
