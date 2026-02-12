#pragma once

/// @file mcmc_densification.hpp
/// @brief MCMC-based densification for 3D Gaussian Splatting.
///
/// Implements the fixed-count densification strategy from Kheradmand et al.,
/// "3D Gaussian Splatting as Markov Chain Monte Carlo" (NeurIPS 2024).
///
/// Instead of growing/shrinking N via clone/split/prune, MCMC densification
/// maintains a constant Gaussian count by:
///   1. Relocating "dead" (low-opacity) Gaussians to high-opacity regions
///   2. Injecting position noise every iteration to guide exploration
///   3. Applying opacity/scale regularization to prevent degenerate solutions
///
/// This is ideal for VRAM-constrained training (e.g., 6GB GPUs) where the
/// Gaussian count must be tightly controlled.

#include "core/gaussian.hpp"

#include <torch/torch.h>

#include <cstdint>

namespace cugs {

/// @brief Configuration for MCMC densification.
struct MCMCConfig {
    // Schedule
    int relocate_from  = 500;    ///< First iteration to relocate
    int relocate_until = 15000;  ///< Last iteration to relocate
    int relocate_every = 100;    ///< Relocation frequency (iterations)

    // Relocation
    float dead_opacity_threshold = 0.005f; ///< sigmoid(opa) below this = "dead"
    float relocate_cap           = 0.05f;  ///< Max fraction of N to relocate per step

    // Noise injection
    float noise_lr_init  = 5e5f;   ///< Initial noise learning rate
    float noise_lr_final = 1e3f;   ///< Final noise learning rate
    int   noise_lr_max_steps = 30000; ///< Steps over which noise LR decays
    float noise_gate_k   = 100.0f; ///< Sigmoid gate steepness
    float noise_gate_t   = 0.995f; ///< Sigmoid gate opacity threshold

    // Regularization
    float lambda_opacity = 0.01f;  ///< Opacity regularization weight
    float lambda_scale   = 0.01f;  ///< Scale regularization weight

    // VRAM
    float min_vram_headroom_mb  = 512.0f; ///< Skip relocation if free VRAM below this
    float effective_vram_limit_mb = 0.0f; ///< Effective VRAM limit (set by Trainer)
};

/// @brief Statistics from a single MCMC relocation step.
struct MCMCStats {
    int num_relocated = 0; ///< Number of dead Gaussians relocated
    int num_dead      = 0; ///< Number of dead Gaussians detected
    int num_total     = 0; ///< Total Gaussian count (constant)
    bool skipped_vram = false; ///< True if skipped due to low VRAM
};

/// @brief MCMC densification controller for fixed-count Gaussian training.
///
/// Maintains a constant number of Gaussians throughout training by:
///   - Relocating dead Gaussians to regions sampled from alive Gaussians
///   - Injecting per-iteration position noise scaled by Gaussian size and opacity
///   - Computing opacity/scale regularization for gradient injection
///
/// Key reference: Kheradmand et al. (NeurIPS 2024), Sections 3.2-3.4.
class MCMCController {
public:
    /// @brief Construct the controller.
    /// @param config MCMC schedule and parameter configuration.
    /// @param scene_extent Scene bounding sphere radius.
    MCMCController(const MCMCConfig& config, float scene_extent);

    /// @brief Check if relocation should run at this step.
    /// @param step Current training iteration.
    /// @return True if relocation is scheduled.
    bool should_relocate(int step) const;

    /// @brief Relocate dead Gaussians to opacity-weighted samples from alive ones.
    ///
    /// Dead Gaussians (sigmoid(opa) < threshold) are teleported to positions
    /// near alive Gaussians, sampled proportionally to opacity. Properties
    /// (SH, rotation) are copied from the source; scale is set small and
    /// opacity low. The total count N remains constant.
    ///
    /// @param model Gaussian model (modified in-place).
    /// @param step Current training iteration.
    /// @return Statistics about the relocation.
    MCMCStats relocate(GaussianModel& model, int step);

    /// @brief Inject position noise for MCMC exploration.
    ///
    /// Each Gaussian's position is perturbed by:
    ///   noise = noise_lr(step) * exp(scales) * gate(opacity) * randn
    /// where gate = sigmoid(-k * (sigmoid(opa) - t)) ensures low-opacity
    /// Gaussians receive more noise (encouraging exploration).
    ///
    /// Called every iteration after the optimizer step, under NoGradGuard.
    ///
    /// @param model Gaussian model (modified in-place).
    /// @param step Current training iteration.
    void inject_noise(GaussianModel& model, int step);

    /// @brief Compute opacity + scale regularization loss.
    ///
    /// Returns:
    ///   lambda_o * mean(sigmoid(opa)) + lambda_s * mean(exp(scales))
    ///
    /// The returned tensors (for opacities and scales) have .grad() populated
    /// so the caller can add them to BackwardOutput gradients.
    ///
    /// @param model Gaussian model (read-only).
    /// @param[out] reg_dL_dopacities Gradient of regularization w.r.t. opacities [N, 1].
    /// @param[out] reg_dL_dscales Gradient of regularization w.r.t. scales [N, 3].
    /// @return Scalar regularization loss value.
    float compute_regularization(const GaussianModel& model,
                                 torch::Tensor& reg_dL_dopacities,
                                 torch::Tensor& reg_dL_dscales);

    /// @brief Compute the noise learning rate at a given step.
    ///
    /// Log-linear decay from noise_lr_init to noise_lr_final, same formula
    /// as position_lr() in lr_schedule.hpp.
    ///
    /// @param step Current training iteration.
    /// @return Noise learning rate for this step.
    float noise_lr(int step) const;

private:
    MCMCConfig config_;
    float scene_extent_;
};

} // namespace cugs
