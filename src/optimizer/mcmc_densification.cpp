/// @file mcmc_densification.cpp
/// @brief MCMC densification implementation.
///
/// Implements the fixed-count densification strategy from Kheradmand et al.
/// (NeurIPS 2024). All operations use libtorch tensor ops — no custom CUDA
/// kernels needed since relocation only runs every ~100 iterations and noise
/// injection is a few elementwise ops per iteration.

#include "optimizer/mcmc_densification.hpp"
#include "utils/memory_monitor.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>

namespace cugs {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MCMCController::MCMCController(const MCMCConfig& config, float scene_extent)
    : config_(config), scene_extent_(scene_extent) {
}

// ---------------------------------------------------------------------------
// Schedule
// ---------------------------------------------------------------------------

bool MCMCController::should_relocate(int step) const {
    return step >= config_.relocate_from &&
           step <= config_.relocate_until &&
           step % config_.relocate_every == 0;
}

// ---------------------------------------------------------------------------
// Noise learning rate
// ---------------------------------------------------------------------------

float MCMCController::noise_lr(int step) const {
    if (step >= config_.noise_lr_max_steps) return config_.noise_lr_final;
    if (step <= 0) return config_.noise_lr_init;

    // Log-linear interpolation: lr = init * (final/init)^(t/T)
    const float t = static_cast<float>(step) /
                    static_cast<float>(config_.noise_lr_max_steps);
    const float log_ratio = std::log(config_.noise_lr_final / config_.noise_lr_init);
    return config_.noise_lr_init * std::exp(t * log_ratio);
}

// ---------------------------------------------------------------------------
// Relocation
// ---------------------------------------------------------------------------

MCMCStats MCMCController::relocate(GaussianModel& model, int step) {
    MCMCStats stats;
    stats.num_total = static_cast<int>(model.num_gaussians());

    // VRAM guard.
    auto vram = vram_info_mb();
    if (vram.valid()) {
        float budget = (config_.effective_vram_limit_mb > 0.0f)
            ? (config_.effective_vram_limit_mb - vram.used_mb())
            : vram.free_mb;
        if (budget < config_.min_vram_headroom_mb) {
            spdlog::warn("Skipping MCMC relocation: {:.0f} MB budget "
                         "(need {:.0f} MB headroom)",
                         budget, config_.min_vram_headroom_mb);
            stats.skipped_vram = true;
            return stats;
        }
    }

    torch::NoGradGuard no_grad;

    const int64_t n = model.num_gaussians();

    // Identify dead Gaussians: sigmoid(opacity) < threshold.
    auto opacity_activated = torch::sigmoid(model.opacities.squeeze(1)); // [N]
    auto dead_mask = opacity_activated.lt(config_.dead_opacity_threshold); // [N] bool
    auto alive_mask = ~dead_mask;

    int num_dead = dead_mask.sum().item<int>();
    int num_alive = n - num_dead;
    stats.num_dead = num_dead;

    if (num_dead == 0 || num_alive == 0) {
        return stats;
    }

    // Cap the number of relocations.
    int max_relocate = static_cast<int>(config_.relocate_cap * n);
    int num_to_relocate = std::min(num_dead, max_relocate);

    // Get indices of dead and alive Gaussians.
    auto dead_indices = dead_mask.nonzero().squeeze(1);   // [num_dead]
    auto alive_indices = alive_mask.nonzero().squeeze(1); // [num_alive]

    // If capped, only relocate the first num_to_relocate dead Gaussians.
    if (num_to_relocate < num_dead) {
        dead_indices = dead_indices.slice(0, 0, num_to_relocate);
    }

    // Sample source indices from alive Gaussians, weighted by opacity.
    auto alive_opacities = opacity_activated.index({alive_indices}); // [num_alive]
    auto weights = alive_opacities / alive_opacities.sum(); // Normalize to probabilities.

    auto source_local = torch::multinomial(
        weights, num_to_relocate, /*replacement=*/true); // [num_to_relocate]
    auto source_indices = alive_indices.index({source_local}); // [num_to_relocate]

    // Copy properties from source to dead Gaussians.
    // SH coefficients and rotations: copy from source.
    model.sh_coeffs.index_put_({dead_indices},
        model.sh_coeffs.index({source_indices}));
    model.rotations.index_put_({dead_indices},
        model.rotations.index({source_indices}));

    // Position: source position + small jitter.
    auto source_pos = model.positions.index({source_indices}); // [M, 3]
    auto jitter = torch::randn_like(source_pos) * scene_extent_ * 0.01f;
    model.positions.index_put_({dead_indices}, source_pos + jitter);

    // Scale: set small (log-space). Use source scale reduced significantly.
    auto source_scales = model.scales.index({source_indices}); // [M, 3]
    auto new_scales = source_scales - std::log(10.0f); // 10x smaller than source.
    model.scales.index_put_({dead_indices}, new_scales);

    // Opacity: set to low value (logit-space). inverse_sigmoid(0.01) ~ -4.595.
    float low_opacity = std::log(0.01f / 0.99f);
    model.opacities.index_put_({dead_indices},
        torch::full({num_to_relocate, 1}, low_opacity,
                    model.opacities.options()));

    stats.num_relocated = num_to_relocate;
    return stats;
}

// ---------------------------------------------------------------------------
// Noise injection
// ---------------------------------------------------------------------------

void MCMCController::inject_noise(GaussianModel& model, int step) {
    torch::NoGradGuard no_grad;

    const float lr = noise_lr(step);

    // Gate: sigmoid(-k * (sigmoid(opa) - t))
    // Low-opacity Gaussians get gate ~ 1 (more noise).
    // High-opacity Gaussians get gate ~ 0 (less noise).
    auto opa_activated = torch::sigmoid(model.opacities); // [N, 1]
    auto gate = torch::sigmoid(
        -config_.noise_gate_k * (opa_activated - config_.noise_gate_t)); // [N, 1]

    // Noise: lr * exp(scales) * gate * randn_like(positions)
    auto actual_scale = torch::exp(model.scales); // [N, 3]
    auto noise = lr * actual_scale * gate * torch::randn_like(model.positions);

    model.positions += noise;
}

// ---------------------------------------------------------------------------
// Regularization
// ---------------------------------------------------------------------------

float MCMCController::compute_regularization(
    const GaussianModel& model,
    torch::Tensor& reg_dL_dopacities,
    torch::Tensor& reg_dL_dscales) {

    // Clone and detach so autograd doesn't touch the model's tensors.
    auto opa_copy = model.opacities.clone().detach().requires_grad_(true);
    auto scl_copy = model.scales.clone().detach().requires_grad_(true);

    // Regularization: penalize high average opacity and large scales.
    auto reg_loss = config_.lambda_opacity * torch::sigmoid(opa_copy).mean()
                  + config_.lambda_scale * torch::exp(scl_copy).mean();

    reg_loss.backward();

    reg_dL_dopacities = opa_copy.grad().clone();
    reg_dL_dscales = scl_copy.grad().clone();

    return reg_loss.item<float>();
}

} // namespace cugs
