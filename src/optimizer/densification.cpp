/// @file densification.cpp
/// @brief Adaptive density control implementation.
///
/// Implements clone/split/prune from Kerbl et al. (SIGGRAPH 2023, Section 5).
/// All operations use libtorch tensor ops — no custom CUDA kernels needed
/// since densification only runs every ~100 iterations.

#include "optimizer/densification.hpp"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <cmath>

namespace cugs {

namespace {

/// @brief Compute inverse sigmoid: log(x / (1 - x)).
inline float inverse_sigmoid(float x) {
    return std::log(x / (1.0f - x));
}

/// Opacity value used during reset: inverse_sigmoid(0.01).
constexpr float kResetOpacity = -4.59511985013459f; // log(0.01 / 0.99)

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

DensificationController::DensificationController(
    const DensificationConfig& config, float scene_extent)
    : config_(config), scene_extent_(scene_extent) {
}

// ---------------------------------------------------------------------------
// Schedule queries
// ---------------------------------------------------------------------------

bool DensificationController::should_densify(int step) const {
    return step >= config_.densify_from &&
           step <= config_.densify_until &&
           step % config_.densify_every == 0;
}

bool DensificationController::should_reset_opacity(int step) const {
    return config_.opacity_reset_every > 0 &&
           step >= config_.densify_from &&
           step % config_.opacity_reset_every == 0;
}

// ---------------------------------------------------------------------------
// Gradient accumulation
// ---------------------------------------------------------------------------

void DensificationController::accumulate_gradients(
    const torch::Tensor& dL_dmeans_2d,
    const torch::Tensor& radii) {

    const int64_t n = dL_dmeans_2d.size(0);

    // Lazy init or re-init if model size changed (e.g., after densification).
    if (!grad_accum_.defined() || grad_accum_.size(0) != n) {
        reset_accumulators(n);
    }

    // Visibility mask: only accumulate for Gaussians visible in this view.
    auto visible = radii.gt(0); // [N] bool

    // Gradient norm per Gaussian: ||dL/d(screen_xy)||_2
    // Following the reference implementation, the densification metric is the
    // norm of the 2D screen-space position gradient, NOT the 3D world-space
    // gradient. This measures how much the projected position should move.
    auto grad_norms = dL_dmeans_2d.norm(2, /*dim=*/1); // [N]

    // Accumulate only for visible Gaussians.
    grad_accum_.index_put_({visible},
        grad_accum_.index({visible}) + grad_norms.index({visible}));
    grad_count_.index_put_({visible},
        grad_count_.index({visible}) + 1);

    // Track max screen radius.
    auto radii_float = radii.to(torch::kFloat32);
    max_radii_2d_ = torch::max(max_radii_2d_, radii_float);
}

// ---------------------------------------------------------------------------
// Densification
// ---------------------------------------------------------------------------

DensificationStats DensificationController::densify(
    GaussianModel& model, int step) {

    DensificationStats stats;
    stats.num_before = static_cast<int>(model.num_gaussians());

    // VRAM guard.
    float free_mb = vram_free_mb();
    if (free_mb > 0.0f && free_mb < config_.min_vram_headroom_mb) {
        spdlog::warn("Skipping densification: only {:.0f} MB free (need {:.0f} MB)",
                     free_mb, config_.min_vram_headroom_mb);
        stats.skipped_vram = true;
        stats.num_after = stats.num_before;
        return stats;
    }

    torch::NoGradGuard no_grad;

    // 1. Clone small Gaussians with high gradients.
    auto clone_mask = compute_clone_mask(model);
    int num_to_clone = clone_mask.sum().item<int>();

    if (num_to_clone > 0) {
        // Budget check: respect max_gaussians.
        if (config_.max_gaussians > 0) {
            int budget = config_.max_gaussians - static_cast<int>(model.num_gaussians());
            if (num_to_clone > budget) {
                // Only clone up to budget — pick highest-gradient ones.
                if (budget <= 0) {
                    num_to_clone = 0;
                    clone_mask.zero_();
                } else {
                    auto avg_grad = grad_accum_ / grad_count_.clamp_min(1);
                    auto clone_grads = avg_grad.masked_fill(~clone_mask, -1.0f);
                    auto [_, indices] = clone_grads.topk(budget);
                    clone_mask.zero_();
                    clone_mask.index_fill_(0, indices, true);
                    num_to_clone = budget;
                }
            }
        }

        if (num_to_clone > 0) {
            append_gaussians(model, clone_mask);
            stats.num_cloned = num_to_clone;
        }
    }

    // 2. Split large Gaussians with high gradients.
    // Compute split mask on the current model but restrict to original Gaussians
    // (indices < num_before). Cloned Gaussians appended at the end are excluded.
    auto split_mask = compute_split_mask(model);
    if (stats.num_cloned > 0) {
        split_mask.slice(0, stats.num_before).zero_();
    }
    int num_to_split = split_mask.sum().item<int>();

    if (num_to_split > 0) {
        // Budget check: each split creates 2 new Gaussians (original is pruned later).
        // Net change per split = +2 -1 = +1, but we add 2 now and prune 1 later.
        if (config_.max_gaussians > 0) {
            int budget = (config_.max_gaussians -
                          static_cast<int>(model.num_gaussians())) / 2;
            if (num_to_split > budget) {
                if (budget <= 0) {
                    num_to_split = 0;
                    split_mask.zero_();
                } else {
                    auto avg_grad = grad_accum_ / grad_count_.clamp_min(1);
                    // Extend avg_grad if model grew from cloning.
                    if (avg_grad.size(0) < model.num_gaussians()) {
                        auto pad = torch::zeros(
                            {model.num_gaussians() - avg_grad.size(0)},
                            avg_grad.options());
                        avg_grad = torch::cat({avg_grad, pad});
                    }
                    auto split_grads = avg_grad.masked_fill(~split_mask, -1.0f);
                    auto [_, indices] = split_grads.topk(budget);
                    split_mask.zero_();
                    split_mask.index_fill_(0, indices, true);
                    num_to_split = budget;
                }
            }
        }

        if (num_to_split > 0) {
            // Generate 2 children per split Gaussian with reduced scale and
            // jittered positions.
            auto selected_pos = model.positions.index({split_mask}); // [M, 3]
            auto selected_scales = model.scales.index({split_mask}); // [M, 3]

            // Reduce scale: new_scale = old_scale - log(1.6)
            float log_scale_factor = std::log(1.6f);
            auto new_scales = selected_scales - log_scale_factor; // [M, 3]

            // Sample 2 sets of positions around the old mean.
            // pos_child = old_pos + randn * exp(new_scale)
            auto actual_scale = torch::exp(new_scales); // [M, 3]
            auto noise1 = torch::randn_like(selected_pos) * actual_scale;
            auto noise2 = torch::randn_like(selected_pos) * actual_scale;
            auto pos1 = selected_pos + noise1;
            auto pos2 = selected_pos + noise2;

            auto child_positions = torch::cat({pos1, pos2}, 0); // [2M, 3]
            auto child_scales = torch::cat({new_scales, new_scales}, 0); // [2M, 3]

            // Build a doubled mask to select other properties too.
            auto selected_sh = model.sh_coeffs.index({split_mask});
            auto selected_opa = model.opacities.index({split_mask});
            auto selected_rot = model.rotations.index({split_mask});

            // Append 2M new Gaussians with child positions/scales and
            // copied SH/opacity/rotation from the parent.
            model.positions = torch::cat(
                {model.positions, child_positions}, 0);
            model.scales = torch::cat(
                {model.scales, child_scales}, 0);
            model.sh_coeffs = torch::cat(
                {model.sh_coeffs,
                 torch::cat({selected_sh, selected_sh}, 0)}, 0);
            model.opacities = torch::cat(
                {model.opacities,
                 torch::cat({selected_opa, selected_opa}, 0)}, 0);
            model.rotations = torch::cat(
                {model.rotations,
                 torch::cat({selected_rot, selected_rot}, 0)}, 0);

            stats.num_split = num_to_split;
        }
    }

    // 3. Prune: remove low-opacity, oversized, and split-original Gaussians.
    auto keep_mask = compute_keep_mask(model, step);

    // Also remove the originals that were split (replaced by 2 children each).
    // split_mask covers the pre-split model size; children are appended at end.
    if (num_to_split > 0) {
        auto remove_originals = torch::zeros(
            {model.num_gaussians()},
            torch::TensorOptions().dtype(torch::kBool).device(split_mask.device()));
        remove_originals.slice(0, 0, split_mask.size(0)).copy_(split_mask);
        keep_mask = keep_mask & ~remove_originals;
    }

    // Newly cloned/split Gaussians should survive pruning unconditionally.
    // They start at index stats.num_before (clones) and at the split append point.
    // The simplest approach: ensure keep_mask is true for all indices >= num_before.
    if (stats.num_cloned > 0 || stats.num_split > 0) {
        int64_t new_start = stats.num_before;
        keep_mask.slice(0, new_start).fill_(true);
    }

    int total_before_prune = static_cast<int>(model.num_gaussians());
    prune_gaussians(model, keep_mask);
    stats.num_pruned = total_before_prune - static_cast<int>(model.num_gaussians());

    stats.num_after = static_cast<int>(model.num_gaussians());

    // Reset accumulators for the new model size.
    reset_accumulators(model.num_gaussians());

    return stats;
}

// ---------------------------------------------------------------------------
// Opacity reset
// ---------------------------------------------------------------------------

void DensificationController::reset_opacity(GaussianModel& model) {
    torch::NoGradGuard no_grad;
    model.opacities.fill_(kResetOpacity);
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

void DensificationController::reset_accumulators(int64_t n) {
    auto device = grad_accum_.defined() ? grad_accum_.device() : torch::kCUDA;
    grad_accum_   = torch::zeros({n}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    grad_count_   = torch::zeros({n}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    max_radii_2d_ = torch::zeros({n}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
}

torch::Tensor DensificationController::compute_clone_mask(
    const GaussianModel& model) {

    auto device = model.positions.device();
    const int64_t n = model.num_gaussians();

    // Average gradient norm (avoid division by zero).
    auto avg_grad = grad_accum_.slice(0, 0, n) /
                    grad_count_.slice(0, 0, n).clamp_min(1);

    // High gradient condition.
    auto high_grad = avg_grad.ge(config_.grad_threshold); // [N] bool

    // Small scale condition: max(exp(scale)) < percent_dense * scene_extent.
    auto max_scale = std::get<0>(torch::exp(model.scales).max(/*dim=*/1)); // [N]
    float size_threshold = config_.percent_dense * scene_extent_;
    auto small = max_scale.lt(size_threshold); // [N] bool

    return high_grad & small;
}

torch::Tensor DensificationController::compute_split_mask(
    const GaussianModel& model) {

    const int64_t n = model.num_gaussians();
    const int64_t accum_n = grad_accum_.size(0);
    const int64_t effective_n = std::min(n, accum_n);

    // Average gradient norm.
    auto avg_grad = grad_accum_.slice(0, 0, effective_n) /
                    grad_count_.slice(0, 0, effective_n).clamp_min(1);

    // If model grew (from cloning), pad with zeros.
    if (effective_n < n) {
        auto pad = torch::zeros({n - effective_n}, avg_grad.options());
        avg_grad = torch::cat({avg_grad, pad});
    }

    // High gradient condition.
    auto high_grad = avg_grad.ge(config_.grad_threshold);

    // Large scale condition: max(exp(scale)) >= percent_dense * scene_extent.
    auto max_scale = std::get<0>(torch::exp(model.scales).max(/*dim=*/1));
    float size_threshold = config_.percent_dense * scene_extent_;
    auto large = max_scale.ge(size_threshold);

    return high_grad & large;
}

torch::Tensor DensificationController::compute_keep_mask(
    const GaussianModel& model, int step) {

    const int64_t n = model.num_gaussians();
    auto device = model.positions.device();

    // Opacity check: keep if sigmoid(opacity) >= threshold.
    auto opacity_activated = torch::sigmoid(model.opacities.squeeze(1)); // [N]
    auto keep = opacity_activated.ge(config_.opacity_threshold); // [N] bool

    // Size pruning: only applies after the first opacity reset.
    // This matches the reference implementation which sets size_threshold = 20
    // only when iteration > opacity_reset_interval (default 3000).
    // Before the opacity reset, Gaussians may legitimately have large screen
    // footprints, so pruning by size would be too aggressive.
    bool apply_size_pruning = config_.opacity_reset_every > 0 &&
                              step > config_.opacity_reset_every;

    if (apply_size_pruning) {
        // Screen size check: remove if max observed radius > max_screen_size.
        if (config_.max_screen_size > 0 && max_radii_2d_.defined()) {
            torch::Tensor radii;
            if (max_radii_2d_.size(0) >= n) {
                radii = max_radii_2d_.slice(0, 0, n);
            } else {
                auto pad = torch::zeros({n - max_radii_2d_.size(0)},
                                        max_radii_2d_.options());
                radii = torch::cat({max_radii_2d_, pad});
            }
            auto not_too_big = radii.le(
                static_cast<float>(config_.max_screen_size));
            keep = keep & not_too_big;
        }

        // World-space size check: remove if max(exp(scale)) > 0.1 * scene_extent.
        // This catches Gaussians that grew too large in world space.
        auto max_scale = std::get<0>(torch::exp(model.scales).max(/*dim=*/1));
        float ws_threshold = 0.1f * scene_extent_;
        auto not_too_large_ws = max_scale.le(ws_threshold);
        keep = keep & not_too_large_ws;
    }

    return keep;
}

void DensificationController::append_gaussians(
    GaussianModel& model,
    const torch::Tensor& mask,
    const torch::Tensor& new_positions,
    const torch::Tensor& new_scales) {

    auto pos = new_positions.defined()
        ? new_positions : model.positions.index({mask});
    auto scl = new_scales.defined()
        ? new_scales : model.scales.index({mask});
    auto sh  = model.sh_coeffs.index({mask});
    auto opa = model.opacities.index({mask});
    auto rot = model.rotations.index({mask});

    model.positions = torch::cat({model.positions, pos}, 0);
    model.scales    = torch::cat({model.scales, scl}, 0);
    model.sh_coeffs = torch::cat({model.sh_coeffs, sh}, 0);
    model.opacities = torch::cat({model.opacities, opa}, 0);
    model.rotations = torch::cat({model.rotations, rot}, 0);
}

void DensificationController::prune_gaussians(
    GaussianModel& model, const torch::Tensor& keep_mask) {

    model.positions = model.positions.index({keep_mask}).detach().clone().contiguous();
    model.sh_coeffs = model.sh_coeffs.index({keep_mask}).detach().clone().contiguous();
    model.opacities = model.opacities.index({keep_mask}).detach().clone().contiguous();
    model.rotations = model.rotations.index({keep_mask}).detach().clone().contiguous();
    model.scales    = model.scales.index({keep_mask}).detach().clone().contiguous();
}

float DensificationController::vram_free_mb() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    auto err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        return -1.0f; // Unknown — don't skip.
    }
    return static_cast<float>(free_bytes) / (1024.0f * 1024.0f);
}

} // namespace cugs
