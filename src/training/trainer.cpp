/// @file trainer.cpp
/// @brief Main training loop implementation for 3D Gaussian Splatting.

#include "training/trainer.hpp"
#include "training/loss.hpp"
#include "rasterizer/rasterizer.hpp"
#include "core/gaussian_init.hpp"

#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <filesystem>

namespace cugs {

// ---------------------------------------------------------------------------
// image_to_tensor
// ---------------------------------------------------------------------------

torch::Tensor image_to_tensor(const Image& image, torch::Device device) {
    assert(image.valid());
    assert(image.channels == 3 || image.channels == 4);

    const int h = image.height;
    const int w = image.width;
    const int c = image.channels;

    // Wrap the raw float data as a CPU tensor. from_blob does NOT own
    // the data, so we immediately clone to get an owned copy.
    auto cpu_tensor = torch::from_blob(
        const_cast<float*>(image.data.data()),
        {h, w, c},
        torch::kFloat32
    ).clone();

    // Discard alpha channel if present: [H, W, 4] -> [H, W, 3]
    if (c == 4) {
        cpu_tensor = cpu_tensor.index({"...", torch::indexing::Slice(0, 3)}).contiguous();
    }

    return cpu_tensor.to(device);
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------

Trainer::Trainer(const TrainConfig& config)
    : config_(config)
    , dataset_(config.data_path, config.resolution_scale)
    , rng_(config.seed) {

    spdlog::info("=== 3D Gaussian Splatting Trainer ===");
    dataset_.print_summary();

    // Initialize Gaussians from sparse SfM points.
    model_ = init_gaussians_from_sparse(
        dataset_.sparse_points(),
        config_.max_sh_degree);

    // Enforce max_gaussians cap if set.
    if (config_.max_gaussians > 0 &&
        model_.num_gaussians() > config_.max_gaussians) {
        spdlog::info("Capping Gaussians from {} to {}",
                     model_.num_gaussians(), config_.max_gaussians);

        const int64_t n = config_.max_gaussians;
        model_.positions  = model_.positions.slice(0, 0, n).contiguous();
        model_.sh_coeffs  = model_.sh_coeffs.slice(0, 0, n).contiguous();
        model_.opacities  = model_.opacities.slice(0, 0, n).contiguous();
        model_.rotations  = model_.rotations.slice(0, 0, n).contiguous();
        model_.scales     = model_.scales.slice(0, 0, n).contiguous();
    }

    spdlog::info("Model: {} Gaussians, SH degree {}",
                 model_.num_gaussians(), model_.max_sh_degree());

    // Move model to CUDA.
    model_.to_device(torch::kCUDA);

    // Create optimizer.
    optimizer_ = std::make_unique<GaussianAdam>(model_, config_.adam);

    // Compute effective VRAM limit.
    effective_vram_limit_ = compute_effective_vram_limit(config_.memory);
    spdlog::info("Memory safety: VRAM limit = {:.0f} MB{}",
                 effective_vram_limit_,
                 config_.memory.vram_limit_mb > 0.0f ? " (user-set)" : " (auto)");

    // Create densification controller.
    if (!config_.no_densify) {
        config_.densification.max_gaussians = config_.max_gaussians;
        config_.densification.effective_vram_limit_mb = effective_vram_limit_;
        densify_ctrl_ = std::make_unique<DensificationController>(
            config_.densification, dataset_.scene_bounds().extent);
        spdlog::info("Densification: from={} until={} every={} grad_thresh={:.1e}",
                     config_.densification.densify_from,
                     config_.densification.densify_until,
                     config_.densification.densify_every,
                     config_.densification.grad_threshold);
    } else {
        spdlog::info("Densification: disabled");
    }

    // Create output directory.
    std::filesystem::create_directories(config_.output_path);

    log_memory_status(effective_vram_limit_);
}

void Trainer::train() {
    spdlog::info("Starting training for {} iterations", config_.max_iterations);

    auto t_start = std::chrono::steady_clock::now();

    for (int step = 0; step < config_.max_iterations; ++step) {
        // Per-iteration VRAM safety check.
        if (!check_vram_safety(step)) {
            spdlog::error("Aborting training at step {} due to critical VRAM shortage", step);
            save_checkpoint(step);
            return;
        }

        auto stats = train_step(step);

        // Logging.
        if (step % config_.log_every == 0 || step == config_.max_iterations - 1) {
            auto elapsed = std::chrono::steady_clock::now() - t_start;
            auto secs = std::chrono::duration<double>(elapsed).count();
            float its = (step + 1) / secs;

            spdlog::info(
                "[{:>5}/{}] loss={:.4f} l1={:.4f} ssim={:.4f} "
                "n={} sh={} lr={:.2e} ({:.1f} it/s)",
                step, config_.max_iterations,
                stats.loss, stats.l1, stats.ssim,
                stats.num_gaussians, stats.active_sh_degree,
                stats.position_lr, its);
        }

        // Checkpointing.
        if (config_.save_every > 0 &&
            (step > 0 && step % config_.save_every == 0)) {
            save_checkpoint(step);
        }

        // Periodic memory logging (every 1000 iterations).
        if (step > 0 && step % 1000 == 0) {
            log_memory_status(effective_vram_limit_);
        }
    }

    // Final checkpoint.
    save_checkpoint(config_.max_iterations);

    auto elapsed = std::chrono::steady_clock::now() - t_start;
    auto secs = std::chrono::duration<double>(elapsed).count();
    spdlog::info("Training complete in {:.1f}s ({:.1f} it/s)",
                 secs, config_.max_iterations / secs);
}

IterationStats Trainer::train_step(int step) {
    // 1. Update learning rate.
    optimizer_->update_lr(step);

    // 2. Determine active SH degree.
    int sh_degree = active_sh_degree_for_step(step, model_.max_sh_degree());

    // 3. Sample a random training image.
    std::uniform_int_distribution<size_t> dist(0, dataset_.num_train() - 1);
    size_t img_idx = dist(rng_);
    const auto& camera = dataset_.train_cameras()[img_idx];
    Image cpu_image = dataset_.load_train_image(img_idx);

    // Resize if the loaded image doesn't match camera dimensions (can happen
    // when COLMAP cameras.bin records the original capture resolution but the
    // image files on disk are already downscaled).
    if (cpu_image.width != camera.width || cpu_image.height != camera.height) {
        cpu_image = resize_image(cpu_image, camera.width, camera.height);
    }

    auto target = image_to_tensor(cpu_image, torch::kCUDA);

    // 4. Render settings.
    RenderSettings settings;
    settings.active_sh_degree = sh_degree;
    if (config_.random_background) {
        auto bg = torch::rand({3}, torch::kFloat32);
        settings.background[0] = bg[0].item<float>();
        settings.background[1] = bg[1].item<float>();
        settings.background[2] = bg[2].item<float>();
    }

    // 5. Forward pass.
    auto render_out = render(model_, camera, settings);

    // 6. Compute loss via libtorch autograd for dL/dcolor.
    auto rendered = render_out.color.clone().detach().requires_grad_(true);
    auto loss_val = combined_loss(rendered, target, config_.lambda_ssim);
    loss_val.backward();
    auto dL_dcolor = rendered.grad().clone();

    // Also compute individual loss components for logging.
    float l1_val, ssim_val;
    {
        torch::NoGradGuard no_grad;
        l1_val = cugs::l1_loss(render_out.color, target).item<float>();
        ssim_val = ssim(render_out.color, target).mean().item<float>();
    }

    // 7. Custom CUDA backward pass.
    auto grads = render_backward(dL_dcolor, render_out, model_, camera, settings);

    // 8. Inject gradients and step.
    optimizer_->zero_grad();
    optimizer_->apply_gradients(grads);
    optimizer_->step();

    // 9. Adaptive density control.
    IterationStats stats;
    if (densify_ctrl_) {
        // Accumulate 2D screen-space gradients every iteration.
        // The densification metric is ||dL/d(screen_xy)||_2, matching the
        // reference implementation (not the 3D world-space gradient).
        densify_ctrl_->accumulate_gradients(
            grads.dL_dmeans_2d, render_out.radii);

        // Densify on schedule.
        if (densify_ctrl_->should_densify(step)) {
            auto dstats = densify_ctrl_->densify(model_, step);
            stats.densified = true;
            stats.num_cloned = dstats.num_cloned;
            stats.num_split  = dstats.num_split;
            stats.num_pruned = dstats.num_pruned;

            if (dstats.num_cloned > 0 || dstats.num_split > 0 ||
                dstats.num_pruned > 0) {
                // Model changed size — rebuild optimizer (Adam moments invalid).
                optimizer_ = std::make_unique<GaussianAdam>(
                    model_, config_.adam);
                optimizer_->update_lr(step);

                // Release cached CUDA memory from old optimizer/tensors.
                c10::cuda::CUDACachingAllocator::emptyCache();

                spdlog::info(
                    "[densify {:>5}] cloned={} split={} pruned={} "
                    "{} -> {} Gaussians{}",
                    step, dstats.num_cloned, dstats.num_split,
                    dstats.num_pruned, dstats.num_before, dstats.num_after,
                    dstats.skipped_vram ? " (VRAM-limited)" : "");
            }
        }

        // Opacity reset on schedule.
        if (densify_ctrl_->should_reset_opacity(step)) {
            densify_ctrl_->reset_opacity(model_);
            spdlog::info("[opacity reset {:>5}]", step);
        }
    }

    // 10. Collect stats.
    stats.iteration = step;
    stats.loss = loss_val.item<float>();
    stats.l1 = l1_val;
    stats.ssim = ssim_val;
    stats.num_gaussians = static_cast<int>(model_.num_gaussians());
    stats.active_sh_degree = sh_degree;
    stats.position_lr = optimizer_->get_lr(ParamGroup::kPositions);

    return stats;
}

void Trainer::save_checkpoint(int step) {
    auto filename = config_.output_path /
        ("point_cloud_" + std::to_string(step) + ".ply");

    spdlog::info("Saving checkpoint: {}", filename.string());

    // save_ply moves tensors to CPU internally.
    if (!model_.save_ply(filename)) {
        spdlog::error("Failed to save checkpoint: {}", filename.string());
    }
}

void Trainer::log_vram_usage() {
    auto info = vram_info_mb();
    if (info.valid()) {
        float budget = effective_vram_limit_ - info.used_mb();
        spdlog::info("VRAM: {:.0f} / {:.0f} MB used | budget: {:.0f} MB | limit: {:.0f} MB",
                     info.used_mb(), info.total_mb, budget, effective_vram_limit_);
    }
    float ram = system_ram_available_mb();
    if (ram >= 0.0f) {
        spdlog::info("RAM:  {:.0f} MB available", ram);
    }
}

bool Trainer::check_vram_safety(int step) {
    auto info = vram_info_mb();
    if (!info.valid()) return true;  // Can't query — assume OK.

    float budget = effective_vram_limit_ - info.used_mb();

    // Check system RAM.
    float ram = system_ram_available_mb();
    if (ram >= 0.0f && ram < config_.memory.ram_warning_mb) {
        spdlog::warn("[step {}] Low system RAM: {:.0f} MB available", step, ram);
    }

    // Check VRAM critical threshold.
    if (budget < config_.memory.vram_critical_mb) {
        vram_critical_streak_++;
        spdlog::warn("[step {}] VRAM critical: {:.0f} MB budget ({}/{} before abort)",
                     step, budget, vram_critical_streak_,
                     config_.memory.vram_critical_count);
        if (vram_critical_streak_ >= config_.memory.vram_critical_count) {
            return false;  // Signal abort.
        }
    } else {
        // Recovered — reset streak.
        if (vram_critical_streak_ > 0) {
            spdlog::info("[step {}] VRAM recovered: {:.0f} MB budget", step, budget);
        }
        vram_critical_streak_ = 0;
    }

    return true;
}

} // namespace cugs
