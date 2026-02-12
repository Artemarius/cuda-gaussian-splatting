#pragma once

/// @file lr_schedule.hpp
/// @brief Learning rate schedules for 3D Gaussian Splatting training.
///
/// Implements the per-parameter-group learning rates from Kerbl et al.
/// (SIGGRAPH 2023), Table 3:
///   - Position: exponential decay from lr_init to lr_final over max_steps
///   - SH coefficients: constant 2.5e-3
///   - Opacity: constant 0.05
///   - Scale: constant 5e-3
///   - Rotation: constant 1e-3
///
/// Also implements progressive SH activation: degree increases every 1000
/// iterations (0 -> 1 -> 2 -> 3).

#include <algorithm>
#include <cmath>

namespace cugs {

/// @brief Identifies the 5 parameter groups for per-group learning rates.
enum class ParamGroup {
    kPositions,
    kSHCoeffs,
    kOpacities,
    kScales,
    kRotations,
};

/// @brief Configuration for the position learning rate schedule.
///
/// Uses log-linear interpolation (exponential decay):
///   lr(t) = lr_init * exp(t / max_steps * log(lr_final / lr_init))
struct PositionLRConfig {
    float lr_init = 1.6e-4f;
    float lr_final = 1.6e-6f;
    int max_steps = 30000;
};

/// @brief Compute the position learning rate at a given step.
///
/// Log-linear interpolation between lr_init and lr_final.
/// Clamps to lr_final beyond max_steps.
///
/// @param step Current training iteration (0-based).
/// @param config Position LR configuration.
/// @return Learning rate for the current step.
inline float position_lr(int step, const PositionLRConfig& config) {
    if (step >= config.max_steps) return config.lr_final;
    if (step <= 0) return config.lr_init;

    // Log-linear interpolation: lr = init * (final/init)^(t/T)
    const float t = static_cast<float>(step) / static_cast<float>(config.max_steps);
    const float log_ratio = std::log(config.lr_final / config.lr_init);
    return config.lr_init * std::exp(t * log_ratio);
}

/// @brief Compute the active SH degree for a given training step.
///
/// Progressive SH activation: degree increases every 1000 iterations.
///   step 0..999   -> degree 0
///   step 1000..1999 -> degree 1
///   step 2000..2999 -> degree 2
///   step 3000+      -> degree 3 (or max_degree)
///
/// @param step Current training iteration.
/// @param max_degree Maximum SH degree the model was allocated for.
/// @return Active SH degree for this step.
inline int active_sh_degree_for_step(int step, int max_degree) {
    return std::min(step / 1000, max_degree);
}

/// @brief Default constant learning rates for non-position parameters.
namespace lr_defaults {
    constexpr float kSHCoeffs  = 2.5e-3f;
    constexpr float kOpacity   = 0.05f;
    constexpr float kScale     = 5e-3f;
    constexpr float kRotation  = 1e-3f;
} // namespace lr_defaults

} // namespace cugs
