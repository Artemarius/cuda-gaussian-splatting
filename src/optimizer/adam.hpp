#pragma once

/// @file adam.hpp
/// @brief Adam optimizer wrapper for 3D Gaussian Splatting training.
///
/// Bridges the custom CUDA backward pass with libtorch's Adam optimizer
/// by manually injecting gradients into parameter .grad() fields.
///
/// The hybrid gradient flow:
///   render()                     -> RenderOutput (no autograd)
///   color.clone().requires_grad_ -> autograd-tracked copy
///   combined_loss(color, target) -> scalar loss
///   loss.backward()              -> dL/dcolor via libtorch autograd
///   render_backward(dL_dcolor)   -> BackwardOutput (custom CUDA gradients)
///   apply_gradients(backward)    -> copies into .grad() fields
///   step()                       -> Adam update

#include "core/gaussian.hpp"
#include "rasterizer/rasterizer.hpp"
#include "training/lr_schedule.hpp"

#include <torch/torch.h>

#include <memory>
#include <vector>

namespace cugs {

/// @brief Configuration for the Gaussian Adam optimizer.
struct AdamConfig {
    PositionLRConfig position_lr_config;

    float lr_sh_coeffs  = lr_defaults::kSHCoeffs;
    float lr_opacities  = lr_defaults::kOpacity;
    float lr_scales     = lr_defaults::kScale;
    float lr_rotations  = lr_defaults::kRotation;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps   = 1e-15f;  // paper uses very small epsilon
};

/// @brief Adam optimizer for Gaussian model parameters.
///
/// Wraps torch::optim::Adam with 5 parameter groups (one per model tensor)
/// and handles the gradient injection from BackwardOutput.
class GaussianAdam {
public:
    /// @brief Construct the optimizer for a model.
    ///
    /// Enables requires_grad on all model tensors and creates a
    /// torch::optim::Adam with per-group learning rates.
    ///
    /// @param model Reference to the GaussianModel to optimize.
    /// @param config Optimizer configuration.
    GaussianAdam(GaussianModel& model, const AdamConfig& config = {});

    /// @brief Copy BackwardOutput gradients into parameter .grad() fields.
    ///
    /// This bridges the custom CUDA backward pass with libtorch's Adam.
    /// Must be called before step().
    ///
    /// @param grads BackwardOutput from render_backward().
    void apply_gradients(const BackwardOutput& grads);

    /// @brief Update the position learning rate for the current step.
    ///
    /// Adjusts the LR of the first parameter group (positions) using
    /// the exponential decay schedule.
    ///
    /// @param step Current training iteration.
    void update_lr(int step);

    /// @brief Zero all parameter gradients.
    void zero_grad();

    /// @brief Perform one Adam update step.
    void step();

    /// @brief Get the current learning rate for a parameter group.
    /// @param group Which parameter group to query.
    /// @return Current learning rate.
    float get_lr(ParamGroup group) const;

private:
    GaussianModel& model_;
    AdamConfig config_;
    std::unique_ptr<torch::optim::Adam> optimizer_;
};

} // namespace cugs
