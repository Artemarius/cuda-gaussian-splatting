#pragma once

/// @file fused_adam.hpp
/// @brief Fused CUDA Adam optimizer for 3D Gaussian Splatting training.
///
/// Replaces the libtorch-wrapped GaussianAdam with a single fused CUDA kernel
/// per parameter group. This eliminates ~50+ kernel launches per step (libtorch
/// Adam launches ~10+ kernels per group) down to exactly 5, reducing launch
/// overhead and improving memory bandwidth.
///
/// Public API is identical to GaussianAdam for drop-in replacement.

#include "core/gaussian.hpp"
#include "rasterizer/rasterizer.hpp"
#include "training/lr_schedule.hpp"
#include "optimizer/adam.hpp"

#include <torch/torch.h>

#include <array>

namespace cugs {

/// @brief Fused CUDA Adam optimizer for Gaussian model parameters.
///
/// Self-manages Adam state (first/second moment tensors) and gradient
/// references. Each step() call launches exactly 5 CUDA kernels (one per
/// parameter group), each performing the full Adam update in a single pass.
class FusedAdam {
public:
    /// @brief Construct the optimizer for a model.
    ///
    /// Allocates zero-initialized first/second moment tensors for all 5
    /// parameter groups and enables requires_grad on model tensors.
    ///
    /// @param model Reference to the GaussianModel to optimize.
    /// @param config Optimizer configuration.
    FusedAdam(GaussianModel& model, const AdamConfig& config = {});

    /// @brief Copy BackwardOutput gradients into internal gradient references.
    ///
    /// Stores tensor references (no copy) for use in the next step().
    ///
    /// @param grads BackwardOutput from render_backward().
    void apply_gradients(const BackwardOutput& grads);

    /// @brief Update the position learning rate for the current step.
    ///
    /// Adjusts the LR of the position parameter group using the exponential
    /// decay schedule.
    ///
    /// @param step Current training iteration.
    void update_lr(int step);

    /// @brief Zero all parameter gradients.
    void zero_grad();

    /// @brief Perform one fused Adam update step.
    ///
    /// Increments step count, computes bias correction factors on the host
    /// (double precision for early-step accuracy), then launches the fused
    /// kernel once per parameter group.
    void step();

    /// @brief Get the current learning rate for a parameter group.
    /// @param group Which parameter group to query.
    /// @return Current learning rate.
    float get_lr(ParamGroup group) const;

private:
    static constexpr int kNumGroups = 5;

    /// @brief Launch the fused Adam kernel for one parameter group.
    ///
    /// Validates tensor state, ensures contiguity, computes grid dimensions,
    /// launches k_fused_adam, and checks for CUDA errors.
    ///
    /// @param param The parameter tensor to update.
    /// @param grad The gradient tensor.
    /// @param m First moment tensor.
    /// @param v Second moment tensor.
    /// @param lr Learning rate for this group.
    /// @param bc1 Bias correction factor for first moment: 1/(1-beta1^t).
    /// @param bc2 Bias correction factor for second moment: 1/(1-beta2^t).
    void launch_kernel(
        torch::Tensor& param,
        const torch::Tensor& grad,
        torch::Tensor& m,
        torch::Tensor& v,
        float lr,
        float bc1,
        float bc2);

    GaussianModel& model_;
    AdamConfig config_;

    // Per-group Adam state.
    std::array<torch::Tensor, kNumGroups> m_;      ///< First moment estimates
    std::array<torch::Tensor, kNumGroups> v_;      ///< Second moment estimates
    std::array<torch::Tensor, kNumGroups> grads_;  ///< Gradient references

    // Per-group learning rates.
    std::array<float, kNumGroups> learning_rates_;

    int step_count_ = 0;
};

} // namespace cugs
