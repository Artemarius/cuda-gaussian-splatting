/// @file adam.cpp
/// @brief Adam optimizer implementation for 3D Gaussian Splatting.

#include "optimizer/adam.hpp"

#include <spdlog/spdlog.h>

#include <cassert>
#include <stdexcept>

namespace cugs {

GaussianAdam::GaussianAdam(GaussianModel& model, const AdamConfig& config)
    : model_(model), config_(config) {
    // Enable gradient tracking on all model tensors.
    model_.positions.requires_grad_(true);
    model_.sh_coeffs.requires_grad_(true);
    model_.opacities.requires_grad_(true);
    model_.scales.requires_grad_(true);
    model_.rotations.requires_grad_(true);

    // Build per-group parameter lists with individual learning rates.
    // Order: positions, sh_coeffs, opacities, scales, rotations
    // (matches ParamGroup enum ordering).
    auto make_group = [&](torch::Tensor& param, float lr) {
        auto options = std::make_unique<torch::optim::AdamOptions>(lr);
        options->betas({config.beta1, config.beta2});
        options->eps(config.eps);

        torch::optim::OptimizerParamGroup group({param}, std::move(options));
        return group;
    };

    std::vector<torch::optim::OptimizerParamGroup> groups;
    groups.push_back(make_group(model_.positions,  config.position_lr_config.lr_init));
    groups.push_back(make_group(model_.sh_coeffs,  config.lr_sh_coeffs));
    groups.push_back(make_group(model_.opacities,  config.lr_opacities));
    groups.push_back(make_group(model_.scales,     config.lr_scales));
    groups.push_back(make_group(model_.rotations,  config.lr_rotations));

    // Create Adam with default options (overridden per-group).
    auto default_options = torch::optim::AdamOptions(0.0);
    default_options.betas({config.beta1, config.beta2});
    default_options.eps(config.eps);

    optimizer_ = std::make_unique<torch::optim::Adam>(groups, default_options);
}

void GaussianAdam::apply_gradients(const BackwardOutput& grads) {
    // Copy custom CUDA gradients into the .grad() fields of each parameter.
    // This is the key bridge between our custom backward pass and libtorch's
    // optimizer.
    model_.positions.mutable_grad() = grads.dL_dpositions;
    model_.sh_coeffs.mutable_grad() = grads.dL_dsh_coeffs;
    model_.opacities.mutable_grad() = grads.dL_dopacities;
    model_.scales.mutable_grad()    = grads.dL_dscales;
    model_.rotations.mutable_grad() = grads.dL_drotations;
}

void GaussianAdam::update_lr(int step) {
    float lr = position_lr(step, config_.position_lr_config);
    auto& group = optimizer_->param_groups()[0];
    auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
    options.lr(lr);
}

void GaussianAdam::zero_grad() {
    optimizer_->zero_grad();
}

void GaussianAdam::step() {
    optimizer_->step();
}

float GaussianAdam::get_lr(ParamGroup group) const {
    int idx = static_cast<int>(group);
    assert(idx >= 0 && idx < static_cast<int>(optimizer_->param_groups().size()));
    const auto& options = static_cast<const torch::optim::AdamOptions&>(
        optimizer_->param_groups()[idx].options());
    return static_cast<float>(options.lr());
}

} // namespace cugs
