/// @file fused_adam.cu
/// @brief Fused CUDA Adam optimizer kernel and class implementation.
///
/// The k_fused_adam kernel performs the complete Adam update for one parameter
/// group in a single kernel launch: read gradient, update first/second moments,
/// compute bias-corrected parameter update. This replaces ~10+ kernel launches
/// that libtorch's Adam uses per parameter group.

#include "optimizer/fused_adam.hpp"
#include "utils/cuda_utils.cuh"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace cugs {

// ---------------------------------------------------------------------------
// Fused Adam CUDA Kernel
// ---------------------------------------------------------------------------

/// @brief Fused Adam update kernel — one thread per float element.
///
/// Grid/block: 1D, 256 threads per block, ceil(N / 256) blocks.
///
/// For each element i:
///   m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
///   v[i] = beta2 * v[i] + (1 - beta2) * grad[i]^2
///   m_hat = m[i] * bc1    (bc1 = 1 / (1 - beta1^t), precomputed on host)
///   v_hat = v[i] * bc2    (bc2 = 1 / (1 - beta2^t), precomputed on host)
///   param[i] -= lr * m_hat / (sqrt(v_hat) + eps)
///
/// @param param  Parameter tensor data (read-write).
/// @param grad   Gradient tensor data (read-only).
/// @param m      First moment tensor data (read-write).
/// @param v      Second moment tensor data (read-write).
/// @param lr     Learning rate.
/// @param beta1  First moment decay rate.
/// @param beta2  Second moment decay rate.
/// @param eps    Epsilon for numerical stability.
/// @param bc1    Bias correction for m: 1 / (1 - beta1^t).
/// @param bc2    Bias correction for v: 1 / (1 - beta2^t).
/// @param n      Total number of float elements.
__global__ void k_fused_adam(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bc1,
    float bc2,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float g = grad[i];

    // Update biased first moment estimate.
    float mi = beta1 * m[i] + (1.0f - beta1) * g;
    m[i] = mi;

    // Update biased second moment estimate.
    float vi = beta2 * v[i] + (1.0f - beta2) * g * g;
    v[i] = vi;

    // Bias-corrected estimates.
    const float m_hat = mi * bc1;
    const float v_hat = vi * bc2;

    // Parameter update.
    param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// ---------------------------------------------------------------------------
// FusedAdam Class Implementation
// ---------------------------------------------------------------------------

FusedAdam::FusedAdam(GaussianModel& model, const AdamConfig& config)
    : model_(model), config_(config), step_count_(0) {

    // Enable gradient tracking on all model tensors.
    model_.positions.requires_grad_(true);
    model_.sh_coeffs.requires_grad_(true);
    model_.opacities.requires_grad_(true);
    model_.scales.requires_grad_(true);
    model_.rotations.requires_grad_(true);

    // Allocate zero-initialized moment tensors.
    // Order matches ParamGroup enum: positions, sh_coeffs, opacities, scales, rotations.
    torch::Tensor* params[kNumGroups] = {
        &model_.positions, &model_.sh_coeffs, &model_.opacities,
        &model_.scales, &model_.rotations
    };

    for (int i = 0; i < kNumGroups; ++i) {
        m_[i] = torch::zeros_like(*params[i]);
        v_[i] = torch::zeros_like(*params[i]);
        grads_[i] = torch::Tensor();  // Empty until apply_gradients().
    }

    // Set initial per-group learning rates.
    learning_rates_[0] = config.position_lr_config.lr_init;
    learning_rates_[1] = config.lr_sh_coeffs;
    learning_rates_[2] = config.lr_opacities;
    learning_rates_[3] = config.lr_scales;
    learning_rates_[4] = config.lr_rotations;
}

void FusedAdam::apply_gradients(const BackwardOutput& grads) {
    // Store tensor references (no copy).
    grads_[0] = grads.dL_dpositions;
    grads_[1] = grads.dL_dsh_coeffs;
    grads_[2] = grads.dL_dopacities;
    grads_[3] = grads.dL_dscales;
    grads_[4] = grads.dL_drotations;
}

void FusedAdam::update_lr(int step) {
    learning_rates_[0] = position_lr(step, config_.position_lr_config);
}

void FusedAdam::zero_grad() {
    // Clear gradient references.
    for (int i = 0; i < kNumGroups; ++i) {
        grads_[i] = torch::Tensor();
    }

    // Zero .grad() fields on model tensors (in case anything else wrote them).
    if (model_.positions.grad().defined())  model_.positions.mutable_grad().zero_();
    if (model_.sh_coeffs.grad().defined())  model_.sh_coeffs.mutable_grad().zero_();
    if (model_.opacities.grad().defined())  model_.opacities.mutable_grad().zero_();
    if (model_.scales.grad().defined())     model_.scales.mutable_grad().zero_();
    if (model_.rotations.grad().defined())  model_.rotations.mutable_grad().zero_();
}

void FusedAdam::step() {
    step_count_++;

    // Compute bias correction factors on the host in double precision.
    // This matters for early steps where beta^t is close to 1.
    const double beta1 = static_cast<double>(config_.beta1);
    const double beta2 = static_cast<double>(config_.beta2);
    const double bc1 = 1.0 / (1.0 - std::pow(beta1, step_count_));
    const double bc2 = 1.0 / (1.0 - std::pow(beta2, step_count_));

    // Launch one kernel per parameter group.
    torch::Tensor* params[kNumGroups] = {
        &model_.positions, &model_.sh_coeffs, &model_.opacities,
        &model_.scales, &model_.rotations
    };

    for (int i = 0; i < kNumGroups; ++i) {
        if (!grads_[i].defined()) continue;
        launch_kernel(
            *params[i], grads_[i], m_[i], v_[i],
            learning_rates_[i],
            static_cast<float>(bc1),
            static_cast<float>(bc2));
    }
}

float FusedAdam::get_lr(ParamGroup group) const {
    int idx = static_cast<int>(group);
    assert(idx >= 0 && idx < kNumGroups);
    return learning_rates_[idx];
}

void FusedAdam::launch_kernel(
    torch::Tensor& param,
    const torch::Tensor& grad,
    torch::Tensor& m,
    torch::Tensor& v,
    float lr,
    float bc1,
    float bc2)
{
    // Validate tensors.
    TORCH_CHECK(param.is_cuda(), "FusedAdam: param must be on CUDA");
    TORCH_CHECK(grad.is_cuda(), "FusedAdam: grad must be on CUDA");
    TORCH_CHECK(param.numel() == grad.numel(),
        "FusedAdam: param/grad size mismatch: ", param.numel(), " vs ", grad.numel());

    // Ensure contiguity.
    auto param_c = param.contiguous();
    auto grad_c = grad.contiguous();
    auto m_c = m.contiguous();
    auto v_c = v.contiguous();

    const int n = static_cast<int>(param_c.numel());
    if (n == 0) return;

    constexpr int kBlockSize = 256;
    const int grid_size = (n + kBlockSize - 1) / kBlockSize;

    k_fused_adam<<<grid_size, kBlockSize>>>(
        param_c.data_ptr<float>(),
        grad_c.data_ptr<float>(),
        m_c.data_ptr<float>(),
        v_c.data_ptr<float>(),
        lr,
        config_.beta1,
        config_.beta2,
        config_.eps,
        bc1,
        bc2,
        n);

    CUDA_CHECK(cudaGetLastError());

    // If contiguous() made a copy, write back. In practice these tensors are
    // already contiguous so this is a no-op.
    if (!param.is_contiguous()) param.copy_(param_c);
    if (!m.is_contiguous()) m.copy_(m_c);
    if (!v.is_contiguous()) v.copy_(v_c);
}

} // namespace cugs
