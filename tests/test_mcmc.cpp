/// @file test_mcmc.cpp
/// @brief Unit tests for Phase 10: MCMC densification.
///
/// Tests:
///   - Schedule boundaries (should_relocate)
///   - Noise LR decay (init/final/monotonic)
///   - Relocation fixes dead Gaussians (positions change, alive untouched, N constant)
///   - Relocate cap respected (num_relocated <= cap * N)
///   - Noise gate selectivity (low-opacity Gaussians get more noise)
///   - Regularization is scalar, positive, has autograd gradients
///   - Model valid after full cycle
///   - Constant N across multiple relocations

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>

#include "optimizer/mcmc_densification.hpp"
#include "core/gaussian.hpp"

namespace cugs {
namespace {

/// Helper: create a minimal valid GaussianModel on CUDA with N Gaussians.
GaussianModel make_mcmc_model(int n, float scale_val = -2.0f,
                               float opacity_val = 2.0f) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    GaussianModel model;
    model.positions = torch::randn({n, 3}, opts) * 0.5f;
    model.sh_coeffs = torch::randn({n, 3, 1}, opts) * 0.1f;
    model.opacities = torch::full({n, 1}, opacity_val, opts);
    model.rotations = torch::randn({n, 4}, opts);
    model.rotations = model.rotations /
        model.rotations.norm(2, 1, true).clamp_min(1e-8f);
    model.scales = torch::full({n, 3}, scale_val, opts);

    return model;
}

// ===========================================================================
// Schedule Tests
// ===========================================================================

TEST(MCMCTest, ShouldRelocateBoundaries) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    MCMCConfig config;
    config.relocate_from  = 500;
    config.relocate_until = 15000;
    config.relocate_every = 100;
    MCMCController ctrl(config, 10.0f);

    // Before start.
    EXPECT_FALSE(ctrl.should_relocate(0));
    EXPECT_FALSE(ctrl.should_relocate(100));
    EXPECT_FALSE(ctrl.should_relocate(400));
    EXPECT_FALSE(ctrl.should_relocate(499));

    // At start boundary.
    EXPECT_TRUE(ctrl.should_relocate(500));

    // Within range, on frequency.
    EXPECT_TRUE(ctrl.should_relocate(600));
    EXPECT_TRUE(ctrl.should_relocate(1000));
    EXPECT_TRUE(ctrl.should_relocate(14900));
    EXPECT_TRUE(ctrl.should_relocate(15000));

    // Within range, off frequency.
    EXPECT_FALSE(ctrl.should_relocate(501));
    EXPECT_FALSE(ctrl.should_relocate(550));
    EXPECT_FALSE(ctrl.should_relocate(999));

    // After end.
    EXPECT_FALSE(ctrl.should_relocate(15100));
    EXPECT_FALSE(ctrl.should_relocate(20000));
}

// ===========================================================================
// Noise LR Decay Tests
// ===========================================================================

TEST(MCMCTest, NoiseLRInitAndFinal) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    MCMCConfig config;
    config.noise_lr_init  = 5e5f;
    config.noise_lr_final = 1e3f;
    config.noise_lr_max_steps = 30000;
    MCMCController ctrl(config, 10.0f);

    // At step 0: should be init.
    EXPECT_FLOAT_EQ(ctrl.noise_lr(0), config.noise_lr_init);

    // At/beyond max_steps: should be final.
    EXPECT_FLOAT_EQ(ctrl.noise_lr(30000), config.noise_lr_final);
    EXPECT_FLOAT_EQ(ctrl.noise_lr(50000), config.noise_lr_final);
}

TEST(MCMCTest, NoiseLRMonotonicDecay) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    MCMCConfig config;
    config.noise_lr_init  = 5e5f;
    config.noise_lr_final = 1e3f;
    config.noise_lr_max_steps = 30000;
    MCMCController ctrl(config, 10.0f);

    float prev = ctrl.noise_lr(0);
    for (int step = 1000; step <= 30000; step += 1000) {
        float curr = ctrl.noise_lr(step);
        EXPECT_LT(curr, prev) << "Noise LR should decrease at step " << step;
        prev = curr;
    }
}

// ===========================================================================
// Relocation Tests
// ===========================================================================

TEST(MCMCTest, RelocationFixesDeadGaussians) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    const int n = 20;
    auto model = make_mcmc_model(n);

    {
        torch::NoGradGuard no_grad;
        // First 10: alive (high opacity). sigmoid(5.0) ~ 0.993.
        model.opacities.slice(0, 0, 10).fill_(5.0f);
        // Last 10: dead (low opacity). sigmoid(-8.0) ~ 0.0003.
        model.opacities.slice(0, 10, 20).fill_(-8.0f);
    }

    // Save alive positions before relocation.
    auto alive_pos_before = model.positions.slice(0, 0, 10).clone();
    auto dead_pos_before = model.positions.slice(0, 10, 20).clone();

    MCMCConfig config;
    config.dead_opacity_threshold = 0.005f;
    config.relocate_cap = 1.0f; // Allow relocating all dead.
    MCMCController ctrl(config, 10.0f);

    auto stats = ctrl.relocate(model, 500);

    // N stays constant.
    EXPECT_EQ(model.num_gaussians(), n);

    // Dead Gaussians were relocated.
    EXPECT_EQ(stats.num_relocated, 10);
    EXPECT_EQ(stats.num_dead, 10);
    EXPECT_EQ(stats.num_total, n);

    // Dead positions should have changed.
    auto dead_pos_after = model.positions.slice(0, 10, 20);
    EXPECT_FALSE(torch::allclose(dead_pos_before, dead_pos_after));

    // Alive positions should be untouched.
    auto alive_pos_after = model.positions.slice(0, 0, 10);
    EXPECT_TRUE(torch::equal(alive_pos_before, alive_pos_after));

    EXPECT_TRUE(model.is_valid());
}

TEST(MCMCTest, RelocateCapRespected) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    const int n = 100;
    auto model = make_mcmc_model(n);

    {
        torch::NoGradGuard no_grad;
        // 80 alive, 20 dead.
        model.opacities.slice(0, 0, 80).fill_(5.0f);
        model.opacities.slice(0, 80, 100).fill_(-8.0f);
    }

    MCMCConfig config;
    config.dead_opacity_threshold = 0.005f;
    config.relocate_cap = 0.05f; // Cap at 5% of 100 = 5.
    MCMCController ctrl(config, 10.0f);

    auto stats = ctrl.relocate(model, 500);

    // Only 5 should be relocated despite 20 being dead.
    EXPECT_EQ(stats.num_relocated, 5);
    EXPECT_EQ(stats.num_dead, 20);
    EXPECT_EQ(model.num_gaussians(), n);
    EXPECT_TRUE(model.is_valid());
}

TEST(MCMCTest, RelocationWithNoDeadIsNoop) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    const int n = 10;
    // All high opacity: sigmoid(5.0) ~ 0.993 > 0.005.
    auto model = make_mcmc_model(n, -2.0f, 5.0f);

    auto pos_before = model.positions.clone();

    MCMCConfig config;
    config.dead_opacity_threshold = 0.005f;
    MCMCController ctrl(config, 10.0f);

    auto stats = ctrl.relocate(model, 500);

    EXPECT_EQ(stats.num_relocated, 0);
    EXPECT_EQ(stats.num_dead, 0);
    EXPECT_TRUE(torch::equal(pos_before, model.positions));
}

// ===========================================================================
// Noise Injection Tests
// ===========================================================================

TEST(MCMCTest, NoiseGateSelectivity) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    // Create model with a mix of low and high opacity.
    const int n = 100;
    auto model = make_mcmc_model(n);

    {
        torch::NoGradGuard no_grad;
        // First half: high opacity (gate ~ 0 => less noise).
        model.opacities.slice(0, 0, 50).fill_(10.0f);
        // Second half: low opacity (gate ~ 1 => more noise).
        model.opacities.slice(0, 50, 100).fill_(-10.0f);
        // Same scale for all.
        model.scales.fill_(-2.0f);
    }

    auto pos_before = model.positions.clone();

    MCMCConfig config;
    config.noise_lr_init = 1.0f; // Moderate noise for test visibility.
    config.noise_lr_final = 1.0f;
    config.noise_gate_k = 100.0f;
    config.noise_gate_t = 0.995f;
    MCMCController ctrl(config, 10.0f);

    // Run noise injection multiple times to accumulate displacement.
    for (int i = 0; i < 10; ++i) {
        ctrl.inject_noise(model, 0);
    }

    // Compute displacement for each half.
    auto displacement = (model.positions - pos_before).norm(2, 1); // [N]
    auto high_opa_disp = displacement.slice(0, 0, 50).mean().item<float>();
    auto low_opa_disp = displacement.slice(0, 50, 100).mean().item<float>();

    // Low-opacity Gaussians should have moved more than high-opacity ones.
    EXPECT_GT(low_opa_disp, high_opa_disp * 2.0f)
        << "Low-opacity Gaussians should receive significantly more noise";
}

TEST(MCMCTest, NoiseInjectionModifiesPositions) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    const int n = 10;
    auto model = make_mcmc_model(n, -2.0f, 0.0f); // Mid opacity.

    auto pos_before = model.positions.clone();

    MCMCConfig config;
    config.noise_lr_init = 1e4f;
    MCMCController ctrl(config, 10.0f);

    ctrl.inject_noise(model, 0);

    // Positions should have changed.
    EXPECT_FALSE(torch::allclose(pos_before, model.positions));

    // Other properties should be unchanged.
    EXPECT_TRUE(model.is_valid());
}

// ===========================================================================
// Regularization Tests
// ===========================================================================

TEST(MCMCTest, RegularizationIsPositiveScalar) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    const int n = 20;
    auto model = make_mcmc_model(n);

    MCMCConfig config;
    config.lambda_opacity = 0.01f;
    config.lambda_scale = 0.01f;
    MCMCController ctrl(config, 10.0f);

    torch::Tensor reg_dL_dopa, reg_dL_dscales;
    float reg_val = ctrl.compute_regularization(model, reg_dL_dopa, reg_dL_dscales);

    // Should be a positive scalar.
    EXPECT_GT(reg_val, 0.0f);

    // Gradients should be defined and have correct shapes.
    EXPECT_TRUE(reg_dL_dopa.defined());
    EXPECT_TRUE(reg_dL_dscales.defined());
    EXPECT_EQ(reg_dL_dopa.sizes(), model.opacities.sizes());
    EXPECT_EQ(reg_dL_dscales.sizes(), model.scales.sizes());

    // Gradients should be non-zero.
    EXPECT_GT(reg_dL_dopa.abs().sum().item<float>(), 0.0f);
    EXPECT_GT(reg_dL_dscales.abs().sum().item<float>(), 0.0f);
}

// ===========================================================================
// Full Cycle Tests
// ===========================================================================

TEST(MCMCTest, ModelValidAfterFullCycle) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    const int n = 50;
    auto model = make_mcmc_model(n);

    {
        torch::NoGradGuard no_grad;
        // Mix of alive and dead.
        model.opacities.slice(0, 0, 40).fill_(3.0f);  // Alive.
        model.opacities.slice(0, 40, 50).fill_(-8.0f); // Dead.
    }

    MCMCConfig config;
    config.relocate_cap = 0.1f;
    config.noise_lr_init = 1e3f;
    config.noise_lr_final = 1e1f;
    config.lambda_opacity = 0.01f;
    config.lambda_scale = 0.01f;
    MCMCController ctrl(config, 10.0f);

    // Simulate a mini training cycle.
    for (int step = 0; step < 10; ++step) {
        // Noise every iteration.
        ctrl.inject_noise(model, step);

        // Regularization.
        torch::Tensor reg_dL_dopa, reg_dL_dscales;
        ctrl.compute_regularization(model, reg_dL_dopa, reg_dL_dscales);
    }

    // Relocate.
    auto stats = ctrl.relocate(model, 500);

    // Model should remain valid with constant N.
    EXPECT_TRUE(model.is_valid());
    EXPECT_EQ(model.num_gaussians(), n);
    EXPECT_GT(stats.num_relocated, 0);
}

TEST(MCMCTest, ConstantNAcrossMultipleRelocations) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    const int n = 30;
    auto model = make_mcmc_model(n);

    {
        torch::NoGradGuard no_grad;
        model.opacities.slice(0, 0, 20).fill_(3.0f);
        model.opacities.slice(0, 20, 30).fill_(-8.0f);
    }

    MCMCConfig config;
    config.relocate_cap = 1.0f; // Allow relocating all dead.
    MCMCController ctrl(config, 10.0f);

    // Multiple relocation steps.
    for (int i = 0; i < 5; ++i) {
        ctrl.relocate(model, 500 + i * 100);
        EXPECT_EQ(model.num_gaussians(), n)
            << "N should remain constant after relocation " << i;
        EXPECT_TRUE(model.is_valid())
            << "Model should be valid after relocation " << i;
    }
}

} // namespace
} // namespace cugs
