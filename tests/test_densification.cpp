/// @file test_densification.cpp
/// @brief Unit tests for Phase 7: adaptive density control.
///
/// Tests:
///   - Schedule boundaries (densify_from, densify_until, densify_every)
///   - Opacity reset schedule
///   - Gradient accumulation
///   - Invisible Gaussians not accumulated
///   - Clone path (high grad, small scale)
///   - Split path (high grad, large scale)
///   - Prune path (low opacity)
///   - Opacity reset sets low value
///   - Model validity after full cycle
///   - Max Gaussians cap enforced

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>

#include "optimizer/densification.hpp"
#include "core/gaussian.hpp"

namespace cugs {
namespace {

/// Helper: create a minimal valid GaussianModel on CUDA with N Gaussians.
GaussianModel make_test_model(int n, float scale_val = -2.0f,
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

TEST(DensificationTest, ShouldDensifyBoundaries) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    DensificationConfig config;
    config.densify_from  = 500;
    config.densify_until = 15000;
    config.densify_every = 100;
    DensificationController ctrl(config, 10.0f);

    // Before start.
    EXPECT_FALSE(ctrl.should_densify(0));
    EXPECT_FALSE(ctrl.should_densify(100));
    EXPECT_FALSE(ctrl.should_densify(400));
    EXPECT_FALSE(ctrl.should_densify(499));

    // At start boundary.
    EXPECT_TRUE(ctrl.should_densify(500));

    // Within range, on frequency.
    EXPECT_TRUE(ctrl.should_densify(600));
    EXPECT_TRUE(ctrl.should_densify(1000));
    EXPECT_TRUE(ctrl.should_densify(14900));
    EXPECT_TRUE(ctrl.should_densify(15000));

    // Within range, off frequency.
    EXPECT_FALSE(ctrl.should_densify(501));
    EXPECT_FALSE(ctrl.should_densify(550));
    EXPECT_FALSE(ctrl.should_densify(999));

    // After end.
    EXPECT_FALSE(ctrl.should_densify(15100));
    EXPECT_FALSE(ctrl.should_densify(20000));
}

TEST(DensificationTest, ShouldResetOpacity) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    DensificationConfig config;
    config.densify_from = 500;
    config.opacity_reset_every = 3000;
    DensificationController ctrl(config, 10.0f);

    // Before densify_from: no reset.
    EXPECT_FALSE(ctrl.should_reset_opacity(0));

    // At multiples of 3000 and >= densify_from.
    EXPECT_TRUE(ctrl.should_reset_opacity(3000));
    EXPECT_TRUE(ctrl.should_reset_opacity(6000));
    EXPECT_TRUE(ctrl.should_reset_opacity(9000));

    // Not at multiple.
    EXPECT_FALSE(ctrl.should_reset_opacity(3001));
    EXPECT_FALSE(ctrl.should_reset_opacity(4000));
}

// ===========================================================================
// Gradient Accumulation Tests
// ===========================================================================

TEST(DensificationTest, AccumulatesGradients) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    DensificationConfig config;
    config.densify_from  = 0;
    config.densify_until = 1000;
    config.densify_every = 5;
    DensificationController ctrl(config, 10.0f);

    const int n = 10;
    auto model = make_test_model(n);

    // Simulate 5 gradient accumulation steps.
    for (int i = 0; i < 5; ++i) {
        auto grads = torch::randn({n, 2}, opts) * 0.001f;
        auto radii = torch::ones({n}, opts.dtype(torch::kInt32));
        ctrl.accumulate_gradients(grads, radii);
    }

    // Densify should not crash.
    auto stats = ctrl.densify(model, 5);
    EXPECT_TRUE(model.is_valid());
    EXPECT_EQ(stats.num_before, n);
}

TEST(DensificationTest, InvisibleGaussiansNotAccumulated) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    DensificationConfig config;
    config.densify_from  = 0;
    config.densify_until = 1000;
    config.densify_every = 5;
    config.grad_threshold = 0.0001f;
    DensificationController ctrl(config, 10.0f);

    const int n = 10;
    auto model = make_test_model(n, -5.0f); // Small scale for clone path.

    // Give all Gaussians high position gradients, but zero radii (invisible).
    for (int i = 0; i < 5; ++i) {
        auto grads = torch::ones({n, 2}, opts) * 10.0f; // Very high gradient.
        auto radii = torch::zeros({n}, opts.dtype(torch::kInt32)); // All invisible.
        ctrl.accumulate_gradients(grads, radii);
    }

    auto stats = ctrl.densify(model, 5);

    // No clones or splits because no Gaussian was ever visible.
    EXPECT_EQ(stats.num_cloned, 0);
    EXPECT_EQ(stats.num_split, 0);
}

// ===========================================================================
// Clone/Split/Prune Tests
// ===========================================================================

TEST(DensificationTest, HighGradSmallScaleGetsCloned) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    DensificationConfig config;
    config.densify_from  = 0;
    config.densify_until = 1000;
    config.densify_every = 5;
    config.grad_threshold = 0.0001f;
    config.percent_dense  = 0.01f;
    DensificationController ctrl(config, 10.0f);
    // Clone threshold: max(exp(scale)) < 0.01 * 10 = 0.1
    // scale = -5 => exp(-5) ≈ 0.0067 < 0.1 → clone.

    const int n = 10;
    auto model = make_test_model(n, -5.0f, 2.0f); // Small scale, high opacity.

    // Accumulate high gradients with all visible.
    for (int i = 0; i < 5; ++i) {
        auto grads = torch::ones({n, 2}, opts) * 1.0f; // Norm ~1.41 >> threshold.
        auto radii = torch::ones({n}, opts.dtype(torch::kInt32));
        ctrl.accumulate_gradients(grads, radii);
    }

    auto stats = ctrl.densify(model, 5);

    EXPECT_GT(stats.num_cloned, 0);
    EXPECT_EQ(stats.num_split, 0);
    EXPECT_GT(stats.num_after, stats.num_before);
    EXPECT_TRUE(model.is_valid());
}

TEST(DensificationTest, HighGradLargeScaleGetsSplit) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    DensificationConfig config;
    config.densify_from  = 0;
    config.densify_until = 1000;
    config.densify_every = 5;
    config.grad_threshold = 0.0001f;
    config.percent_dense  = 0.01f;
    DensificationController ctrl(config, 10.0f);
    // Split threshold: max(exp(scale)) >= 0.01 * 10 = 0.1
    // scale = 0 => exp(0) = 1.0 >= 0.1 → split.

    const int n = 10;
    auto model = make_test_model(n, 0.0f, 2.0f); // Large scale, high opacity.

    // Accumulate high gradients.
    for (int i = 0; i < 5; ++i) {
        auto grads = torch::ones({n, 2}, opts) * 1.0f;
        auto radii = torch::ones({n}, opts.dtype(torch::kInt32));
        ctrl.accumulate_gradients(grads, radii);
    }

    auto stats = ctrl.densify(model, 5);

    EXPECT_GT(stats.num_split, 0);
    EXPECT_TRUE(model.is_valid());
}

TEST(DensificationTest, LowOpacityGetsPruned) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    DensificationConfig config;
    config.densify_from  = 0;
    config.densify_until = 1000;
    config.densify_every = 5;
    config.opacity_threshold = 0.5f; // High threshold for easier testing.
    config.grad_threshold = 1000.0f; // Impossibly high to prevent clone/split.
    DensificationController ctrl(config, 10.0f);

    // Create 10 Gaussians: 5 high opacity, 5 low opacity.
    const int n = 10;
    auto model = make_test_model(n);

    {
        torch::NoGradGuard no_grad;
        // First 5: sigmoid(5.0) ≈ 0.993 > 0.5 → keep.
        model.opacities.slice(0, 0, 5).fill_(5.0f);
        // Last 5: sigmoid(-5.0) ≈ 0.007 < 0.5 → prune.
        model.opacities.slice(0, 5, 10).fill_(-5.0f);
    }

    // Need at least one accumulation step (even with no gradient).
    auto grads = torch::zeros({n, 2}, opts);
    auto radii = torch::ones({n}, opts.dtype(torch::kInt32));
    ctrl.accumulate_gradients(grads, radii);

    auto stats = ctrl.densify(model, 5);

    EXPECT_EQ(stats.num_pruned, 5);
    EXPECT_EQ(stats.num_after, 5);
    EXPECT_TRUE(model.is_valid());
}

TEST(DensificationTest, OpacityResetSetsLowValue) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    DensificationConfig config;
    DensificationController ctrl(config, 10.0f);

    const int n = 10;
    auto model = make_test_model(n, -2.0f, 5.0f); // High opacity.

    ctrl.reset_opacity(model);

    // Check that all opacities are set to inverse_sigmoid(0.01).
    float expected = std::log(0.01f / 0.99f);
    auto cpu_opa = model.opacities.cpu();
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(cpu_opa[i][0].item<float>(), expected, 0.01f);
    }
}

TEST(DensificationTest, ModelRemainsValidAfterFullCycle) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    DensificationConfig config;
    config.densify_from  = 0;
    config.densify_until = 1000;
    config.densify_every = 5;
    config.grad_threshold = 0.0001f;
    config.percent_dense  = 0.01f;
    config.opacity_threshold = 0.5f;
    DensificationController ctrl(config, 10.0f);

    const int n = 20;
    auto model = make_test_model(n);

    {
        torch::NoGradGuard no_grad;
        // Mix of small and large Gaussians.
        model.scales.slice(0, 0, 10).fill_(-5.0f);   // Small (clone candidates).
        model.scales.slice(0, 10, 20).fill_(0.0f);    // Large (split candidates).

        // Mix of high and low opacity.
        model.opacities.slice(0, 0, 5).fill_(-5.0f);  // Low → will be pruned.
        model.opacities.slice(0, 5, 20).fill_(3.0f);  // High → will survive.
    }

    // Accumulate gradients.
    for (int i = 0; i < 5; ++i) {
        auto grads = torch::ones({n, 2}, opts) * 1.0f;
        auto radii = torch::ones({n}, opts.dtype(torch::kInt32));
        ctrl.accumulate_gradients(grads, radii);
    }

    auto stats = ctrl.densify(model, 5);

    EXPECT_TRUE(model.is_valid());
    EXPECT_GT(stats.num_before, 0);
    EXPECT_GT(stats.num_after, 0);
    // Should have some clones, splits, and prunes.
    EXPECT_GE(stats.num_cloned + stats.num_split + stats.num_pruned, 1);
}

TEST(DensificationTest, MaxGaussiansRespected) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    DensificationConfig config;
    config.densify_from  = 0;
    config.densify_until = 1000;
    config.densify_every = 5;
    config.grad_threshold = 0.0001f;
    config.percent_dense  = 0.01f;
    config.max_gaussians  = 15; // Hard cap.
    DensificationController ctrl(config, 10.0f);

    const int n = 10;
    auto model = make_test_model(n, -5.0f, 2.0f); // Small scale, high opacity.

    // Accumulate high gradients to trigger cloning.
    for (int i = 0; i < 5; ++i) {
        auto grads = torch::ones({n, 2}, opts) * 1.0f;
        auto radii = torch::ones({n}, opts.dtype(torch::kInt32));
        ctrl.accumulate_gradients(grads, radii);
    }

    auto stats = ctrl.densify(model, 5);

    // Model should not exceed the cap.
    EXPECT_LE(stats.num_after, config.max_gaussians);
    EXPECT_LE(model.num_gaussians(), config.max_gaussians);
    EXPECT_TRUE(model.is_valid());
}

} // namespace
} // namespace cugs
