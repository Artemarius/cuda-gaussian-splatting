/// @file test_fused_adam.cpp
/// @brief Unit tests for Phase 8: Fused CUDA Adam optimizer.
///
/// Tests numerical equivalence between FusedAdam and the libtorch-wrapped
/// GaussianAdam, plus correct LR handling and bias correction behavior.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>

#include "core/gaussian.hpp"
#include "optimizer/adam.hpp"
#include "optimizer/fused_adam.hpp"
#include "rasterizer/rasterizer.hpp"
#include "training/lr_schedule.hpp"

namespace cugs {
namespace {

/// @brief Create a small synthetic Gaussian model on CUDA for testing.
///
/// @param n Number of Gaussians.
/// @param seed Random seed for reproducibility.
/// @return GaussianModel on CUDA.
GaussianModel make_test_model(int n, int seed = 42) {
    torch::manual_seed(seed);
    GaussianModel model;
    model.positions  = torch::randn({n, 3}, torch::kFloat32).to(torch::kCUDA);
    model.sh_coeffs  = torch::randn({n, 3, 16}, torch::kFloat32).to(torch::kCUDA);
    model.opacities  = torch::randn({n, 1}, torch::kFloat32).to(torch::kCUDA);
    model.scales     = torch::randn({n, 3}, torch::kFloat32).to(torch::kCUDA);
    model.rotations  = torch::randn({n, 4}, torch::kFloat32).to(torch::kCUDA);
    return model;
}

/// @brief Clone a GaussianModel (deep copy of all tensors).
GaussianModel clone_model(const GaussianModel& src) {
    GaussianModel dst;
    dst.positions  = src.positions.clone();
    dst.sh_coeffs  = src.sh_coeffs.clone();
    dst.opacities  = src.opacities.clone();
    dst.scales     = src.scales.clone();
    dst.rotations  = src.rotations.clone();
    return dst;
}

/// @brief Create a synthetic BackwardOutput with given random seed.
BackwardOutput make_test_grads(int n, int seed) {
    torch::manual_seed(seed);
    BackwardOutput grads;
    grads.dL_dpositions = torch::randn({n, 3}, torch::kFloat32).to(torch::kCUDA);
    grads.dL_dsh_coeffs = torch::randn({n, 3, 16}, torch::kFloat32).to(torch::kCUDA);
    grads.dL_dopacities = torch::randn({n, 1}, torch::kFloat32).to(torch::kCUDA);
    grads.dL_dscales    = torch::randn({n, 3}, torch::kFloat32).to(torch::kCUDA);
    grads.dL_drotations = torch::randn({n, 4}, torch::kFloat32).to(torch::kCUDA);
    grads.dL_dmeans_2d  = torch::zeros({n, 2}, torch::kFloat32).to(torch::kCUDA);
    return grads;
}

/// @brief Check that all 5 parameter tensors are close between two models.
///
/// @param a First model.
/// @param b Second model.
/// @param rtol Relative tolerance.
/// @param atol Absolute tolerance.
/// @param label Test label for error messages.
void assert_models_close(
    const GaussianModel& a,
    const GaussianModel& b,
    double rtol, double atol,
    const std::string& label)
{
    auto check = [&](const torch::Tensor& ta, const torch::Tensor& tb,
                     const std::string& name) {
        ASSERT_TRUE(torch::allclose(ta, tb, rtol, atol))
            << label << ": " << name << " mismatch.\n"
            << "  max abs diff = "
            << (ta - tb).abs().max().item<float>() << "\n"
            << "  max rel diff = "
            << ((ta - tb).abs() / (tb.abs() + atol)).max().item<float>();
    };

    check(a.positions,  b.positions,  "positions");
    check(a.sh_coeffs,  b.sh_coeffs,  "sh_coeffs");
    check(a.opacities,  b.opacities,  "opacities");
    check(a.scales,     b.scales,     "scales");
    check(a.rotations,  b.rotations,  "rotations");
}

// ===========================================================================
// Test 1: Single step produces results numerically equivalent to libtorch Adam
// ===========================================================================

TEST(FusedAdamTest, SingleStepEquivalence) {
    const int n = 100;
    auto model_ref = make_test_model(n, 42);
    auto model_fused = clone_model(model_ref);

    AdamConfig config;
    GaussianAdam ref_opt(model_ref, config);
    FusedAdam fused_opt(model_fused, config);

    auto grads = make_test_grads(n, 123);

    // Reference: libtorch Adam.
    ref_opt.zero_grad();
    ref_opt.apply_gradients(grads);
    ref_opt.step();

    // Fused: custom CUDA kernel.
    fused_opt.zero_grad();
    fused_opt.apply_gradients(grads);
    fused_opt.step();

    assert_models_close(model_fused, model_ref, 1e-5, 1e-6, "SingleStep");
}

// ===========================================================================
// Test 2: Multi-step produces equivalent results over 10 iterations
// ===========================================================================

TEST(FusedAdamTest, MultiStepEquivalence) {
    const int n = 50;
    auto model_ref = make_test_model(n, 7);
    auto model_fused = clone_model(model_ref);

    AdamConfig config;
    GaussianAdam ref_opt(model_ref, config);
    FusedAdam fused_opt(model_fused, config);

    for (int step = 0; step < 10; ++step) {
        auto grads = make_test_grads(n, 100 + step);

        ref_opt.zero_grad();
        ref_opt.apply_gradients(grads);
        ref_opt.step();

        fused_opt.zero_grad();
        fused_opt.apply_gradients(grads);
        fused_opt.step();
    }

    assert_models_close(model_fused, model_ref, 1e-4, 1e-5, "MultiStep");
}

// ===========================================================================
// Test 3: Initial learning rates match AdamConfig defaults
// ===========================================================================

TEST(FusedAdamTest, ParamGroupLRs) {
    const int n = 10;
    auto model = make_test_model(n, 1);

    AdamConfig config;
    FusedAdam opt(model, config);

    EXPECT_FLOAT_EQ(opt.get_lr(ParamGroup::kPositions),
                    config.position_lr_config.lr_init);
    EXPECT_FLOAT_EQ(opt.get_lr(ParamGroup::kSHCoeffs),
                    config.lr_sh_coeffs);
    EXPECT_FLOAT_EQ(opt.get_lr(ParamGroup::kOpacities),
                    config.lr_opacities);
    EXPECT_FLOAT_EQ(opt.get_lr(ParamGroup::kScales),
                    config.lr_scales);
    EXPECT_FLOAT_EQ(opt.get_lr(ParamGroup::kRotations),
                    config.lr_rotations);
}

// ===========================================================================
// Test 4: update_lr applies exponential decay to position group
// ===========================================================================

TEST(FusedAdamTest, LRUpdateDecay) {
    const int n = 10;
    auto model = make_test_model(n, 2);

    AdamConfig config;
    FusedAdam opt(model, config);

    float lr0 = opt.get_lr(ParamGroup::kPositions);
    EXPECT_FLOAT_EQ(lr0, config.position_lr_config.lr_init);

    // After update to step 15000 (halfway), LR should decrease.
    opt.update_lr(15000);
    float lr_mid = opt.get_lr(ParamGroup::kPositions);
    EXPECT_LT(lr_mid, lr0);

    // Should match the schedule function.
    float expected = position_lr(15000, config.position_lr_config);
    EXPECT_FLOAT_EQ(lr_mid, expected);

    // Other groups should be unchanged.
    EXPECT_FLOAT_EQ(opt.get_lr(ParamGroup::kSHCoeffs), config.lr_sh_coeffs);
    EXPECT_FLOAT_EQ(opt.get_lr(ParamGroup::kOpacities), config.lr_opacities);
}

// ===========================================================================
// Test 5: Zero gradients produce no parameter change
// ===========================================================================

TEST(FusedAdamTest, ZeroGradientNoChange) {
    const int n = 20;
    auto model = make_test_model(n, 3);
    auto model_before = clone_model(model);

    AdamConfig config;
    FusedAdam opt(model, config);

    // Create zero gradients.
    BackwardOutput zero_grads;
    zero_grads.dL_dpositions = torch::zeros({n, 3}, torch::kFloat32).to(torch::kCUDA);
    zero_grads.dL_dsh_coeffs = torch::zeros({n, 3, 16}, torch::kFloat32).to(torch::kCUDA);
    zero_grads.dL_dopacities = torch::zeros({n, 1}, torch::kFloat32).to(torch::kCUDA);
    zero_grads.dL_dscales    = torch::zeros({n, 3}, torch::kFloat32).to(torch::kCUDA);
    zero_grads.dL_drotations = torch::zeros({n, 4}, torch::kFloat32).to(torch::kCUDA);
    zero_grads.dL_dmeans_2d  = torch::zeros({n, 2}, torch::kFloat32).to(torch::kCUDA);

    opt.zero_grad();
    opt.apply_gradients(zero_grads);
    opt.step();

    // All parameters should remain unchanged (m=0, v=0, update=0).
    assert_models_close(model, model_before, 0.0, 0.0, "ZeroGrad");
}

// ===========================================================================
// Test 6: Step 1 produces larger updates than step 100
// ===========================================================================

TEST(FusedAdamTest, StepCountBiasCorrection) {
    const int n = 20;

    // Model A: takes 1 step (step_count=1, bias correction is large).
    auto model_a = make_test_model(n, 5);
    auto model_a_before = clone_model(model_a);

    AdamConfig config;
    FusedAdam opt_a(model_a, config);
    auto grads = make_test_grads(n, 200);
    opt_a.zero_grad();
    opt_a.apply_gradients(grads);
    opt_a.step();

    float diff_step1 = (model_a.positions - model_a_before.positions)
                           .abs().mean().item<float>();

    // Model B: takes 100 steps with zero gradients, then 1 step with same
    // gradients. At step 101, bias correction is near 1.
    auto model_b = make_test_model(n, 5);
    auto model_b_before = clone_model(model_b);

    FusedAdam opt_b(model_b, config);

    BackwardOutput zero_grads;
    zero_grads.dL_dpositions = torch::zeros({n, 3}, torch::kFloat32).to(torch::kCUDA);
    zero_grads.dL_dsh_coeffs = torch::zeros({n, 3, 16}, torch::kFloat32).to(torch::kCUDA);
    zero_grads.dL_dopacities = torch::zeros({n, 1}, torch::kFloat32).to(torch::kCUDA);
    zero_grads.dL_dscales    = torch::zeros({n, 3}, torch::kFloat32).to(torch::kCUDA);
    zero_grads.dL_drotations = torch::zeros({n, 4}, torch::kFloat32).to(torch::kCUDA);
    zero_grads.dL_dmeans_2d  = torch::zeros({n, 2}, torch::kFloat32).to(torch::kCUDA);

    for (int i = 0; i < 100; ++i) {
        opt_b.zero_grad();
        opt_b.apply_gradients(zero_grads);
        opt_b.step();
    }

    // Now apply the same real gradients.
    model_b_before = clone_model(model_b);  // snapshot before real step
    opt_b.zero_grad();
    opt_b.apply_gradients(grads);
    opt_b.step();

    float diff_step101 = (model_b.positions - model_b_before.positions)
                             .abs().mean().item<float>();

    // Step 1 has large bias correction (1/(1-0.9) = 10x for m), so the
    // update magnitude should be larger than at step 101 where bc ~ 1.
    EXPECT_GT(diff_step1, diff_step101)
        << "Step 1 update (" << diff_step1
        << ") should be larger than step 101 update (" << diff_step101 << ")";
}

} // namespace
} // namespace cugs
