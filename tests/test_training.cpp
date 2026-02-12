/// @file test_training.cpp
/// @brief Unit tests for Phase 6: training loop components.
///
/// Tests:
///   - LR schedule: position decay endpoints, clamping, active SH degree
///   - Image-to-tensor: RGB and RGBA conversion, pixel value verification
///   - Convergence: 20 synthetic Gaussians, 100 iterations, loss decreases >10%

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cmath>

#include "training/lr_schedule.hpp"
#include "training/trainer.hpp"
#include "training/loss.hpp"
#include "optimizer/adam.hpp"
#include "rasterizer/rasterizer.hpp"
#include "core/gaussian.hpp"
#include "core/types.hpp"
#include "data/image_io.hpp"

namespace cugs {
namespace {

// ===========================================================================
// LR Schedule Tests
// ===========================================================================

TEST(LRScheduleTest, PositionLRInitial) {
    PositionLRConfig config;
    float lr = position_lr(0, config);
    EXPECT_NEAR(lr, 1.6e-4f, 1e-8f);
}

TEST(LRScheduleTest, PositionLRFinal) {
    PositionLRConfig config;
    float lr = position_lr(config.max_steps, config);
    EXPECT_NEAR(lr, 1.6e-6f, 1e-10f);
}

TEST(LRScheduleTest, PositionLRBeyondMaxSteps) {
    PositionLRConfig config;
    float lr = position_lr(config.max_steps + 1000, config);
    EXPECT_NEAR(lr, config.lr_final, 1e-10f);
}

TEST(LRScheduleTest, PositionLRMonotonicallyDecreasing) {
    PositionLRConfig config;
    float prev = position_lr(0, config);
    for (int step = 100; step <= config.max_steps; step += 100) {
        float curr = position_lr(step, config);
        EXPECT_LT(curr, prev) << "LR should decrease at step " << step;
        prev = curr;
    }
}

TEST(LRScheduleTest, PositionLRMidpoint) {
    PositionLRConfig config;
    float mid = position_lr(config.max_steps / 2, config);
    // Should be geometric mean of init and final at halfway
    float expected = std::sqrt(config.lr_init * config.lr_final);
    EXPECT_NEAR(mid, expected, 1e-8f);
}

TEST(LRScheduleTest, ActiveSHDegreeClamping) {
    EXPECT_EQ(active_sh_degree_for_step(0, 3), 0);
    EXPECT_EQ(active_sh_degree_for_step(500, 3), 0);
    EXPECT_EQ(active_sh_degree_for_step(999, 3), 0);
    EXPECT_EQ(active_sh_degree_for_step(1000, 3), 1);
    EXPECT_EQ(active_sh_degree_for_step(1999, 3), 1);
    EXPECT_EQ(active_sh_degree_for_step(2000, 3), 2);
    EXPECT_EQ(active_sh_degree_for_step(3000, 3), 3);
    EXPECT_EQ(active_sh_degree_for_step(30000, 3), 3);
}

TEST(LRScheduleTest, ActiveSHDegreeMaxClamped) {
    // If max_degree is 1, should never exceed 1
    EXPECT_EQ(active_sh_degree_for_step(5000, 1), 1);
    EXPECT_EQ(active_sh_degree_for_step(0, 0), 0);
    EXPECT_EQ(active_sh_degree_for_step(5000, 0), 0);
}

// ===========================================================================
// Image-to-Tensor Tests
// ===========================================================================

TEST(ImageToTensorTest, RGB) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    Image img;
    img.width = 4;
    img.height = 3;
    img.channels = 3;
    img.data.resize(4 * 3 * 3);

    // Fill with known pattern: pixel (x,y) has R=x/3, G=y/2, B=0.5
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 4; ++x) {
            int idx = (y * 4 + x) * 3;
            img.data[idx + 0] = static_cast<float>(x) / 3.0f;
            img.data[idx + 1] = static_cast<float>(y) / 2.0f;
            img.data[idx + 2] = 0.5f;
        }
    }

    auto tensor = image_to_tensor(img, torch::kCUDA);

    EXPECT_EQ(tensor.dim(), 3);
    EXPECT_EQ(tensor.size(0), 3);  // H
    EXPECT_EQ(tensor.size(1), 4);  // W
    EXPECT_EQ(tensor.size(2), 3);  // C
    EXPECT_TRUE(tensor.is_cuda());
    EXPECT_EQ(tensor.dtype(), torch::kFloat32);

    // Verify a few pixel values on CPU.
    auto cpu = tensor.cpu();
    EXPECT_NEAR(cpu[0][0][0].item<float>(), 0.0f, 1e-5f);   // R at (0,0)
    EXPECT_NEAR(cpu[0][0][1].item<float>(), 0.0f, 1e-5f);   // G at (0,0)
    EXPECT_NEAR(cpu[0][0][2].item<float>(), 0.5f, 1e-5f);   // B at (0,0)
    EXPECT_NEAR(cpu[1][3][0].item<float>(), 1.0f, 1e-5f);   // R at (3,1)
    EXPECT_NEAR(cpu[1][3][1].item<float>(), 0.5f, 1e-5f);   // G at (3,1)
}

TEST(ImageToTensorTest, RGBA) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    Image img;
    img.width = 2;
    img.height = 2;
    img.channels = 4;
    img.data = {
        // Row 0
        0.1f, 0.2f, 0.3f, 1.0f,   // pixel (0,0) RGBA
        0.4f, 0.5f, 0.6f, 0.8f,   // pixel (1,0) RGBA
        // Row 1
        0.7f, 0.8f, 0.9f, 0.5f,   // pixel (0,1) RGBA
        0.0f, 0.1f, 0.2f, 0.0f,   // pixel (1,1) RGBA
    };

    auto tensor = image_to_tensor(img, torch::kCUDA);

    // Alpha channel should be stripped.
    EXPECT_EQ(tensor.size(2), 3);
    EXPECT_EQ(tensor.size(0), 2);
    EXPECT_EQ(tensor.size(1), 2);

    auto cpu = tensor.cpu();
    EXPECT_NEAR(cpu[0][0][0].item<float>(), 0.1f, 1e-5f);
    EXPECT_NEAR(cpu[0][0][1].item<float>(), 0.2f, 1e-5f);
    EXPECT_NEAR(cpu[0][0][2].item<float>(), 0.3f, 1e-5f);
    // Alpha should NOT be present
    EXPECT_NEAR(cpu[0][1][0].item<float>(), 0.4f, 1e-5f);
}

// ===========================================================================
// Convergence Test: perturb SH coefficients, recover via optimization
// ===========================================================================

TEST(TrainingTest, SyntheticConvergence) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::manual_seed(42);

    // Create a synthetic model with 20 Gaussians.
    const int n = 20;
    auto positions = torch::randn({n, 3}, opts) * 0.3f;
    positions.select(1, 2).abs_().add_(3.5f);

    auto rotations = torch::randn({n, 4}, opts);
    rotations = rotations / rotations.norm(2, 1, true).clamp_min(1e-8f);

    auto scales = torch::full({n, 3}, -1.5f, opts) + torch::randn({n, 3}, opts) * 0.2f;
    auto opacities_logit = torch::full({n, 1}, 2.0f, opts);
    auto sh_coeffs = torch::randn({n, 3, 1}, opts) * 0.5f;

    GaussianModel model;
    model.positions = positions.clone();
    model.rotations = rotations.clone();
    model.scales    = scales.clone();
    model.opacities = opacities_logit.clone();
    model.sh_coeffs = sh_coeffs.clone();

    // Camera at origin looking down +Z.
    CameraInfo cam;
    cam.width = 64;
    cam.height = 48;
    cam.intrinsics.fx = 100.0f;
    cam.intrinsics.fy = 100.0f;
    cam.intrinsics.cx = 32.0f;
    cam.intrinsics.cy = 24.0f;
    cam.rotation = Eigen::Matrix3f::Identity();
    cam.translation = Eigen::Vector3f::Zero();

    RenderSettings settings;
    settings.active_sh_degree = 0;

    // Render a target from the original model.
    torch::Tensor target;
    {
        torch::NoGradGuard no_grad;
        target = render(model, cam, settings).color.clone();
    }

    // Perturb SH coefficients significantly to create a gap.
    // This is a well-conditioned optimization problem: recovering SH values
    // while geometry stays near-correct.
    {
        torch::NoGradGuard no_grad;
        model.sh_coeffs.add_(torch::randn_like(model.sh_coeffs) * 1.0f);
    }

    // Create optimizer — only SH needs a high LR; keep geometry LRs low.
    AdamConfig adam_config;
    adam_config.position_lr_config.lr_init = 1e-4f;
    adam_config.lr_sh_coeffs = 5e-2f;
    adam_config.lr_opacities = 1e-2f;
    adam_config.lr_scales    = 1e-3f;
    adam_config.lr_rotations = 1e-4f;

    GaussianAdam optimizer(model, adam_config);

    // Compute initial loss (should be high due to SH perturbation).
    float initial_loss;
    {
        torch::NoGradGuard no_grad;
        initial_loss = combined_loss(render(model, cam, settings).color, target).item<float>();
    }

    // Train for 100 iterations.
    const int num_iters = 100;
    float final_loss = initial_loss;

    for (int i = 0; i < num_iters; ++i) {
        optimizer.zero_grad();

        auto render_out = render(model, cam, settings);

        // Autograd for dL/dcolor.
        auto rendered = render_out.color.clone().detach().requires_grad_(true);
        auto loss = combined_loss(rendered, target);
        loss.backward();
        auto dL_dcolor = rendered.grad().clone();

        // Custom backward.
        auto grads = render_backward(dL_dcolor, render_out, model, cam, settings);

        // Inject and step.
        optimizer.apply_gradients(grads);
        optimizer.step();

        final_loss = loss.item<float>();
    }

    // Loss should decrease by at least 10%.
    float decrease_pct = (initial_loss - final_loss) / initial_loss * 100.0f;
    EXPECT_GT(decrease_pct, 10.0f)
        << "Loss should decrease by >10% after 100 iterations. "
        << "Initial: " << initial_loss << ", Final: " << final_loss
        << " (" << decrease_pct << "% decrease)";
}

// ===========================================================================
// Adam optimizer basic test
// ===========================================================================

TEST(AdamTest, ParamGroupLRs) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    GaussianModel model;
    model.positions = torch::randn({5, 3}, opts);
    model.sh_coeffs = torch::randn({5, 3, 1}, opts);
    model.opacities = torch::randn({5, 1}, opts);
    model.scales    = torch::randn({5, 3}, opts);
    model.rotations = torch::randn({5, 4}, opts);

    AdamConfig config;
    GaussianAdam optimizer(model, config);

    EXPECT_NEAR(optimizer.get_lr(ParamGroup::kPositions),
                config.position_lr_config.lr_init, 1e-8f);
    EXPECT_NEAR(optimizer.get_lr(ParamGroup::kSHCoeffs),
                config.lr_sh_coeffs, 1e-8f);
    EXPECT_NEAR(optimizer.get_lr(ParamGroup::kOpacities),
                config.lr_opacities, 1e-8f);
    EXPECT_NEAR(optimizer.get_lr(ParamGroup::kScales),
                config.lr_scales, 1e-8f);
    EXPECT_NEAR(optimizer.get_lr(ParamGroup::kRotations),
                config.lr_rotations, 1e-8f);
}

TEST(AdamTest, LRUpdateChangesPositionLR) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    GaussianModel model;
    model.positions = torch::randn({5, 3}, opts);
    model.sh_coeffs = torch::randn({5, 3, 1}, opts);
    model.opacities = torch::randn({5, 1}, opts);
    model.scales    = torch::randn({5, 3}, opts);
    model.rotations = torch::randn({5, 4}, opts);

    AdamConfig config;
    GaussianAdam optimizer(model, config);

    float lr_init = optimizer.get_lr(ParamGroup::kPositions);
    optimizer.update_lr(15000);  // halfway
    float lr_mid = optimizer.get_lr(ParamGroup::kPositions);

    EXPECT_LT(lr_mid, lr_init) << "Position LR should decrease over training";

    // Other groups should be unchanged.
    EXPECT_NEAR(optimizer.get_lr(ParamGroup::kSHCoeffs),
                config.lr_sh_coeffs, 1e-8f);
}

} // namespace
} // namespace cugs
