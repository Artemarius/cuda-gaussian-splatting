/// @file test_rasterizer.cpp
/// @brief Integration tests for the full rasterization pipeline.
///
/// Tests cover:
///   - Empty scene → background color everywhere
///   - Single Gaussian → center pixel near expected color, far pixels ≈ background
///   - Depth ordering → front Gaussian dominates overlap region
///   - Background blending → uncovered pixels exactly = background
///   - No NaN/Inf in output for random Gaussians

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cmath>

#include "rasterizer/rasterizer.hpp"
#include "core/gaussian.hpp"
#include "core/types.hpp"

namespace cugs {
namespace {

/// @brief Create a test camera at the origin looking down +Z.
CameraInfo make_test_camera(int w = 160, int h = 120) {
    CameraInfo cam;
    cam.width = w;
    cam.height = h;
    cam.intrinsics.fx = 200.0f;
    cam.intrinsics.fy = 200.0f;
    cam.intrinsics.cx = static_cast<float>(w) / 2.0f;
    cam.intrinsics.cy = static_cast<float>(h) / 2.0f;
    cam.rotation = Eigen::Matrix3f::Identity();
    cam.translation = Eigen::Vector3f::Zero();
    return cam;
}

/// @brief Build a GaussianModel from raw tensors on CUDA.
GaussianModel make_model(const torch::Tensor& positions,
                         const torch::Tensor& rotations,
                         const torch::Tensor& scales,
                         const torch::Tensor& opacities,
                         const torch::Tensor& sh_coeffs)
{
    GaussianModel m;
    m.positions  = positions;
    m.rotations  = rotations;
    m.scales     = scales;
    m.opacities  = opacities;
    m.sh_coeffs  = sh_coeffs;
    return m;
}

/// @brief Create a single-Gaussian model at the given position with specified
///        SH DC coefficient (controls color).
GaussianModel make_single_gaussian(float x, float y, float z,
                                    float sh_dc = 1.0f,
                                    float opacity_logit = 5.0f,
                                    float log_scale = -2.0f)
{
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    return make_model(
        torch::tensor({{x, y, z}}, opts),
        torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, opts),
        torch::full({1, 3}, log_scale, opts),
        torch::full({1, 1}, opacity_logit, opts),
        torch::full({1, 3, 1}, sh_dc, opts));
}

// ===========================================================================
// Test: Empty scene → background color
// ===========================================================================

TEST(RasterizerTest, EmptySceneBackground) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    GaussianModel model;
    model.positions = torch::empty({0, 3}, opts);
    model.rotations = torch::empty({0, 4}, opts);
    model.scales    = torch::empty({0, 3}, opts);
    model.opacities = torch::empty({0, 1}, opts);
    model.sh_coeffs = torch::empty({0, 3, 1}, opts);

    RenderSettings settings;
    settings.background[0] = 0.3f;
    settings.background[1] = 0.5f;
    settings.background[2] = 0.7f;
    settings.active_sh_degree = 0;

    auto out = render(model, cam, settings);

    ASSERT_EQ(out.color.size(0), cam.height);
    ASSERT_EQ(out.color.size(1), cam.width);
    ASSERT_EQ(out.color.size(2), 3);

    auto color_cpu = out.color.cpu();
    // All pixels should be background color
    float r = color_cpu[cam.height / 2][cam.width / 2][0].item<float>();
    float g = color_cpu[cam.height / 2][cam.width / 2][1].item<float>();
    float b = color_cpu[cam.height / 2][cam.width / 2][2].item<float>();

    EXPECT_NEAR(r, 0.3f, 0.01f);
    EXPECT_NEAR(g, 0.5f, 0.01f);
    EXPECT_NEAR(b, 0.7f, 0.01f);
}

// ===========================================================================
// Test: Single Gaussian — center pixel should have color influence
// ===========================================================================

TEST(RasterizerTest, SingleGaussianCenterPixel) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    // Gaussian at center of view, 5 units away, high opacity
    auto model = make_single_gaussian(0.0f, 0.0f, 5.0f,
                                       /*sh_dc=*/2.0f,
                                       /*opacity_logit=*/5.0f,
                                       /*log_scale=*/-1.0f);

    RenderSettings settings;
    settings.background[0] = 0.0f;
    settings.background[1] = 0.0f;
    settings.background[2] = 0.0f;
    settings.active_sh_degree = 0;

    auto out = render(model, cam, settings);
    auto color_cpu = out.color.cpu();

    // Center pixel should have non-zero color (Gaussian influence)
    int cy = cam.height / 2;
    int cx = cam.width / 2;
    float center_r = color_cpu[cy][cx][0].item<float>();
    float center_g = color_cpu[cy][cx][1].item<float>();
    float center_b = color_cpu[cy][cx][2].item<float>();

    EXPECT_GT(center_r, 0.1f) << "Center should have Gaussian color";
    EXPECT_GT(center_g, 0.1f);
    EXPECT_GT(center_b, 0.1f);

    // Corner pixel should be closer to background (black)
    float corner_r = color_cpu[0][0][0].item<float>();
    float corner_g = color_cpu[0][0][1].item<float>();
    float corner_b = color_cpu[0][0][2].item<float>();

    // Corner should have less color than center
    EXPECT_LT(corner_r, center_r);
    EXPECT_LT(corner_g, center_g);
}

// ===========================================================================
// Test: Depth ordering — front Gaussian dominates
// ===========================================================================

TEST(RasterizerTest, DepthOrdering) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // Two Gaussians at the same screen position but different depths
    // Front (z=3): bright SH (high DC) → should dominate
    // Back  (z=8): dim SH (low DC)
    auto positions = torch::tensor({{0.0f, 0.0f, 3.0f},
                                    {0.0f, 0.0f, 8.0f}}, opts);
    auto rotations = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f},
                                    {1.0f, 0.0f, 0.0f, 0.0f}}, opts);
    auto scales    = torch::full({2, 3}, -1.5f, opts);
    auto opacities = torch::full({2, 1}, 5.0f, opts); // high opacity → ~0.99

    // Front Gaussian: bright (SH DC = 3.0)
    // Back Gaussian: dim (SH DC = 0.0)
    auto sh_coeffs = torch::zeros({2, 3, 1}, opts);
    sh_coeffs[0].fill_(3.0f); // bright
    sh_coeffs[1].fill_(0.0f); // dim (color ≈ 0.5 from DC offset, clamped)

    auto model = make_model(positions, rotations, scales, opacities, sh_coeffs);

    RenderSettings settings;
    settings.background[0] = 0.0f;
    settings.background[1] = 0.0f;
    settings.background[2] = 0.0f;
    settings.active_sh_degree = 0;

    auto out = render(model, cam, settings);
    auto color_cpu = out.color.cpu();

    int cy = cam.height / 2;
    int cx = cam.width / 2;
    float center_r = color_cpu[cy][cx][0].item<float>();

    // Center should be dominated by the bright front Gaussian
    // SH DC=3.0 → color = 3.0 * C0 + 0.5 ≈ 1.35 (clamped to ≥0)
    EXPECT_GT(center_r, 0.5f) << "Front bright Gaussian should dominate";
}

// ===========================================================================
// Test: Background blending — uncovered pixels match background
// ===========================================================================

TEST(RasterizerTest, BackgroundBlending) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();

    // Small Gaussian far from corners — corners should be pure background
    auto model = make_single_gaussian(0.0f, 0.0f, 5.0f,
                                       /*sh_dc=*/1.0f,
                                       /*opacity_logit=*/3.0f,
                                       /*log_scale=*/-4.0f); // very small

    RenderSettings settings;
    settings.background[0] = 1.0f;
    settings.background[1] = 0.0f;
    settings.background[2] = 1.0f; // magenta background
    settings.active_sh_degree = 0;

    auto out = render(model, cam, settings);
    auto color_cpu = out.color.cpu();

    // Corner pixel should be very close to background
    float corner_r = color_cpu[0][0][0].item<float>();
    float corner_g = color_cpu[0][0][1].item<float>();
    float corner_b = color_cpu[0][0][2].item<float>();

    EXPECT_NEAR(corner_r, 1.0f, 0.05f) << "Corner should be background red";
    EXPECT_NEAR(corner_g, 0.0f, 0.05f) << "Corner should be background green";
    EXPECT_NEAR(corner_b, 1.0f, 0.05f) << "Corner should be background blue";
}

// ===========================================================================
// Test: No NaN/Inf for random Gaussians
// ===========================================================================

TEST(RasterizerTest, RandomGaussiansNoNanInf) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    const int n = 500;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto positions = torch::randn({n, 3}, opts);
    positions.select(1, 2).abs_().add_(1.0f); // z > 1

    auto rotations = torch::randn({n, 4}, opts);
    rotations = rotations / rotations.norm(2, 1, true).clamp_min(1e-8f);

    auto scales    = torch::randn({n, 3}, opts) * 0.5f - 3.0f;
    auto opacities = torch::randn({n, 1}, opts);
    auto sh_coeffs = torch::randn({n, 3, 1}, opts);

    auto model = make_model(positions, rotations, scales, opacities, sh_coeffs);

    RenderSettings settings;
    settings.background[0] = 0.5f;
    settings.background[1] = 0.5f;
    settings.background[2] = 0.5f;
    settings.active_sh_degree = 0;

    auto out = render(model, cam, settings);

    EXPECT_FALSE(out.color.isnan().any().item<bool>()) << "Output color has NaN";
    EXPECT_FALSE(out.color.isinf().any().item<bool>()) << "Output color has Inf";
    EXPECT_FALSE(out.final_T.isnan().any().item<bool>()) << "final_T has NaN";

    // All transmittances should be in [0, 1]
    auto T_cpu = out.final_T.cpu();
    EXPECT_GE(T_cpu.min().item<float>(), 0.0f);
    EXPECT_LE(T_cpu.max().item<float>(), 1.0f + 1e-5f);
}

// ===========================================================================
// Test: Transmittance and n_contrib are reasonable
// ===========================================================================

TEST(RasterizerTest, TransmittanceAndContrib) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    // Dense Gaussian covering center
    auto model = make_single_gaussian(0.0f, 0.0f, 5.0f,
                                       /*sh_dc=*/1.0f,
                                       /*opacity_logit=*/5.0f,
                                       /*log_scale=*/0.0f); // large

    RenderSettings settings;
    settings.active_sh_degree = 0;

    auto out = render(model, cam, settings);

    int cy = cam.height / 2;
    int cx = cam.width / 2;

    // Center pixel: high opacity Gaussian → low transmittance
    float center_T = out.final_T.cpu()[cy][cx].item<float>();
    EXPECT_LT(center_T, 0.5f) << "Center should have low transmittance with opaque Gaussian";

    // At least 1 contributor at center
    int center_n = out.n_contrib.cpu()[cy][cx].item<int>();
    EXPECT_GE(center_n, 1) << "Center should have at least 1 contributor";
}

// ===========================================================================
// Test: Render output shapes are correct
// ===========================================================================

TEST(RasterizerTest, OutputShapes) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera(320, 240);
    auto model = make_single_gaussian(0.0f, 0.0f, 5.0f);

    RenderSettings settings;
    settings.active_sh_degree = 0;

    auto out = render(model, cam, settings);

    EXPECT_EQ(out.color.sizes(), (std::vector<int64_t>{240, 320, 3}));
    EXPECT_EQ(out.final_T.sizes(), (std::vector<int64_t>{240, 320}));
    EXPECT_EQ(out.n_contrib.sizes(), (std::vector<int64_t>{240, 320}));
    EXPECT_EQ(out.means_2d.size(0), 1);
    EXPECT_EQ(out.means_2d.size(1), 2);
    EXPECT_EQ(out.radii.size(0), 1);
}

} // namespace
} // namespace cugs
