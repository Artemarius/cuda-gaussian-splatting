/// @file test_projection.cpp
/// @brief Unit tests for the Gaussian projection pipeline.
///
/// Tests cover:
///   - Single Gaussian at known position → verify 2D mean, depth, non-zero radius
///   - Frustum culling: Gaussian behind camera → radius = 0
///   - Batch of random Gaussians: no NaN/Inf in outputs
///   - Anisotropic Gaussian: non-uniform scale → elongated 2D projection

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cmath>

#include "rasterizer/projection.hpp"
#include "core/gaussian.hpp"
#include "core/types.hpp"

namespace cugs {
namespace {

/// @brief Create a simple pinhole camera looking down the -Z axis (OpenGL convention)
///        or along +Z axis in the standard COLMAP convention (world-to-camera).
///
/// Camera at origin, looking down +Z, image is 640×480, focal length 500.
CameraInfo make_test_camera() {
    CameraInfo cam;
    cam.width = 640;
    cam.height = 480;
    cam.intrinsics.fx = 500.0f;
    cam.intrinsics.fy = 500.0f;
    cam.intrinsics.cx = 320.0f;
    cam.intrinsics.cy = 240.0f;
    cam.rotation = Eigen::Matrix3f::Identity();
    cam.translation = Eigen::Vector3f::Zero();
    return cam;
}

/// @brief Create tensors for a single Gaussian at the given world position.
struct SingleGaussianFixture {
    torch::Tensor positions;
    torch::Tensor rotations;
    torch::Tensor scales;
    torch::Tensor opacities;
    torch::Tensor sh_coeffs;

    SingleGaussianFixture(float x, float y, float z,
                          float log_sx = -2.0f, float log_sy = -2.0f, float log_sz = -2.0f)
    {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        positions  = torch::tensor({{x, y, z}}, opts);
        rotations  = torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, opts); // identity quaternion
        scales     = torch::tensor({{log_sx, log_sy, log_sz}}, opts);
        opacities  = torch::tensor({{0.0f}}, opts); // sigmoid(0) = 0.5
        // SH degree 0 only: 1 coefficient per channel → white-ish Gaussian
        // DC = 1.0 → color ≈ 1.0 * C0 + 0.5 = ~0.78
        sh_coeffs  = torch::ones({1, 3, 1}, opts);
    }
};

// ===========================================================================
// Test: Single Gaussian in front of camera
// ===========================================================================

TEST(ProjectionTest, SingleGaussianInFront) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    // Place Gaussian at (0, 0, 5) — centered, 5 units in front
    SingleGaussianFixture g(0.0f, 0.0f, 5.0f);

    auto out = project_gaussians(
        g.positions, g.rotations, g.scales, g.opacities, g.sh_coeffs,
        cam, /*active_sh_degree=*/0);

    // Should not be culled
    auto radii = out.radii.cpu();
    EXPECT_GT(radii[0].item<int>(), 0) << "Gaussian should have non-zero radius";

    // 2D position should be near image center (cx, cy)
    auto means = out.means_2d.cpu();
    float mx = means[0][0].item<float>();
    float my = means[0][1].item<float>();
    EXPECT_NEAR(mx, 320.0f, 1.0f) << "X should be near principal point";
    EXPECT_NEAR(my, 240.0f, 1.0f) << "Y should be near principal point";

    // Depth should be 5.0
    auto depths = out.depths.cpu();
    EXPECT_NEAR(depths[0].item<float>(), 5.0f, 0.01f);

    // Opacity should be sigmoid(0) = 0.5
    auto opacity = out.opacities_act.cpu();
    EXPECT_NEAR(opacity[0].item<float>(), 0.5f, 0.01f);

    // Tiles touched should be > 0
    auto tiles = out.tiles_touched.cpu();
    EXPECT_GT(tiles[0].item<int>(), 0);

    // RGB should be non-negative
    auto rgb = out.rgb.cpu();
    EXPECT_GE(rgb[0][0].item<float>(), 0.0f);
    EXPECT_GE(rgb[0][1].item<float>(), 0.0f);
    EXPECT_GE(rgb[0][2].item<float>(), 0.0f);
}

// ===========================================================================
// Test: Gaussian behind camera → culled
// ===========================================================================

TEST(ProjectionTest, GaussianBehindCameraCulled) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    // Place Gaussian at (0, 0, -5) — behind the camera
    SingleGaussianFixture g(0.0f, 0.0f, -5.0f);

    auto out = project_gaussians(
        g.positions, g.rotations, g.scales, g.opacities, g.sh_coeffs,
        cam, /*active_sh_degree=*/0);

    auto radii = out.radii.cpu();
    EXPECT_EQ(radii[0].item<int>(), 0) << "Gaussian behind camera should be culled";

    auto tiles = out.tiles_touched.cpu();
    EXPECT_EQ(tiles[0].item<int>(), 0);
}

// ===========================================================================
// Test: Off-center Gaussian projects to expected position
// ===========================================================================

TEST(ProjectionTest, OffCenterProjection) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    // Place Gaussian at (1, 0, 5) — off to the right
    SingleGaussianFixture g(1.0f, 0.0f, 5.0f);

    auto out = project_gaussians(
        g.positions, g.rotations, g.scales, g.opacities, g.sh_coeffs,
        cam, /*active_sh_degree=*/0);

    auto means = out.means_2d.cpu();
    float mx = means[0][0].item<float>();
    float my = means[0][1].item<float>();

    // Expected: x = fx * 1/5 + cx = 500 * 0.2 + 320 = 420
    EXPECT_NEAR(mx, 420.0f, 1.0f);
    EXPECT_NEAR(my, 240.0f, 1.0f);
}

// ===========================================================================
// Test: Batch of random Gaussians — no NaN/Inf
// ===========================================================================

TEST(ProjectionTest, BatchNoNanInf) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    const int n = 1000;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // Random Gaussians scattered in front of the camera
    auto positions = torch::randn({n, 3}, opts);
    positions.select(1, 2).abs_().add_(1.0f); // z > 1 (in front)

    auto rotations = torch::randn({n, 4}, opts);
    // Normalise quaternions
    rotations = rotations / rotations.norm(2, 1, true).clamp_min(1e-8f);

    auto scales    = torch::randn({n, 3}, opts) * 0.5f - 3.0f; // small log-scales
    auto opacities = torch::randn({n, 1}, opts);
    auto sh_coeffs = torch::randn({n, 3, 1}, opts);

    auto out = project_gaussians(
        positions, rotations, scales, opacities, sh_coeffs,
        cam, /*active_sh_degree=*/0);

    // Check no NaN/Inf in means_2d, depths, cov_2d_inv
    EXPECT_FALSE(out.means_2d.isnan().any().item<bool>()) << "means_2d has NaN";
    EXPECT_FALSE(out.means_2d.isinf().any().item<bool>()) << "means_2d has Inf";
    EXPECT_FALSE(out.depths.isnan().any().item<bool>()) << "depths has NaN";
    EXPECT_FALSE(out.cov_2d_inv.isnan().any().item<bool>()) << "cov_2d_inv has NaN";
    EXPECT_FALSE(out.rgb.isnan().any().item<bool>()) << "rgb has NaN";
    EXPECT_FALSE(out.opacities_act.isnan().any().item<bool>()) << "opacities_act has NaN";
}

// ===========================================================================
// Test: Anisotropic Gaussian — non-uniform scales produce non-circular projection
// ===========================================================================

TEST(ProjectionTest, AnisotropicProjection) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();

    // Gaussian at (0, 0, 5) with large X scale, small Y/Z scale
    SingleGaussianFixture g_wide(0.0f, 0.0f, 5.0f, 0.0f, -4.0f, -4.0f);
    // Gaussian at (0, 0, 5) with isotropic small scale
    SingleGaussianFixture g_iso(0.0f, 0.0f, 5.0f, -4.0f, -4.0f, -4.0f);

    auto out_wide = project_gaussians(
        g_wide.positions, g_wide.rotations, g_wide.scales, g_wide.opacities,
        g_wide.sh_coeffs, cam, 0);

    auto out_iso = project_gaussians(
        g_iso.positions, g_iso.rotations, g_iso.scales, g_iso.opacities,
        g_iso.sh_coeffs, cam, 0);

    // The wide Gaussian should have a larger radius than the isotropic one
    int radius_wide = out_wide.radii.cpu()[0].item<int>();
    int radius_iso  = out_iso.radii.cpu()[0].item<int>();

    EXPECT_GT(radius_wide, 0);
    EXPECT_GT(radius_iso, 0);
    EXPECT_GT(radius_wide, radius_iso)
        << "Anisotropic Gaussian should have larger radius than isotropic";
}

// ===========================================================================
// Test: Empty input
// ===========================================================================

TEST(ProjectionTest, EmptyInput) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    auto out = project_gaussians(
        torch::empty({0, 3}, opts),
        torch::empty({0, 4}, opts),
        torch::empty({0, 3}, opts),
        torch::empty({0, 1}, opts),
        torch::empty({0, 3, 1}, opts),
        cam, 0);

    EXPECT_EQ(out.means_2d.size(0), 0);
    EXPECT_EQ(out.radii.size(0), 0);
}

// ===========================================================================
// Test: Scale modifier affects radius
// ===========================================================================

TEST(ProjectionTest, ScaleModifierAffectsRadius) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    SingleGaussianFixture g(0.0f, 0.0f, 5.0f, -2.0f, -2.0f, -2.0f);

    auto out_normal = project_gaussians(
        g.positions, g.rotations, g.scales, g.opacities, g.sh_coeffs,
        cam, 0, /*scale_modifier=*/1.0f);

    auto out_scaled = project_gaussians(
        g.positions, g.rotations, g.scales, g.opacities, g.sh_coeffs,
        cam, 0, /*scale_modifier=*/2.0f);

    int r_normal = out_normal.radii.cpu()[0].item<int>();
    int r_scaled = out_scaled.radii.cpu()[0].item<int>();

    EXPECT_GT(r_normal, 0);
    EXPECT_GT(r_scaled, 0);
    EXPECT_GT(r_scaled, r_normal)
        << "Larger scale_modifier should produce larger radius";
}

} // namespace
} // namespace cugs
