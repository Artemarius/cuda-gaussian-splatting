#include "core/gaussian_init.hpp"
#include "core/gaussian.hpp"
#include "core/sh.hpp"
#include "core/types.hpp"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>
#include <vector>

namespace cugs {
namespace {

/// Helper: create N mock sparse points arranged on a grid.
std::vector<SparsePoint> make_grid_points(int n, float spacing = 1.0f) {
    std::vector<SparsePoint> points;
    points.reserve(n);
    int side = static_cast<int>(std::ceil(std::cbrt(static_cast<float>(n))));
    int count = 0;
    for (int x = 0; x < side && count < n; ++x) {
        for (int y = 0; y < side && count < n; ++y) {
            for (int z = 0; z < side && count < n; ++z) {
                SparsePoint pt;
                pt.point_id = count;
                pt.position = Eigen::Vector3f(
                    static_cast<float>(x) * spacing,
                    static_cast<float>(y) * spacing,
                    static_cast<float>(z) * spacing
                );
                pt.color[0] = static_cast<uint8_t>((count * 37) % 256);
                pt.color[1] = static_cast<uint8_t>((count * 73) % 256);
                pt.color[2] = static_cast<uint8_t>((count * 113) % 256);
                pt.error = 0.5f;
                points.push_back(pt);
                ++count;
            }
        }
    }
    return points;
}

// ---------------------------------------------------------------------------
// Basic initialization
// ---------------------------------------------------------------------------

TEST(GaussianInit, EmptyPoints) {
    std::vector<SparsePoint> empty;
    auto model = init_gaussians_from_sparse(empty, 3);
    EXPECT_TRUE(model.is_valid());
    EXPECT_EQ(model.num_gaussians(), 0);
}

TEST(GaussianInit, SinglePoint) {
    std::vector<SparsePoint> pts(1);
    pts[0].position = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
    pts[0].color[0] = 128;
    pts[0].color[1] = 64;
    pts[0].color[2] = 255;

    auto model = init_gaussians_from_sparse(pts, 3);
    ASSERT_TRUE(model.is_valid());
    EXPECT_EQ(model.num_gaussians(), 1);

    // Position should match
    auto pos = model.positions.accessor<float, 2>();
    EXPECT_NEAR(pos[0][0], 1.0f, 1e-6);
    EXPECT_NEAR(pos[0][1], 2.0f, 1e-6);
    EXPECT_NEAR(pos[0][2], 3.0f, 1e-6);

    // Rotation should be identity quaternion
    auto rot = model.rotations.accessor<float, 2>();
    EXPECT_NEAR(rot[0][0], 1.0f, 1e-6);  // w
    EXPECT_NEAR(rot[0][1], 0.0f, 1e-6);  // x
    EXPECT_NEAR(rot[0][2], 0.0f, 1e-6);  // y
    EXPECT_NEAR(rot[0][3], 0.0f, 1e-6);  // z

    // Opacity should be inverse_sigmoid(0.1)
    auto opa = model.opacities.accessor<float, 2>();
    EXPECT_NEAR(opa[0][0], -2.1972f, 1e-3);
}

TEST(GaussianInit, CorrectTensorShapes) {
    auto pts = make_grid_points(100);
    auto model = init_gaussians_from_sparse(pts, 2);
    ASSERT_TRUE(model.is_valid());

    EXPECT_EQ(model.positions.sizes(), (std::vector<int64_t>{100, 3}));
    EXPECT_EQ(model.sh_coeffs.sizes(), (std::vector<int64_t>{100, 3, 9}));
    EXPECT_EQ(model.opacities.sizes(), (std::vector<int64_t>{100, 1}));
    EXPECT_EQ(model.rotations.sizes(), (std::vector<int64_t>{100, 4}));
    EXPECT_EQ(model.scales.sizes(), (std::vector<int64_t>{100, 3}));
}

// ---------------------------------------------------------------------------
// SH DC coefficient initialization
// ---------------------------------------------------------------------------

TEST(GaussianInit, DCCoeffFromColor) {
    // Create a point with known color and verify DC coefficient
    std::vector<SparsePoint> pts(1);
    pts[0].position = Eigen::Vector3f::Zero();
    pts[0].color[0] = 255;  // R = 1.0
    pts[0].color[1] = 0;    // G = 0.0
    pts[0].color[2] = 128;  // B ~ 0.502

    auto model = init_gaussians_from_sparse(pts, 0);
    auto sh = model.sh_coeffs.accessor<float, 3>();

    // DC coeff = (color_float - 0.5) / C0
    // Evaluating SH(degree=0) should give back the original color
    auto dir = torch::tensor({{0.0f, 0.0f, 1.0f}});
    auto rgb = evaluate_sh_cpu(0, model.sh_coeffs, dir);

    EXPECT_NEAR(rgb[0][0].item<float>(), 1.0f, 1e-2);       // R
    EXPECT_NEAR(rgb[0][1].item<float>(), 0.0f, 1e-2);       // G
    EXPECT_NEAR(rgb[0][2].item<float>(), 128.0f / 255.0f, 1e-2);  // B
}

TEST(GaussianInit, HigherSHBandsAreZero) {
    auto pts = make_grid_points(10);
    auto model = init_gaussians_from_sparse(pts, 3);
    auto sh = model.sh_coeffs.accessor<float, 3>();

    // All coefficients beyond index 0 should be zero
    for (int64_t i = 0; i < 10; ++i) {
        for (int ch = 0; ch < 3; ++ch) {
            for (int k = 1; k < 16; ++k) {
                EXPECT_NEAR(sh[i][ch][k], 0.0f, 1e-7)
                    << "Non-zero higher SH at Gaussian " << i
                    << " ch " << ch << " k " << k;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Scale initialization (k-NN)
// ---------------------------------------------------------------------------

TEST(GaussianInit, ScaleIsReasonable) {
    // Grid with spacing 1.0 — nearest neighbors should be at distance 1.0
    auto pts = make_grid_points(27, 1.0f);  // 3x3x3 grid
    auto model = init_gaussians_from_sparse(pts, 0, /*k_neighbors=*/3);

    auto scales = model.scales.accessor<float, 2>();
    // For interior points, k=3 nearest neighbors are at distance 1.0
    // So log(mean_dist) ~ log(1.0) = 0.0
    // Edge/corner points will have slightly larger distances.
    // Overall: scales should be in a reasonable range around 0.
    float min_scale = model.scales.min().item<float>();
    float max_scale = model.scales.max().item<float>();

    EXPECT_GT(min_scale, -5.0f);  // not unreasonably small
    EXPECT_LT(max_scale, 5.0f);   // not unreasonably large
}

TEST(GaussianInit, ScaleIsIsotropic) {
    auto pts = make_grid_points(50);
    auto model = init_gaussians_from_sparse(pts, 0);

    auto scales = model.scales.accessor<float, 2>();
    // All 3 scale components should be equal (isotropic init)
    for (int64_t i = 0; i < model.num_gaussians(); ++i) {
        EXPECT_NEAR(scales[i][0], scales[i][1], 1e-7);
        EXPECT_NEAR(scales[i][1], scales[i][2], 1e-7);
    }
}

// ---------------------------------------------------------------------------
// Opacity initialization
// ---------------------------------------------------------------------------

TEST(GaussianInit, OpacityAllSame) {
    auto pts = make_grid_points(20);
    auto model = init_gaussians_from_sparse(pts, 0);

    auto opa = model.opacities.accessor<float, 2>();
    const float expected = std::log(0.1f / 0.9f);

    for (int64_t i = 0; i < 20; ++i) {
        EXPECT_NEAR(opa[i][0], expected, 1e-4);
    }
}

TEST(GaussianInit, OpacitySigmoidRecovery) {
    // Applying sigmoid to the logit should recover 0.1
    auto pts = make_grid_points(5);
    auto model = init_gaussians_from_sparse(pts, 0);

    auto sigmoid_opa = torch::sigmoid(model.opacities);
    auto val = sigmoid_opa[0][0].item<float>();
    EXPECT_NEAR(val, 0.1f, 1e-5);
}

// ---------------------------------------------------------------------------
// Different SH degrees
// ---------------------------------------------------------------------------

TEST(GaussianInit, AllDegrees) {
    auto pts = make_grid_points(10);
    for (int d = 0; d <= 3; ++d) {
        auto model = init_gaussians_from_sparse(pts, d);
        ASSERT_TRUE(model.is_valid()) << "Invalid model for degree " << d;
        EXPECT_EQ(model.max_sh_degree(), d);
        EXPECT_EQ(model.sh_coeffs.size(2), sh_coeff_count(d));
    }
}

} // anonymous namespace
} // namespace cugs
