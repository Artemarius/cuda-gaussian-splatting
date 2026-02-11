#include "core/gaussian.hpp"
#include "utils/ply_io.hpp"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <filesystem>
#include <cmath>

namespace cugs {
namespace {

/// Helper: create a small GaussianModel with known values.
GaussianModel make_test_model(int64_t n, int sh_degree = 3) {
    const int num_coeffs = sh_coeff_count(sh_degree);

    GaussianModel m;
    m.positions  = torch::randn({n, 3}, torch::kFloat32);
    m.sh_coeffs  = torch::randn({n, 3, num_coeffs}, torch::kFloat32);
    m.opacities  = torch::randn({n, 1}, torch::kFloat32);
    m.rotations  = torch::randn({n, 4}, torch::kFloat32);
    m.scales     = torch::randn({n, 3}, torch::kFloat32);
    return m;
}

// ---------------------------------------------------------------------------
// Basic construction & validation
// ---------------------------------------------------------------------------

TEST(GaussianModel, DefaultIsInvalid) {
    GaussianModel m;
    EXPECT_FALSE(m.is_valid());
    EXPECT_EQ(m.num_gaussians(), 0);
}

TEST(GaussianModel, ValidAfterCreation) {
    auto m = make_test_model(100);
    EXPECT_TRUE(m.is_valid());
    EXPECT_EQ(m.num_gaussians(), 100);
}

TEST(GaussianModel, MaxSHDegree) {
    for (int d = 0; d <= 3; ++d) {
        auto m = make_test_model(10, d);
        EXPECT_EQ(m.max_sh_degree(), d);
    }
}

TEST(GaussianModel, ShCoeffCount) {
    EXPECT_EQ(sh_coeff_count(0), 1);
    EXPECT_EQ(sh_coeff_count(1), 4);
    EXPECT_EQ(sh_coeff_count(2), 9);
    EXPECT_EQ(sh_coeff_count(3), 16);
}

TEST(GaussianModel, InvalidShapesDetected) {
    auto m = make_test_model(10);
    // Break position shape
    auto saved = m.positions;
    m.positions = torch::randn({10, 4});
    EXPECT_FALSE(m.is_valid());
    m.positions = saved;

    // Break count mismatch
    m.opacities = torch::randn({5, 1});
    EXPECT_FALSE(m.is_valid());
}

TEST(GaussianModel, EmptyModelIsValid) {
    GaussianModel m;
    m.positions  = torch::zeros({0, 3}, torch::kFloat32);
    m.sh_coeffs  = torch::zeros({0, 3, 16}, torch::kFloat32);
    m.opacities  = torch::zeros({0, 1}, torch::kFloat32);
    m.rotations  = torch::zeros({0, 4}, torch::kFloat32);
    m.scales     = torch::zeros({0, 3}, torch::kFloat32);
    EXPECT_TRUE(m.is_valid());
    EXPECT_EQ(m.num_gaussians(), 0);
}

// ---------------------------------------------------------------------------
// PLY roundtrip
// ---------------------------------------------------------------------------

class GaussianPlyTest : public ::testing::Test {
protected:
    std::filesystem::path temp_dir_;

    void SetUp() override {
        temp_dir_ = std::filesystem::temp_directory_path() / "cugs_test_gaussian_ply";
        std::filesystem::create_directories(temp_dir_);
    }

    void TearDown() override {
        std::filesystem::remove_all(temp_dir_);
    }
};

TEST_F(GaussianPlyTest, RoundtripDegree3) {
    auto original = make_test_model(50, 3);
    auto ply_path = temp_dir_ / "test_d3.ply";

    ASSERT_TRUE(original.save_ply(ply_path));
    ASSERT_TRUE(std::filesystem::exists(ply_path));

    auto loaded = GaussianModel::load_ply(ply_path);
    ASSERT_TRUE(loaded.is_valid());
    EXPECT_EQ(loaded.num_gaussians(), 50);
    EXPECT_EQ(loaded.max_sh_degree(), 3);

    // Values should match within float precision
    EXPECT_TRUE(torch::allclose(original.positions, loaded.positions, 1e-5, 1e-5));
    EXPECT_TRUE(torch::allclose(original.sh_coeffs, loaded.sh_coeffs, 1e-5, 1e-5));
    EXPECT_TRUE(torch::allclose(original.opacities, loaded.opacities, 1e-5, 1e-5));
    EXPECT_TRUE(torch::allclose(original.rotations, loaded.rotations, 1e-5, 1e-5));
    EXPECT_TRUE(torch::allclose(original.scales, loaded.scales, 1e-5, 1e-5));
}

TEST_F(GaussianPlyTest, RoundtripDegree0) {
    auto original = make_test_model(20, 0);
    auto ply_path = temp_dir_ / "test_d0.ply";

    ASSERT_TRUE(original.save_ply(ply_path));
    auto loaded = GaussianModel::load_ply(ply_path);

    ASSERT_TRUE(loaded.is_valid());
    EXPECT_EQ(loaded.num_gaussians(), 20);
    EXPECT_EQ(loaded.max_sh_degree(), 0);
    EXPECT_TRUE(torch::allclose(original.positions, loaded.positions, 1e-5, 1e-5));
    EXPECT_TRUE(torch::allclose(original.sh_coeffs, loaded.sh_coeffs, 1e-5, 1e-5));
}

TEST_F(GaussianPlyTest, RoundtripDegree2) {
    auto original = make_test_model(30, 2);
    auto ply_path = temp_dir_ / "test_d2.ply";

    ASSERT_TRUE(original.save_ply(ply_path));
    auto loaded = GaussianModel::load_ply(ply_path);

    ASSERT_TRUE(loaded.is_valid());
    EXPECT_EQ(loaded.max_sh_degree(), 2);
    EXPECT_TRUE(torch::allclose(original.sh_coeffs, loaded.sh_coeffs, 1e-5, 1e-5));
}

TEST_F(GaussianPlyTest, EmptyModel) {
    GaussianModel m;
    m.positions  = torch::zeros({0, 3}, torch::kFloat32);
    m.sh_coeffs  = torch::zeros({0, 3, 16}, torch::kFloat32);
    m.opacities  = torch::zeros({0, 1}, torch::kFloat32);
    m.rotations  = torch::zeros({0, 4}, torch::kFloat32);
    m.scales     = torch::zeros({0, 3}, torch::kFloat32);

    auto ply_path = temp_dir_ / "empty.ply";
    ASSERT_TRUE(m.save_ply(ply_path));

    auto loaded = GaussianModel::load_ply(ply_path);
    EXPECT_EQ(loaded.num_gaussians(), 0);
}

} // anonymous namespace
} // namespace cugs
