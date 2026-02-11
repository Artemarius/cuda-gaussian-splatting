#include "core/sh.hpp"
#include "core/gaussian.hpp"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cmath>

namespace cugs {
namespace {

// ---------------------------------------------------------------------------
// CPU SH evaluation tests
// ---------------------------------------------------------------------------

TEST(SHEvalCPU, Degree0ConstantColor) {
    // A constant color should be fully captured by the DC coefficient.
    // SH DC term: color = C0 * coeff + 0.5
    // For color 0.7:  coeff = (0.7 - 0.5) / C0
    const float target_color = 0.7f;
    const float dc_coeff = (target_color - 0.5f) / kSH_C0;

    auto sh = torch::zeros({1, 3, 1}, torch::kFloat32);
    sh[0][0][0] = dc_coeff;
    sh[0][1][0] = dc_coeff;
    sh[0][2][0] = dc_coeff;

    // Any direction should give the same color for degree 0
    auto dir = torch::tensor({{0.0f, 0.0f, 1.0f}});
    auto result = evaluate_sh_cpu(0, sh, dir);

    EXPECT_NEAR(result[0][0].item<float>(), target_color, 1e-5);
    EXPECT_NEAR(result[0][1].item<float>(), target_color, 1e-5);
    EXPECT_NEAR(result[0][2].item<float>(), target_color, 1e-5);
}

TEST(SHEvalCPU, Degree0DirectionIndependent) {
    // Degree 0 should give the same result regardless of direction
    const float dc_coeff = 1.5f;
    auto sh = torch::zeros({1, 3, 1}, torch::kFloat32);
    sh[0][0][0] = dc_coeff;
    sh[0][1][0] = dc_coeff;
    sh[0][2][0] = dc_coeff;

    auto dir1 = torch::tensor({{1.0f, 0.0f, 0.0f}});
    auto dir2 = torch::tensor({{0.0f, 1.0f, 0.0f}});
    auto dir3 = torch::tensor({{0.0f, 0.0f, 1.0f}});

    auto r1 = evaluate_sh_cpu(0, sh, dir1);
    auto r2 = evaluate_sh_cpu(0, sh, dir2);
    auto r3 = evaluate_sh_cpu(0, sh, dir3);

    EXPECT_NEAR(r1[0][0].item<float>(), r2[0][0].item<float>(), 1e-6);
    EXPECT_NEAR(r2[0][0].item<float>(), r3[0][0].item<float>(), 1e-6);
}

TEST(SHEvalCPU, Degree1DirectionDependent) {
    // With degree 1 coefficients, different directions should give different colors
    auto sh = torch::zeros({1, 3, 4}, torch::kFloat32);
    // Set DC to zero, set degree-1 coefficients to non-zero
    sh[0][0][0] = 0.0f;
    sh[0][0][1] = 1.0f;  // Y_1^{-1} direction
    sh[0][0][2] = 0.0f;
    sh[0][0][3] = 0.0f;

    auto dir_y_pos = torch::tensor({{0.0f, 1.0f, 0.0f}});
    auto dir_y_neg = torch::tensor({{0.0f, -1.0f, 0.0f}});
    auto dir_x = torch::tensor({{1.0f, 0.0f, 0.0f}});

    auto ry_pos = evaluate_sh_cpu(1, sh, dir_y_pos);
    auto ry_neg = evaluate_sh_cpu(1, sh, dir_y_neg);
    auto rx = evaluate_sh_cpu(1, sh, dir_x);

    // Y_1^{-1} = -C1*y, so +y and -y should give opposite signs
    float val_yp = ry_pos[0][0].item<float>() - 0.5f;  // subtract bias
    float val_yn = ry_neg[0][0].item<float>() - 0.5f;
    float val_x  = rx[0][0].item<float>() - 0.5f;

    EXPECT_NEAR(val_yp, -val_yn, 1e-5);
    // X direction should be zero for Y_1^{-1} basis
    EXPECT_NEAR(val_x, 0.0f, 1e-5);
}

TEST(SHEvalCPU, BatchEvaluation) {
    // Evaluate multiple Gaussians at once
    const int n = 5;
    auto sh = torch::randn({n, 3, 4}, torch::kFloat32);
    auto dirs = torch::randn({n, 3}, torch::kFloat32);
    dirs = dirs / dirs.norm(2, /*dim=*/1, /*keepdim=*/true);  // normalize

    auto result = evaluate_sh_cpu(1, sh, dirs);
    EXPECT_EQ(result.size(0), n);
    EXPECT_EQ(result.size(1), 3);

    // Verify each element individually
    for (int i = 0; i < n; ++i) {
        auto sh_i = sh.index({i}).unsqueeze(0);      // [1, 3, 4]
        auto dir_i = dirs.index({i}).unsqueeze(0);    // [1, 3]
        auto result_i = evaluate_sh_cpu(1, sh_i, dir_i);

        for (int ch = 0; ch < 3; ++ch) {
            EXPECT_NEAR(result[i][ch].item<float>(),
                       result_i[0][ch].item<float>(), 1e-5)
                << "Mismatch at Gaussian " << i << " channel " << ch;
        }
    }
}

TEST(SHEvalCPU, HigherDegreesWithZeroCoeffs) {
    // If higher-order coefficients are zero, degree 3 should give same result
    // as degree 0
    const float dc_val = 1.2f;
    auto sh = torch::zeros({1, 3, 16}, torch::kFloat32);
    sh[0][0][0] = dc_val;
    sh[0][1][0] = dc_val;
    sh[0][2][0] = dc_val;

    auto dir = torch::tensor({{0.577f, 0.577f, 0.577f}});  // normalised-ish

    auto r0 = evaluate_sh_cpu(0, sh, dir);
    auto r3 = evaluate_sh_cpu(3, sh, dir);

    // Should be the same since higher coefficients are zero
    EXPECT_NEAR(r0[0][0].item<float>(), r3[0][0].item<float>(), 1e-5);
}

TEST(SHEvalCPU, InvalidInputs) {
    // Wrong batch sizes
    auto sh = torch::zeros({5, 3, 4}, torch::kFloat32);
    auto dirs = torch::zeros({3, 3}, torch::kFloat32);
    EXPECT_THROW(evaluate_sh_cpu(1, sh, dirs), c10::Error);

    // Not enough coefficients for degree
    auto sh2 = torch::zeros({1, 3, 1}, torch::kFloat32);
    auto dir2 = torch::zeros({1, 3}, torch::kFloat32);
    EXPECT_THROW(evaluate_sh_cpu(1, sh2, dir2), c10::Error);

    // Invalid degree
    auto sh3 = torch::zeros({1, 3, 16}, torch::kFloat32);
    auto dir3 = torch::zeros({1, 3}, torch::kFloat32);
    EXPECT_THROW(evaluate_sh_cpu(4, sh3, dir3), c10::Error);
    EXPECT_THROW(evaluate_sh_cpu(-1, sh3, dir3), c10::Error);
}

// ---------------------------------------------------------------------------
// CUDA SH evaluation tests (only run if CUDA is available)
// ---------------------------------------------------------------------------

class SHEvalCUDA : public ::testing::Test {
protected:
    bool has_cuda_ = false;

    void SetUp() override {
        has_cuda_ = torch::cuda::is_available();
        if (!has_cuda_) {
            GTEST_SKIP() << "CUDA not available, skipping GPU SH tests";
        }
    }
};

TEST_F(SHEvalCUDA, MatchesCPUDegree0) {
    auto sh = torch::randn({10, 3, 1}, torch::kFloat32);
    auto dirs = torch::randn({10, 3}, torch::kFloat32);
    dirs = dirs / dirs.norm(2, 1, true);

    auto cpu_result = evaluate_sh_cpu(0, sh, dirs);
    auto gpu_result = evaluate_sh_cuda(0, sh.cuda(), dirs.cuda());

    EXPECT_TRUE(torch::allclose(cpu_result, gpu_result.cpu(), 1e-4, 1e-4));
}

TEST_F(SHEvalCUDA, MatchesCPUDegree1) {
    auto sh = torch::randn({100, 3, 4}, torch::kFloat32);
    auto dirs = torch::randn({100, 3}, torch::kFloat32);
    dirs = dirs / dirs.norm(2, 1, true);

    auto cpu_result = evaluate_sh_cpu(1, sh, dirs);
    auto gpu_result = evaluate_sh_cuda(1, sh.cuda(), dirs.cuda());

    EXPECT_TRUE(torch::allclose(cpu_result, gpu_result.cpu(), 1e-4, 1e-4));
}

TEST_F(SHEvalCUDA, MatchesCPUDegree2) {
    auto sh = torch::randn({100, 3, 9}, torch::kFloat32);
    auto dirs = torch::randn({100, 3}, torch::kFloat32);
    dirs = dirs / dirs.norm(2, 1, true);

    auto cpu_result = evaluate_sh_cpu(2, sh, dirs);
    auto gpu_result = evaluate_sh_cuda(2, sh.cuda(), dirs.cuda());

    EXPECT_TRUE(torch::allclose(cpu_result, gpu_result.cpu(), 1e-4, 1e-4));
}

TEST_F(SHEvalCUDA, MatchesCPUDegree3) {
    auto sh = torch::randn({200, 3, 16}, torch::kFloat32);
    auto dirs = torch::randn({200, 3}, torch::kFloat32);
    dirs = dirs / dirs.norm(2, 1, true);

    auto cpu_result = evaluate_sh_cpu(3, sh, dirs);
    auto gpu_result = evaluate_sh_cuda(3, sh.cuda(), dirs.cuda());

    EXPECT_TRUE(torch::allclose(cpu_result, gpu_result.cpu(), 1e-4, 1e-4));
}

TEST_F(SHEvalCUDA, LargeBatch) {
    // Test with a realistic Gaussian count
    const int n = 10000;
    auto sh = torch::randn({n, 3, 16}, torch::kFloat32);
    auto dirs = torch::randn({n, 3}, torch::kFloat32);
    dirs = dirs / dirs.norm(2, 1, true);

    auto cpu_result = evaluate_sh_cpu(3, sh, dirs);
    auto gpu_result = evaluate_sh_cuda(3, sh.cuda(), dirs.cuda());

    EXPECT_TRUE(torch::allclose(cpu_result, gpu_result.cpu(), 1e-3, 1e-3));
}

} // anonymous namespace
} // namespace cugs
