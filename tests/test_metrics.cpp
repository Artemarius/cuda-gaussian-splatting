/// @file test_metrics.cpp
/// @brief Unit tests for evaluation metrics (PSNR, SSIM, EvalResults JSON).

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include "training/metrics.hpp"

namespace cugs {
namespace {

/// @brief Helper: create a uniform-color [H, W, 3] CUDA tensor.
torch::Tensor make_uniform(int h, int w, float r, float g, float b) {
    auto img = torch::empty({h, w, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    img.select(2, 0).fill_(r);
    img.select(2, 1).fill_(g);
    img.select(2, 2).fill_(b);
    return img;
}

/// @brief Helper: create a random [H, W, 3] CUDA tensor with values in [0, 1].
torch::Tensor make_random(int h, int w) {
    return torch::rand({h, w, 3},
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
}

// ===========================================================================
// PSNR Tests
// ===========================================================================

TEST(MetricsTest, PSNRIdentical) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto img = make_random(64, 64);
    float psnr = compute_psnr(img, img);

    // Identical images should produce very high PSNR (clamped at 100 dB).
    EXPECT_GE(psnr, 100.0f);
}

TEST(MetricsTest, PSNRKnownDifference) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    // Uniform images with known MSE.
    // a = 0.5 everywhere, b = 0.7 everywhere
    // MSE = (0.2)^2 = 0.04
    // PSNR = 10 * log10(1 / 0.04) = 10 * log10(25) = 10 * 1.3979... = 13.979 dB
    auto a = make_uniform(32, 32, 0.5f, 0.5f, 0.5f);
    auto b = make_uniform(32, 32, 0.7f, 0.7f, 0.7f);

    float psnr = compute_psnr(a, b);
    float expected = 10.0f * std::log10(1.0f / 0.04f);

    EXPECT_NEAR(psnr, expected, 0.01f);
}

TEST(MetricsTest, PSNRSymmetric) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto a = make_random(64, 64);
    auto b = make_random(64, 64);

    float psnr_ab = compute_psnr(a, b);
    float psnr_ba = compute_psnr(b, a);

    EXPECT_NEAR(psnr_ab, psnr_ba, 1e-5f);
}

TEST(MetricsTest, PSNRRange) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto a = make_random(64, 64);
    auto b = make_random(64, 64);

    float psnr = compute_psnr(a, b);

    // PSNR for random [0,1] images should be positive and finite.
    EXPECT_GT(psnr, 0.0f);
    EXPECT_TRUE(std::isfinite(psnr));
}

// ===========================================================================
// SSIM Tests
// ===========================================================================

TEST(MetricsTest, SSIMIdentical) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto img = make_random(64, 64);
    float ssim_val = compute_ssim(img, img);

    // SSIM of identical images should be ~1.0.
    EXPECT_NEAR(ssim_val, 1.0f, 1e-4f);
}

TEST(MetricsTest, SSIMDifferent) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto a = make_uniform(64, 64, 0.0f, 0.0f, 0.0f);
    auto b = make_uniform(64, 64, 1.0f, 1.0f, 1.0f);

    float ssim_val = compute_ssim(a, b);

    // Very different images should have low SSIM.
    EXPECT_LT(ssim_val, 0.5f);
}

TEST(MetricsTest, SSIMSymmetric) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto a = make_random(64, 64);
    auto b = make_random(64, 64);

    float ssim_ab = compute_ssim(a, b);
    float ssim_ba = compute_ssim(b, a);

    EXPECT_NEAR(ssim_ab, ssim_ba, 1e-5f);
}

// ===========================================================================
// EvalResults JSON Tests
// ===========================================================================

TEST(MetricsTest, EvalResultsJsonRoundtrip) {
    EvalResults results;
    results.mean_psnr = 25.5f;
    results.mean_ssim = 0.88f;
    results.num_gaussians = 100000;
    results.sh_degree = 3;
    results.eval_time_seconds = 12.5f;

    ImageMetrics im1;
    im1.image_name = "test_001.jpg";
    im1.psnr = 24.3f;
    im1.ssim = 0.86f;
    results.per_image.push_back(im1);

    ImageMetrics im2;
    im2.image_name = "test_002.jpg";
    im2.psnr = 26.7f;
    im2.ssim = 0.90f;
    results.per_image.push_back(im2);

    std::string json_str = results.to_json();

    // Verify all expected fields are present in the JSON string.
    EXPECT_NE(json_str.find("\"mean_psnr\""), std::string::npos);
    EXPECT_NE(json_str.find("\"mean_ssim\""), std::string::npos);
    EXPECT_NE(json_str.find("\"num_gaussians\""), std::string::npos);
    EXPECT_NE(json_str.find("\"sh_degree\""), std::string::npos);
    EXPECT_NE(json_str.find("\"eval_time_seconds\""), std::string::npos);
    EXPECT_NE(json_str.find("\"num_test_images\""), std::string::npos);
    EXPECT_NE(json_str.find("\"per_image\""), std::string::npos);
    EXPECT_NE(json_str.find("test_001.jpg"), std::string::npos);
    EXPECT_NE(json_str.find("test_002.jpg"), std::string::npos);

    // Parse back and verify values.
    auto j = nlohmann::json::parse(json_str);
    EXPECT_NEAR(j["mean_psnr"].get<float>(), 25.5f, 0.01f);
    EXPECT_NEAR(j["mean_ssim"].get<float>(), 0.88f, 0.01f);
    EXPECT_EQ(j["num_gaussians"].get<int>(), 100000);
    EXPECT_EQ(j["sh_degree"].get<int>(), 3);
    EXPECT_EQ(j["num_test_images"].get<int>(), 2);
    EXPECT_EQ(j["per_image"].size(), 2u);
    EXPECT_EQ(j["per_image"][0]["image_name"].get<std::string>(), "test_001.jpg");
    EXPECT_NEAR(j["per_image"][0]["psnr"].get<float>(), 24.3f, 0.01f);
}

} // namespace
} // namespace cugs
