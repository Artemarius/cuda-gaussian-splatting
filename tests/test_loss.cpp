/// @file test_loss.cpp
/// @brief Unit tests for L1, SSIM, and combined loss functions.

#include <gtest/gtest.h>
#include <torch/torch.h>

#include "training/loss.hpp"

namespace cugs {
namespace {

/// @brief Helper: create a uniform-color [H, W, 3] CUDA tensor.
torch::Tensor make_uniform(int h, int w, float r, float g, float b) {
    auto img = torch::empty({h, w, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    img.select(2, 0).fill_(r);
    img.select(2, 1).fill_(g);
    img.select(2, 2).fill_(b);
    return img;
}

/// @brief Helper: create a random [H, W, 3] CUDA tensor with values in [0, 1].
torch::Tensor make_random(int h, int w) {
    return torch::rand({h, w, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
}

// ===========================================================================
// L1 Loss Tests
// ===========================================================================

TEST(LossTest, L1IdenticalImages) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto img = make_random(64, 64);
    auto loss = ::cugs::l1_loss(img, img);

    EXPECT_NEAR(loss.item<float>(), 0.0f, 1e-6f);
}

TEST(LossTest, L1KnownDifference) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto a = make_uniform(32, 32, 0.8f, 0.8f, 0.8f);
    auto b = make_uniform(32, 32, 0.3f, 0.3f, 0.3f);

    auto loss = ::cugs::l1_loss(a, b);
    // |0.8 - 0.3| = 0.5 for every element
    EXPECT_NEAR(loss.item<float>(), 0.5f, 1e-5f);
}

TEST(LossTest, L1NonNegative) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto a = make_random(64, 64);
    auto b = make_random(64, 64);

    auto loss = ::cugs::l1_loss(a, b);
    EXPECT_GE(loss.item<float>(), 0.0f);
}

// ===========================================================================
// SSIM Tests
// ===========================================================================

TEST(LossTest, SSIMIdenticalImages) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto img = make_random(64, 64);
    auto ssim_map = ssim(img, img);

    // SSIM of identical images should be ~1.0 everywhere
    EXPECT_NEAR(ssim_map.mean().item<float>(), 1.0f, 1e-4f);
}

TEST(LossTest, SSIMDifferentImages) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto a = make_uniform(64, 64, 0.0f, 0.0f, 0.0f);
    auto b = make_uniform(64, 64, 1.0f, 1.0f, 1.0f);

    auto ssim_map = ssim(a, b);
    // Very different images should have low SSIM
    EXPECT_LT(ssim_map.mean().item<float>(), 0.1f);
}

TEST(LossTest, SSIMSymmetry) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto a = make_random(64, 64);
    auto b = make_random(64, 64);

    auto ssim_ab = ssim(a, b).mean().item<float>();
    auto ssim_ba = ssim(b, a).mean().item<float>();

    EXPECT_NEAR(ssim_ab, ssim_ba, 1e-5f);
}

TEST(LossTest, SSIMRange) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto a = make_random(64, 64);
    auto b = make_random(64, 64);

    auto ssim_map = ssim(a, b);

    EXPECT_GE(ssim_map.min().item<float>(), -1.0f - 1e-5f);
    EXPECT_LE(ssim_map.max().item<float>(), 1.0f + 1e-5f);
}

// ===========================================================================
// Combined Loss Tests
// ===========================================================================

TEST(LossTest, CombinedLossIdentical) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto img = make_random(64, 64);
    auto loss = combined_loss(img, img);

    // Identical images: L1=0, SSIM=1 → combined = 0
    EXPECT_NEAR(loss.item<float>(), 0.0f, 1e-4f);
}

TEST(LossTest, CombinedLossDecreases) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto target = make_random(64, 64);

    // "Close" image: small perturbation
    auto close = (target + 0.05f * torch::randn_like(target)).clamp(0.0f, 1.0f);
    // "Far" image: large perturbation
    auto far = (target + 0.5f * torch::randn_like(target)).clamp(0.0f, 1.0f);

    auto loss_close = combined_loss(close, target);
    auto loss_far   = combined_loss(far, target);

    EXPECT_LT(loss_close.item<float>(), loss_far.item<float>());
}

// ===========================================================================
// Input Validation Tests
// ===========================================================================

TEST(LossTest, InputValidation) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cuda_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto cpu_opts  = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

    auto valid = torch::rand({32, 32, 3}, cuda_opts);

    // Mismatched shapes
    auto wrong_shape = torch::rand({64, 64, 3}, cuda_opts);
    EXPECT_THROW(::cugs::l1_loss(valid, wrong_shape), c10::Error);

    // Wrong number of channels
    auto wrong_channels = torch::rand({32, 32, 4}, cuda_opts);
    EXPECT_THROW(::cugs::l1_loss(wrong_channels, wrong_channels), c10::Error);

    // CPU tensor (should require CUDA)
    auto cpu_img = torch::rand({32, 32, 3}, cpu_opts);
    EXPECT_THROW(::cugs::l1_loss(cpu_img, cpu_img), c10::Error);

    // Wrong dtype
    auto int_img = torch::randint(0, 255, {32, 32, 3},
                                  torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    EXPECT_THROW(::cugs::l1_loss(int_img, int_img), c10::Error);

    // Even window_size
    EXPECT_THROW(ssim(valid, valid, /*window_size=*/10), c10::Error);
}

} // namespace
} // namespace cugs
