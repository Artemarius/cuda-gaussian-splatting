/// @file loss.cpp
/// @brief Loss function implementations for 3D Gaussian Splatting training.
///
/// Uses libtorch tensor operations (conv2d dispatches to cuDNN on CUDA tensors).
/// No custom CUDA kernels — these are not performance bottlenecks.

#include "training/loss.hpp"

#include <cmath>

namespace cugs {
namespace {

/// @brief Validate that an image tensor has the expected shape and device.
void validate_image(const torch::Tensor& img, const char* name) {
    TORCH_CHECK(img.dim() == 3,
                name, " must be 3-dimensional [H, W, 3], got ", img.dim(), " dims");
    TORCH_CHECK(img.size(2) == 3,
                name, " must have 3 channels, got ", img.size(2));
    TORCH_CHECK(img.dtype() == torch::kFloat32,
                name, " must be float32, got ", img.dtype());
    TORCH_CHECK(img.is_cuda(),
                name, " must be on a CUDA device");
}

/// @brief Validate that two image tensors have matching shapes.
void validate_pair(const torch::Tensor& rendered, const torch::Tensor& target) {
    validate_image(rendered, "rendered");
    validate_image(target, "target");
    TORCH_CHECK(rendered.sizes() == target.sizes(),
                "rendered and target must have the same shape, got ",
                rendered.sizes(), " vs ", target.sizes());
}

/// @brief Create a 2D Gaussian kernel for SSIM computation.
///
/// The kernel is shaped [3, 1, size, size] for use with grouped conv2d
/// (one filter per RGB channel). Cached as a static local per (size, device)
/// pair to avoid recreating it every call.
///
/// @param window_size Side length (must be odd).
/// @param device      Target CUDA device.
/// @return Gaussian kernel [3, 1, size, size], float32.
torch::Tensor get_gaussian_kernel(int window_size, torch::Device device) {
    // We cache the kernel in a static map keyed by (window_size, device_index).
    // This avoids re-creating it every call while supporting multiple devices.
    static std::unordered_map<int64_t, torch::Tensor> cache;

    const int64_t key = static_cast<int64_t>(window_size) * 1000 +
                        device.index();

    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }

    // 1D Gaussian kernel with sigma = 1.5
    const float sigma = 1.5f;
    const int half = window_size / 2;
    auto kernel_1d = torch::zeros({window_size}, torch::kFloat32);
    auto kernel_1d_a = kernel_1d.accessor<float, 1>();
    for (int i = 0; i < window_size; ++i) {
        float x = static_cast<float>(i - half);
        kernel_1d_a[i] = std::exp(-x * x / (2.0f * sigma * sigma));
    }
    kernel_1d = kernel_1d / kernel_1d.sum();

    // 2D kernel as outer product
    auto kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0); // [size, size]
    kernel_2d = kernel_2d / kernel_2d.sum(); // normalize

    // Shape for grouped conv2d: [3, 1, size, size]
    auto kernel = kernel_2d.unsqueeze(0).unsqueeze(0).expand({3, 1, window_size, window_size})
                      .contiguous()
                      .to(device);

    cache[key] = kernel;
    return kernel;
}

} // namespace

torch::Tensor l1_loss(const torch::Tensor& rendered, const torch::Tensor& target) {
    validate_pair(rendered, target);
    return (rendered - target).abs().mean();
}

torch::Tensor ssim(const torch::Tensor& rendered, const torch::Tensor& target,
                   int window_size) {
    validate_pair(rendered, target);
    TORCH_CHECK(window_size % 2 == 1, "window_size must be odd, got ", window_size);
    TORCH_CHECK(window_size >= 3, "window_size must be >= 3, got ", window_size);

    const int padding = window_size / 2;
    auto kernel = get_gaussian_kernel(window_size, rendered.device());

    // Permute from [H, W, 3] to [1, 3, H, W] for conv2d
    auto x = rendered.permute({2, 0, 1}).unsqueeze(0); // [1, 3, H, W]
    auto y = target.permute({2, 0, 1}).unsqueeze(0);   // [1, 3, H, W]

    // Weighted means
    auto mu_x = torch::conv2d(x, kernel, /*bias=*/{}, /*stride=*/1, padding, /*dilation=*/1, /*groups=*/3);
    auto mu_y = torch::conv2d(y, kernel, /*bias=*/{}, /*stride=*/1, padding, /*dilation=*/1, /*groups=*/3);

    auto mu_x_sq = mu_x * mu_x;
    auto mu_y_sq = mu_y * mu_y;
    auto mu_xy   = mu_x * mu_y;

    // Weighted variances and covariance
    auto sigma_x_sq = torch::conv2d(x * x, kernel, /*bias=*/{}, /*stride=*/1, padding, /*dilation=*/1, /*groups=*/3) - mu_x_sq;
    auto sigma_y_sq = torch::conv2d(y * y, kernel, /*bias=*/{}, /*stride=*/1, padding, /*dilation=*/1, /*groups=*/3) - mu_y_sq;
    auto sigma_xy   = torch::conv2d(x * y, kernel, /*bias=*/{}, /*stride=*/1, padding, /*dilation=*/1, /*groups=*/3) - mu_xy;

    // SSIM constants for dynamic range L = 1.0
    constexpr float C1 = 0.01f * 0.01f; // 0.0001
    constexpr float C2 = 0.03f * 0.03f; // 0.0009

    // SSIM map: [1, 3, H, W]
    auto ssim_map = ((2.0f * mu_xy + C1) * (2.0f * sigma_xy + C2)) /
                    ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2));

    // [1, 3, H, W] -> [H, W, 3] -> mean across channels -> [H, W]
    return ssim_map.squeeze(0).permute({1, 2, 0}).mean(/*dim=*/2);
}

torch::Tensor ssim_loss(const torch::Tensor& rendered, const torch::Tensor& target,
                        int window_size) {
    return 1.0f - ssim(rendered, target, window_size).mean();
}

torch::Tensor combined_loss(const torch::Tensor& rendered, const torch::Tensor& target,
                            float lambda_) {
    return (1.0f - lambda_) * ::cugs::l1_loss(rendered, target) +
           lambda_ * ssim_loss(rendered, target);
}

} // namespace cugs
