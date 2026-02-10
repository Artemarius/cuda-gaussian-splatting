#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace cugs {

/// @brief A loaded image with float pixel data in [0, 1].
struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;  // 3 (RGB) or 4 (RGBA)
    std::vector<float> data;  // row-major, interleaved channels: [R,G,B,(A),R,G,B,(A),...]

    /// @brief Total number of float elements (width * height * channels).
    size_t size() const { return data.size(); }

    /// @brief Check if the image contains valid data.
    bool valid() const { return width > 0 && height > 0 && !data.empty(); }
};

/// @brief Load an image from disk and convert to float [0,1] RGB.
///
/// If the source has an alpha channel, it is discarded (output is always 3-channel RGB).
///
/// @param path Path to the image file (JPEG, PNG, BMP, TGA, etc.).
/// @return Loaded image.
/// @throws std::runtime_error if the file cannot be read or decoded.
Image load_image(const std::filesystem::path& path);

/// @brief Resize an image using simple bilinear interpolation.
///
/// @param src Source image.
/// @param target_width Desired width.
/// @param target_height Desired height.
/// @return Resized image.
Image resize_image(const Image& src, int target_width, int target_height);

/// @brief Load an image from disk, optionally downscaling it.
///
/// If resolution_scale <= 1, the image is returned at original resolution.
/// Otherwise, both dimensions are divided by resolution_scale.
///
/// @param path Path to the image file.
/// @param resolution_scale Downscale factor (1 = original, 2 = half, 4 = quarter).
/// @return Loaded (and possibly resized) image.
/// @throws std::runtime_error if the file cannot be read or decoded.
Image load_image_resized(const std::filesystem::path& path, int resolution_scale = 1);

} // namespace cugs
