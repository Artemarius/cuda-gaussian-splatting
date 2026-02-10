#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "data/image_io.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cugs {

Image load_image(const std::filesystem::path& path) {
    int w = 0, h = 0, channels_in_file = 0;
    // Request 3 channels (RGB) regardless of source format
    constexpr int kDesiredChannels = 3;
    unsigned char* raw = stbi_load(path.string().c_str(), &w, &h,
                                   &channels_in_file, kDesiredChannels);
    if (!raw) {
        throw std::runtime_error("Failed to load image: " + path.string() +
                                 " (" + stbi_failure_reason() + ")");
    }

    Image img;
    img.width = w;
    img.height = h;
    img.channels = kDesiredChannels;

    const size_t num_pixels = static_cast<size_t>(w) * static_cast<size_t>(h);
    const size_t num_floats = num_pixels * kDesiredChannels;
    img.data.resize(num_floats);

    // Convert uint8 [0,255] → float [0,1]
    constexpr float kScale = 1.0f / 255.0f;
    for (size_t i = 0; i < num_floats; ++i) {
        img.data[i] = static_cast<float>(raw[i]) * kScale;
    }

    stbi_image_free(raw);

    spdlog::debug("Loaded image {} ({}x{}, {} ch)", path.string(), w, h,
                  channels_in_file);
    return img;
}

Image resize_image(const Image& src, int target_width, int target_height) {
    if (target_width <= 0 || target_height <= 0) {
        throw std::runtime_error("Invalid target dimensions for resize");
    }

    Image dst;
    dst.width = target_width;
    dst.height = target_height;
    dst.channels = src.channels;
    dst.data.resize(static_cast<size_t>(target_width) *
                    static_cast<size_t>(target_height) *
                    static_cast<size_t>(src.channels));

    const float x_scale = static_cast<float>(src.width) / static_cast<float>(target_width);
    const float y_scale = static_cast<float>(src.height) / static_cast<float>(target_height);
    const int ch = src.channels;

    for (int y = 0; y < target_height; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5f) * y_scale - 0.5f;
        const int y0 = std::max(0, static_cast<int>(std::floor(src_y)));
        const int y1 = std::min(src.height - 1, y0 + 1);
        const float fy = src_y - static_cast<float>(y0);

        for (int x = 0; x < target_width; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5f) * x_scale - 0.5f;
            const int x0 = std::max(0, static_cast<int>(std::floor(src_x)));
            const int x1 = std::min(src.width - 1, x0 + 1);
            const float fx = src_x - static_cast<float>(x0);

            const size_t dst_idx = (static_cast<size_t>(y) * target_width + x) * ch;

            for (int c = 0; c < ch; ++c) {
                // Bilinear interpolation
                const float v00 = src.data[(static_cast<size_t>(y0) * src.width + x0) * ch + c];
                const float v10 = src.data[(static_cast<size_t>(y0) * src.width + x1) * ch + c];
                const float v01 = src.data[(static_cast<size_t>(y1) * src.width + x0) * ch + c];
                const float v11 = src.data[(static_cast<size_t>(y1) * src.width + x1) * ch + c];

                const float top = v00 + (v10 - v00) * fx;
                const float bot = v01 + (v11 - v01) * fx;
                dst.data[dst_idx + c] = top + (bot - top) * fy;
            }
        }
    }

    return dst;
}

Image load_image_resized(const std::filesystem::path& path, int resolution_scale) {
    Image img = load_image(path);
    if (resolution_scale <= 1) {
        return img;
    }

    int target_w = std::max(1, img.width / resolution_scale);
    int target_h = std::max(1, img.height / resolution_scale);

    spdlog::debug("Resizing {}x{} -> {}x{} (scale 1/{})",
                  img.width, img.height, target_w, target_h, resolution_scale);

    return resize_image(img, target_w, target_h);
}

} // namespace cugs
