#include "data/dataset.hpp"
#include "data/colmap_loader.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace cugs {

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

Dataset::Dataset(const std::filesystem::path& base_path,
                 int resolution_scale,
                 int test_every_n)
    : base_path_(base_path), resolution_scale_(resolution_scale) {

    if (!std::filesystem::exists(base_path)) {
        throw std::runtime_error("Dataset path does not exist: " +
                                 base_path.string());
    }

    // Find the sparse reconstruction directory
    // Try sparse/0/ first (COLMAP default), then sparse/
    std::filesystem::path sparse_dir = base_path / "sparse" / "0";
    if (!std::filesystem::exists(sparse_dir / "cameras.bin")) {
        sparse_dir = base_path / "sparse";
    }
    if (!std::filesystem::exists(sparse_dir / "cameras.bin")) {
        throw std::runtime_error(
            "Cannot find COLMAP sparse reconstruction in " +
            base_path.string() + " (looked for sparse/0/ and sparse/)");
    }

    // Parse COLMAP binary files
    auto colmap_data = parse_colmap_sparse(sparse_dir);
    points_ = std::move(colmap_data.points);

    // Merge cameras and images into CameraInfo
    auto all_cameras = merge_cameras_images(colmap_data.cameras,
                                            colmap_data.images);

    // Resolve image directory and set image paths
    auto images_dir = resolve_images_dir();
    for (auto& cam : all_cameras) {
        cam.image_path = (images_dir / cam.image_name).string();
    }

    // Sort all cameras by image name for deterministic split
    std::sort(all_cameras.begin(), all_cameras.end(),
              [](const CameraInfo& a, const CameraInfo& b) {
                  return a.image_name < b.image_name;
              });

    // Train/test split: every Nth image → test
    if (test_every_n <= 0) {
        // All images to train
        train_cameras_ = std::move(all_cameras);
    } else {
        for (size_t i = 0; i < all_cameras.size(); ++i) {
            if (static_cast<int>(i % static_cast<size_t>(test_every_n)) == 0) {
                test_cameras_.push_back(std::move(all_cameras[i]));
            } else {
                train_cameras_.push_back(std::move(all_cameras[i]));
            }
        }
    }

    // Adjust intrinsics for resolution scale
    if (resolution_scale > 1) {
        float scale = 1.0f / static_cast<float>(resolution_scale);
        auto adjust = [scale](CameraInfo& cam) {
            cam.width = std::max(1, static_cast<int>(cam.width * scale));
            cam.height = std::max(1, static_cast<int>(cam.height * scale));
            cam.intrinsics.fx *= scale;
            cam.intrinsics.fy *= scale;
            cam.intrinsics.cx *= scale;
            cam.intrinsics.cy *= scale;
        };
        for (auto& cam : train_cameras_) adjust(cam);
        for (auto& cam : test_cameras_) adjust(cam);
    }

    compute_bounds();
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------

Image Dataset::load_train_image(size_t index) const {
    if (index >= train_cameras_.size()) {
        throw std::runtime_error("Train image index out of bounds: " +
                                 std::to_string(index));
    }
    return load_image_resized(train_cameras_[index].image_path,
                              resolution_scale_);
}

Image Dataset::load_test_image(size_t index) const {
    if (index >= test_cameras_.size()) {
        throw std::runtime_error("Test image index out of bounds: " +
                                 std::to_string(index));
    }
    return load_image_resized(test_cameras_[index].image_path,
                              resolution_scale_);
}

// ---------------------------------------------------------------------------
// Summary
// ---------------------------------------------------------------------------

void Dataset::print_summary() const {
    spdlog::info("Dataset: {}", base_path_.string());
    spdlog::info("  Train images : {}", train_cameras_.size());
    spdlog::info("  Test images  : {}", test_cameras_.size());
    spdlog::info("  Sparse points: {}", points_.size());
    if (!train_cameras_.empty()) {
        const auto& cam = train_cameras_[0];
        spdlog::info("  Image size   : {}x{} (scale 1/{})",
                     cam.width, cam.height, resolution_scale_);
        spdlog::info("  Focal length : fx={:.1f}, fy={:.1f}",
                     cam.intrinsics.fx, cam.intrinsics.fy);
    }
    spdlog::info("  Scene center : ({:.2f}, {:.2f}, {:.2f})",
                 bounds_.center.x(), bounds_.center.y(), bounds_.center.z());
    spdlog::info("  Scene extent : {:.2f}", bounds_.extent);
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

std::filesystem::path Dataset::resolve_images_dir() const {
    // Try images_N/ first for downscaled images
    if (resolution_scale_ > 1) {
        auto scaled_dir = base_path_ / ("images_" + std::to_string(resolution_scale_));
        if (std::filesystem::exists(scaled_dir)) {
            return scaled_dir;
        }
    }

    auto images_dir = base_path_ / "images";
    if (std::filesystem::exists(images_dir)) {
        return images_dir;
    }

    // If neither exists, just return images/ and let load_image fail later
    // with a clear error about the specific file
    return base_path_ / "images";
}

void Dataset::compute_bounds() {
    if (points_.empty()) {
        spdlog::warn("No sparse points — scene bounds are undefined");
        return;
    }

    Eigen::Vector3f min_pt = Eigen::Vector3f::Constant(
        std::numeric_limits<float>::max());
    Eigen::Vector3f max_pt = Eigen::Vector3f::Constant(
        std::numeric_limits<float>::lowest());

    // Include sparse points
    for (const auto& pt : points_) {
        min_pt = min_pt.cwiseMin(pt.position);
        max_pt = max_pt.cwiseMax(pt.position);
    }

    // Include camera centers
    auto include_cameras = [&](const std::vector<CameraInfo>& cameras) {
        for (const auto& cam : cameras) {
            Eigen::Vector3f c = cam.camera_center();
            min_pt = min_pt.cwiseMin(c);
            max_pt = max_pt.cwiseMax(c);
        }
    };
    include_cameras(train_cameras_);
    include_cameras(test_cameras_);

    bounds_.min_bound = min_pt;
    bounds_.max_bound = max_pt;
    bounds_.center = (min_pt + max_pt) * 0.5f;
    bounds_.extent = (max_pt - min_pt).maxCoeff() * 0.5f;
}

} // namespace cugs
