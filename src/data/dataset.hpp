#pragma once

#include "core/types.hpp"
#include "data/image_io.hpp"

#include <Eigen/Core>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace cugs {

/// @brief Scene bounding information computed from sparse points and cameras.
struct SceneBounds {
    Eigen::Vector3f min_bound = Eigen::Vector3f::Zero();
    Eigen::Vector3f max_bound = Eigen::Vector3f::Zero();
    Eigen::Vector3f center = Eigen::Vector3f::Zero();
    float extent = 0.0f;  // max half-extent (radius of bounding sphere)
};

/// @brief Dataset that wraps a COLMAP sparse reconstruction with images.
///
/// The constructor parses COLMAP binary files and resolves image paths, but
/// does NOT load pixel data. Use load_train_image() / load_test_image() for
/// lazy on-demand loading.
///
/// Train/test split: images are sorted by name, and every Nth image goes to
/// the test set (default N=8, matching the 3DGS convention).
class Dataset {
public:
    /// @brief Construct a dataset from a COLMAP scene directory.
    ///
    /// Expected directory layout:
    /// ```
    ///   base_path/
    ///     images/          (or images_N/ for downscaled)
    ///     sparse/0/        (cameras.bin, images.bin, points3D.bin)
    /// ```
    ///
    /// @param base_path Root directory of the dataset.
    /// @param resolution_scale Downscale factor for images (1 = original).
    /// @param test_every_n Every Nth image (sorted by name) goes to test set.
    /// @throws std::runtime_error if required files are missing.
    Dataset(const std::filesystem::path& base_path,
            int resolution_scale = 1,
            int test_every_n = 8);

    // -- Accessors --

    size_t num_train() const { return train_cameras_.size(); }
    size_t num_test() const { return test_cameras_.size(); }
    size_t num_points() const { return points_.size(); }

    const std::vector<CameraInfo>& train_cameras() const { return train_cameras_; }
    const std::vector<CameraInfo>& test_cameras() const { return test_cameras_; }
    const std::vector<SparsePoint>& sparse_points() const { return points_; }
    const SceneBounds& scene_bounds() const { return bounds_; }

    /// @brief Lazily load a training image from disk.
    /// @param index Index into train_cameras().
    /// @return Loaded image (RGB float [0,1]).
    Image load_train_image(size_t index) const;

    /// @brief Lazily load a test image from disk.
    /// @param index Index into test_cameras().
    /// @return Loaded image (RGB float [0,1]).
    Image load_test_image(size_t index) const;

    /// @brief Print a human-readable summary of the dataset.
    void print_summary() const;

private:
    std::filesystem::path base_path_;
    int resolution_scale_;

    std::vector<CameraInfo> train_cameras_;
    std::vector<CameraInfo> test_cameras_;
    std::vector<SparsePoint> points_;
    SceneBounds bounds_;

    /// @brief Resolve the images directory path (images/ or images_N/).
    std::filesystem::path resolve_images_dir() const;

    /// @brief Compute scene bounding box from sparse points and camera centers.
    void compute_bounds();
};

} // namespace cugs
