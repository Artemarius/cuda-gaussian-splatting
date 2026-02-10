#pragma once

#include "core/types.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace cugs {

/// @brief Raw COLMAP binary parse results.
struct ColmapData {
    std::vector<ColmapCamera> cameras;
    std::vector<ColmapImage> images;
    std::vector<SparsePoint> points;
};

/// @brief Parse cameras.bin from a COLMAP sparse reconstruction.
/// @param path Path to cameras.bin file.
/// @return Vector of parsed cameras.
/// @throws std::runtime_error on I/O or format errors.
std::vector<ColmapCamera> parse_cameras_bin(const std::filesystem::path& path);

/// @brief Parse images.bin from a COLMAP sparse reconstruction.
/// @param path Path to images.bin file.
/// @return Vector of parsed images (without 2D point data, which is skipped).
/// @throws std::runtime_error on I/O or format errors.
std::vector<ColmapImage> parse_images_bin(const std::filesystem::path& path);

/// @brief Parse points3D.bin from a COLMAP sparse reconstruction.
/// @param path Path to points3D.bin file.
/// @return Vector of sparse 3D points with position, color, and error.
/// @throws std::runtime_error on I/O or format errors.
std::vector<SparsePoint> parse_points3d_bin(const std::filesystem::path& path);

/// @brief Parse all three COLMAP binary files from a sparse/ directory.
/// @param sparse_dir Path to the sparse reconstruction directory (containing
///        cameras.bin, images.bin, points3D.bin).
/// @return Combined parse results.
/// @throws std::runtime_error on I/O or format errors.
ColmapData parse_colmap_sparse(const std::filesystem::path& sparse_dir);

/// @brief Merge raw COLMAP cameras and images into pipeline-ready CameraInfo.
///
/// For each ColmapImage, looks up the corresponding ColmapCamera by ID,
/// converts the quaternion to a rotation matrix, and normalises intrinsics
/// to (fx, fy, cx, cy).
///
/// @param cameras  Raw COLMAP cameras.
/// @param images   Raw COLMAP images.
/// @return Vector of CameraInfo, one per image.
/// @throws std::runtime_error if a camera ID referenced by an image is missing.
std::vector<CameraInfo> merge_cameras_images(
    const std::vector<ColmapCamera>& cameras,
    const std::vector<ColmapImage>& images);

} // namespace cugs
