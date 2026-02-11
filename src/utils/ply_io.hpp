#pragma once

#include "core/types.hpp"

#include <filesystem>
#include <span>

namespace cugs {

/// @brief Write sparse 3D points to a binary little-endian PLY file.
///
/// The output can be opened in MeshLab, CloudCompare, or any PLY viewer for
/// visual verification of the COLMAP reconstruction.
///
/// @param path Output file path (will be overwritten if it exists).
/// @param points Sparse points with position and RGB color.
/// @return true on success, false on I/O error.
bool write_points_ply(const std::filesystem::path& path,
                      std::span<const SparsePoint> points);

/// @brief Write camera centers as colored points to a binary PLY file.
///
/// Useful for verifying that camera positions form a reasonable pattern
/// around the scene.  All cameras are written with the given RGB color.
///
/// @param path Output file path.
/// @param cameras Camera info array (camera_center() is used for position).
/// @param r Red channel (0-255) for camera markers.
/// @param g Green channel (0-255) for camera markers.
/// @param b Blue channel (0-255) for camera markers.
/// @return true on success, false on I/O error.
bool write_cameras_ply(const std::filesystem::path& path,
                       std::span<const CameraInfo> cameras,
                       uint8_t r = 255, uint8_t g = 0, uint8_t b = 0);

} // namespace cugs
