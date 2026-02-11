#pragma once

#include "core/types.hpp"

#include <filesystem>
#include <span>

// Forward declaration — avoids circular include with gaussian.hpp
namespace cugs { struct GaussianModel; }

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

/// @brief Write a GaussianModel to a binary PLY file.
///
/// Uses the reference implementation's PLY format for compatibility:
///   - x, y, z (position)
///   - nx, ny, nz (normals — written as zero, kept for compatibility)
///   - f_dc_0..f_dc_2 (DC SH coefficients, 3 channels)
///   - f_rest_0..f_rest_N (higher-order SH coefficients, interleaved)
///   - opacity
///   - scale_0..scale_2
///   - rot_0..rot_3 (quaternion wxyz)
///
/// @param path Output file path.
/// @param model GaussianModel to write. Tensors are moved to CPU internally.
/// @return true on success.
bool write_gaussian_ply(const std::filesystem::path& path,
                        const GaussianModel& model);

/// @brief Read a GaussianModel from a binary PLY file.
///
/// Parses the reference implementation's PLY format and reconstructs
/// the GaussianModel tensors on CPU.
///
/// @param path Input PLY file path.
/// @return Loaded GaussianModel on CPU.
/// @throws std::runtime_error on parse error.
GaussianModel read_gaussian_ply(const std::filesystem::path& path);

} // namespace cugs
