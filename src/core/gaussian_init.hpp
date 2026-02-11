#pragma once

#include "core/gaussian.hpp"
#include "core/types.hpp"

#include <span>

namespace cugs {

/// @brief Initialize a GaussianModel from COLMAP sparse points.
///
/// Initialization follows the original 3DGS paper (Kerbl et al., 2023):
///   - Position: directly from SfM point XYZ
///   - SH DC term: from point RGB color, converted to SH space
///     (color_float / C0 - 0.5 / C0, but stored as raw coefficient)
///   - Higher SH bands: zero
///   - Opacity: inverse_sigmoid(0.1) ~ -2.197
///   - Scale: log of mean distance to k nearest neighbors (k=3)
///   - Rotation: identity quaternion [1, 0, 0, 0]
///
/// @param points Sparse 3D points from COLMAP reconstruction.
/// @param sh_degree Maximum SH degree to allocate (0..3). Default 3.
/// @param k_neighbors Number of neighbors for scale initialization. Default 3.
/// @return GaussianModel on CPU with all parameters initialized.
GaussianModel init_gaussians_from_sparse(
    std::span<const SparsePoint> points,
    int sh_degree = 3,
    int k_neighbors = 3);

} // namespace cugs
