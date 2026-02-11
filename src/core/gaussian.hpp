#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace cugs {

/// @brief Maximum supported spherical harmonics degree.
constexpr int kMaxSHDegree = 3;

/// @brief Number of SH coefficients for a given degree: (degree+1)^2.
constexpr int sh_coeff_count(int degree) {
    return (degree + 1) * (degree + 1);
}

/// @brief Structure-of-Arrays storage for 3D Gaussian parameters.
///
/// All parameters are stored as contiguous torch::Tensor buffers for GPU
/// efficiency. Parameters live in their activation-function-friendly spaces:
///   - scales:    log-space (apply exp() to get actual scale)
///   - opacities: logit-space (apply sigmoid() to get [0,1] opacity)
///   - rotations: raw quaternions (normalize before use)
///
/// Tensor shapes (N = number of Gaussians):
///   - positions:  [N, 3]
///   - sh_coeffs:  [N, 3, (D+1)^2]  where D = max_sh_degree
///   - opacities:  [N, 1]
///   - rotations:  [N, 4]  (wxyz, scalar-first)
///   - scales:     [N, 3]
struct GaussianModel {
    torch::Tensor positions;   // [N, 3]      — world-space means
    torch::Tensor sh_coeffs;   // [N, 3, C]   — SH coefficients per color channel
    torch::Tensor opacities;   // [N, 1]      — logit-space opacity
    torch::Tensor rotations;   // [N, 4]      — quaternions (w, x, y, z)
    torch::Tensor scales;      // [N, 3]      — log-space scale

    /// @brief Number of Gaussians in the model.
    int64_t num_gaussians() const {
        return positions.defined() ? positions.size(0) : 0;
    }

    /// @brief Maximum SH degree this model was allocated for.
    int max_sh_degree() const {
        if (!sh_coeffs.defined()) return 0;
        // sh_coeffs shape is [N, 3, (D+1)^2]
        int num_coeffs = static_cast<int>(sh_coeffs.size(2));
        // (D+1)^2 = num_coeffs  =>  D = sqrt(num_coeffs) - 1
        int d = static_cast<int>(std::sqrt(static_cast<float>(num_coeffs))) - 1;
        return d;
    }

    /// @brief Move all tensors to the specified device.
    void to_device(torch::Device device) {
        if (positions.defined())  positions  = positions.to(device);
        if (sh_coeffs.defined())  sh_coeffs  = sh_coeffs.to(device);
        if (opacities.defined())  opacities  = opacities.to(device);
        if (rotations.defined())  rotations  = rotations.to(device);
        if (scales.defined())     scales     = scales.to(device);
    }

    /// @brief Check that all tensors are on the same device and have
    ///        consistent leading dimension.
    bool is_valid() const {
        if (!positions.defined()) return false;
        const int64_t n = positions.size(0);
        if (positions.dim() != 2 || positions.size(1) != 3) return false;
        if (!sh_coeffs.defined() || sh_coeffs.dim() != 3 ||
            sh_coeffs.size(0) != n || sh_coeffs.size(1) != 3) return false;
        if (!opacities.defined() || opacities.dim() != 2 ||
            opacities.size(0) != n || opacities.size(1) != 1) return false;
        if (!rotations.defined() || rotations.dim() != 2 ||
            rotations.size(0) != n || rotations.size(1) != 4) return false;
        if (!scales.defined() || scales.dim() != 2 ||
            scales.size(0) != n || scales.size(1) != 3) return false;

        // All on same device
        auto dev = positions.device();
        if (sh_coeffs.device() != dev || opacities.device() != dev ||
            rotations.device() != dev || scales.device() != dev) return false;

        return true;
    }

    /// @brief Save the Gaussian model to a PLY file.
    ///
    /// Uses the reference implementation's PLY format for compatibility with
    /// existing viewers. All tensors are moved to CPU before writing.
    ///
    /// @param path Output PLY file path.
    /// @return true on success.
    bool save_ply(const std::filesystem::path& path) const;

    /// @brief Load a Gaussian model from a PLY file.
    ///
    /// @param path Input PLY file path.
    /// @return Loaded model (on CPU).
    static GaussianModel load_ply(const std::filesystem::path& path);
};

} // namespace cugs
