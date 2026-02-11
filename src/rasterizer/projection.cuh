#pragma once

/// @file projection.cuh
/// @brief Device-side math helpers for projecting 3D Gaussians to 2D.
///
/// Implements the EWA splatting projection from Kerbl et al. (2023) Section 4:
///   1. Quaternion → rotation matrix
///   2. 3D covariance from rotation + scale: Σ = R S Sᵀ Rᵀ
///   3. Project 3D covariance to 2D via the local affine approximation (Zwicker 2001)
///   4. Compute pixel radius from 2D covariance eigenvalues (3σ ellipse)
///
/// All functions are __device__ __forceinline__ so they compile into the caller's
/// kernel without requiring CUDA separable compilation.

#include <cuda_runtime.h>
#include <cmath>

namespace cugs {

// ---------------------------------------------------------------------------
// Quaternion → 3×3 rotation matrix (row-major, stored as float[9])
// ---------------------------------------------------------------------------

/// @brief Convert a unit quaternion (w, x, y, z) to a 3×3 rotation matrix.
/// @param w Scalar part.
/// @param x,y,z Vector part.
/// @param R Output array of 9 floats in row-major order.
__device__ __forceinline__
void quat_to_rotation(float w, float x, float y, float z, float R[9]) {
    // Normalise
    float inv_norm = rsqrtf(w * w + x * x + y * y + z * z + 1e-12f);
    w *= inv_norm;
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;

    // Row 0
    R[0] = 1.0f - 2.0f * (y * y + z * z);
    R[1] = 2.0f * (x * y - w * z);
    R[2] = 2.0f * (x * z + w * y);
    // Row 1
    R[3] = 2.0f * (x * y + w * z);
    R[4] = 1.0f - 2.0f * (x * x + z * z);
    R[5] = 2.0f * (y * z - w * x);
    // Row 2
    R[6] = 2.0f * (x * z - w * y);
    R[7] = 2.0f * (y * z + w * x);
    R[8] = 1.0f - 2.0f * (x * x + y * y);
}

// ---------------------------------------------------------------------------
// 3D covariance from rotation + scale
// ---------------------------------------------------------------------------

/// @brief Compute the 3D covariance matrix Σ = R * S * Sᵀ * Rᵀ = M * Mᵀ
///        where M = R * diag(exp(log_scale)).
///
/// The 3×3 symmetric matrix is stored as 6 upper-triangle values:
///   cov[0] = Σ(0,0), cov[1] = Σ(0,1), cov[2] = Σ(0,2),
///   cov[3] = Σ(1,1), cov[4] = Σ(1,2), cov[5] = Σ(2,2)
///
/// @param log_scale Log-space scales [3].
/// @param rot Quaternion [4] (w, x, y, z).
/// @param cov Output 6 floats (upper triangle of symmetric matrix).
__device__ __forceinline__
void compute_cov_3d(const float log_scale[3], const float rot[4], float cov[6]) {
    // Scale: apply exp to log-space scales
    float sx = expf(log_scale[0]);
    float sy = expf(log_scale[1]);
    float sz = expf(log_scale[2]);

    // Rotation matrix
    float R[9];
    quat_to_rotation(rot[0], rot[1], rot[2], rot[3], R);

    // M = R * S  where S = diag(sx, sy, sz)
    // M[i][j] = R[i][j] * s[j]
    float M[9];
    M[0] = R[0] * sx;  M[1] = R[1] * sy;  M[2] = R[2] * sz;
    M[3] = R[3] * sx;  M[4] = R[4] * sy;  M[5] = R[5] * sz;
    M[6] = R[6] * sx;  M[7] = R[7] * sy;  M[8] = R[8] * sz;

    // Σ = M * Mᵀ — symmetric, so only compute upper triangle
    cov[0] = M[0] * M[0] + M[1] * M[1] + M[2] * M[2];  // (0,0)
    cov[1] = M[0] * M[3] + M[1] * M[4] + M[2] * M[5];  // (0,1)
    cov[2] = M[0] * M[6] + M[1] * M[7] + M[2] * M[8];  // (0,2)
    cov[3] = M[3] * M[3] + M[4] * M[4] + M[5] * M[5];  // (1,1)
    cov[4] = M[3] * M[6] + M[4] * M[7] + M[5] * M[8];  // (1,2)
    cov[5] = M[6] * M[6] + M[7] * M[7] + M[8] * M[8];  // (2,2)
}

// ---------------------------------------------------------------------------
// Project 3D covariance to 2D
// ---------------------------------------------------------------------------

/// @brief Project a 3D covariance matrix to 2D image space using the local
///        affine approximation from Zwicker et al. (2001) / Kerbl et al. (2023).
///
/// Computes: Σ' = J * W * Σ * Wᵀ * Jᵀ + 0.3·I
/// where J is the Jacobian of the perspective projection and W is the
/// world-to-camera rotation (view matrix upper-left 3×3).
///
/// Returns the 2D covariance as 3 floats: (a, b, c) for the symmetric matrix
///   [[a, b],
///    [b, c]]
///
/// @param cov_3d Upper-triangle 3D covariance [6 floats].
/// @param W      View rotation matrix [9 floats, row-major].
/// @param t      Point position in camera space [3 floats].
/// @param fx     Focal length x.
/// @param fy     Focal length y.
/// @param cov_2d Output 3 floats: (a, b, c).
__device__ __forceinline__
void compute_cov_2d(const float cov_3d[6], const float W[9],
                    const float t[3], float fx, float fy,
                    float cov_2d[3]) {
    float tx = t[0], ty = t[1], tz = t[2];

    // Guard against division by zero
    float tz_inv = 1.0f / (tz + 1e-6f);
    float tz_inv2 = tz_inv * tz_inv;

    // Jacobian of the perspective projection
    // J = [[fx/tz,  0,     -fx*tx/tz²],
    //      [0,      fy/tz, -fy*ty/tz²]]
    float J[6]; // 2×3, row-major
    J[0] = fx * tz_inv;
    J[1] = 0.0f;
    J[2] = -fx * tx * tz_inv2;
    J[3] = 0.0f;
    J[4] = fy * tz_inv;
    J[5] = -fy * ty * tz_inv2;

    // T = J * W  (2×3 = 2×3 * 3×3)
    float T[6]; // 2×3, row-major
    T[0] = J[0] * W[0] + J[1] * W[3] + J[2] * W[6];
    T[1] = J[0] * W[1] + J[1] * W[4] + J[2] * W[7];
    T[2] = J[0] * W[2] + J[1] * W[5] + J[2] * W[8];
    T[3] = J[3] * W[0] + J[4] * W[3] + J[5] * W[6];
    T[4] = J[3] * W[1] + J[4] * W[4] + J[5] * W[7];
    T[5] = J[3] * W[2] + J[4] * W[5] + J[5] * W[8];

    // Expand symmetric cov_3d to full 3×3
    // cov_3d = [Σ00, Σ01, Σ02, Σ11, Σ12, Σ22]
    float S00 = cov_3d[0], S01 = cov_3d[1], S02 = cov_3d[2];
    float S11 = cov_3d[3], S12 = cov_3d[4], S22 = cov_3d[5];

    // Compute T * Σ (2×3 * 3×3 → 2×3)
    float TS[6]; // 2×3
    TS[0] = T[0] * S00 + T[1] * S01 + T[2] * S02;
    TS[1] = T[0] * S01 + T[1] * S11 + T[2] * S12;
    TS[2] = T[0] * S02 + T[1] * S12 + T[2] * S22;
    TS[3] = T[3] * S00 + T[4] * S01 + T[5] * S02;
    TS[4] = T[3] * S01 + T[4] * S11 + T[5] * S12;
    TS[5] = T[3] * S02 + T[4] * S12 + T[5] * S22;

    // Compute (T * Σ) * Tᵀ → 2×2 symmetric
    cov_2d[0] = TS[0] * T[0] + TS[1] * T[1] + TS[2] * T[2];  // a
    cov_2d[1] = TS[0] * T[3] + TS[1] * T[4] + TS[2] * T[5];  // b
    cov_2d[2] = TS[3] * T[3] + TS[4] * T[4] + TS[5] * T[5];  // c

    // Add low-pass filter (anti-aliasing) per Kerbl et al.
    cov_2d[0] += 0.3f;
    cov_2d[2] += 0.3f;
}

// ---------------------------------------------------------------------------
// Compute pixel radius from 2D covariance
// ---------------------------------------------------------------------------

/// @brief Compute the pixel radius of the 3σ bounding circle from a 2D
///        covariance matrix [[a, b], [b, c]].
///
/// Finds the larger eigenvalue λ_max and returns ceil(3 * sqrt(λ_max)).
///
/// @param cov_2d 2D covariance (a, b, c).
/// @return Pixel radius (integer), or 0 if degenerate.
__device__ __forceinline__
int compute_radius(const float cov_2d[3]) {
    float a = cov_2d[0], b = cov_2d[1], c = cov_2d[2];

    // Eigenvalues of [[a, b], [b, c]]:
    // λ = 0.5 * ((a+c) ± sqrt((a-c)² + 4b²))
    float det = a * c - b * b;
    float trace = a + c;
    float disc = fmaxf(trace * trace - 4.0f * det, 0.0f);
    float sqrt_disc = sqrtf(disc);
    float lambda_max = 0.5f * (trace + sqrt_disc);

    if (lambda_max <= 0.0f) return 0;

    // 3σ radius
    float radius = ceilf(3.0f * sqrtf(lambda_max));
    return static_cast<int>(radius);
}

// ---------------------------------------------------------------------------
// Inverse of 2D covariance
// ---------------------------------------------------------------------------

/// @brief Compute the inverse of a 2×2 symmetric matrix [[a, b], [b, c]].
///
/// For Gaussian evaluation: G(x) = exp(-0.5 * [dx, dy] * Σ⁻¹ * [dx, dy]ᵀ)
///
/// @param cov_2d Input covariance (a, b, c).
/// @param cov_2d_inv Output inverse covariance (a_inv, b_inv, c_inv).
/// @return Determinant. If ≤ 0, the inverse is invalid.
__device__ __forceinline__
float compute_cov_2d_inverse(const float cov_2d[3], float cov_2d_inv[3]) {
    float a = cov_2d[0], b = cov_2d[1], c = cov_2d[2];
    float det = a * c - b * b;

    if (det <= 0.0f) {
        cov_2d_inv[0] = 0.0f;
        cov_2d_inv[1] = 0.0f;
        cov_2d_inv[2] = 0.0f;
        return 0.0f;
    }

    float inv_det = 1.0f / det;
    cov_2d_inv[0] =  c * inv_det;
    cov_2d_inv[1] = -b * inv_det;
    cov_2d_inv[2] =  a * inv_det;

    return det;
}

} // namespace cugs
