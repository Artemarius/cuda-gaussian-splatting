#pragma once

/// @file backward.cuh
/// @brief Device-side helper functions for the backward pass chain rules.
///
/// Five __device__ __forceinline__ functions implementing the gradients through:
///   1. 2D covariance inverse → 2D covariance
///   2. 2D covariance → 3D covariance (via T = J*W projection)
///   3. 3D covariance → M matrix (Σ = M * Mᵀ)
///   4. Rotation matrix → quaternion
///   5. 2D covariance → camera-space position (through Jacobian J's t_cam dependence)
///
/// Reference: Kerbl et al. "3D Gaussian Splatting" (SIGGRAPH 2023), supplementary.

#include <cuda_runtime.h>
#include <cmath>

namespace cugs {

// ---------------------------------------------------------------------------
// 1. dL/d(cov2d_inv) → dL/d(cov2d)
// ---------------------------------------------------------------------------

/// @brief Propagate gradient through the 2×2 symmetric matrix inverse.
///
/// For S_inv = inverse(S), the derivative is:
///   dL/dS = -S_inv^T @ dL/dS_inv @ S_inv^T
/// Since both S and S_inv are symmetric, S_inv^T = S_inv:
///   dL/dS = -S_inv @ dL/dS_inv @ S_inv
///
/// Storage: symmetric 2×2 as (a, b, c) → [[a, b], [b, c]].
///
/// @param cov2d_inv Inverse 2D covariance (a_inv, b_inv, c_inv).
/// @param dL_dcov2d_inv Gradient w.r.t. inverse covariance [3].
/// @param dL_dcov2d Output gradient w.r.t. covariance [3].
__device__ __forceinline__
void compute_dL_dcov2d_from_dL_dcov2d_inv(
    const float cov2d_inv[3],
    const float dL_dcov2d_inv[3],
    float dL_dcov2d[3])
{
    // S_inv = [[a, b], [b, c]]
    float a = cov2d_inv[0], b = cov2d_inv[1], c = cov2d_inv[2];

    // The rasterizer backward stores the off-diagonal gradient in "combined"
    // format: dL_dcov2d_inv[1] = dL/d(M_01) + dL/d(M_10).  When expanding
    // to the full symmetric 2×2 matrix [[da, db], [db, dc]], we must halve
    // the off-diagonal to avoid double-counting.
    float da = dL_dcov2d_inv[0];
    float db = dL_dcov2d_inv[1] * 0.5f;
    float dc = dL_dcov2d_inv[2];

    // -S_inv @ dS_inv @ S_inv (2×2 symmetric)
    // First: tmp = S_inv @ dS_inv
    float tmp00 = a * da + b * db;
    float tmp01 = a * db + b * dc;
    float tmp10 = b * da + c * db;
    float tmp11 = b * db + c * dc;

    // Then: result = -tmp @ S_inv
    dL_dcov2d[0] = -(tmp00 * a + tmp01 * b);  // (0,0)
    dL_dcov2d[1] = -(tmp00 * b + tmp01 * c);  // (0,1)
    dL_dcov2d[2] = -(tmp10 * b + tmp11 * c);  // (1,1)
}

// ---------------------------------------------------------------------------
// 2. dL/d(cov2d) → dL/d(cov3d)
// ---------------------------------------------------------------------------

/// @brief Propagate gradient from 2D covariance to 3D covariance.
///
/// Forward: cov_2d = T @ Σ_3d @ T^T (where T = J * W, 2×3 matrix)
/// Backward: dL/d(Σ_3d) = T^T @ dL/d(cov_2d) @ T
///
/// The 3D covariance is symmetric, stored as 6 upper-triangle values:
///   [Σ00, Σ01, Σ02, Σ11, Σ12, Σ22]
///
/// @param T_mat The 2×3 projection matrix (J * W), row-major [6 floats].
/// @param dL_dcov2d Gradient w.r.t. 2D covariance [3] (a, b, c).
/// @param dL_dcov3d Output gradient w.r.t. 3D covariance [6].
__device__ __forceinline__
void compute_dL_dcov3d(
    const float T_mat[6],
    const float dL_dcov2d[3],
    float dL_dcov3d[6])
{
    // dL/d(cov_2d) as full 2×2: [[da, db], [db, dc]]
    float da = dL_dcov2d[0], db = dL_dcov2d[1], dc = dL_dcov2d[2];

    // T^T @ dL_d(cov2d) = 3×2 @ 2×2 → 3×2
    // T_mat row-major: T[0..2] = row 0, T[3..5] = row 1
    float TtD[6]; // 3×2
    TtD[0] = T_mat[0] * da + T_mat[3] * db;  // row0, col0
    TtD[1] = T_mat[0] * db + T_mat[3] * dc;  // row0, col1
    TtD[2] = T_mat[1] * da + T_mat[4] * db;  // row1, col0
    TtD[3] = T_mat[1] * db + T_mat[4] * dc;  // row1, col1
    TtD[4] = T_mat[2] * da + T_mat[5] * db;  // row2, col0
    TtD[5] = T_mat[2] * db + T_mat[5] * dc;  // row2, col1

    // (T^T @ dL) @ T = 3×2 @ 2×3 → 3×3, but we only need upper triangle
    dL_dcov3d[0] = TtD[0] * T_mat[0] + TtD[1] * T_mat[3];  // (0,0)
    dL_dcov3d[1] = TtD[0] * T_mat[1] + TtD[1] * T_mat[4];  // (0,1)
    dL_dcov3d[2] = TtD[0] * T_mat[2] + TtD[1] * T_mat[5];  // (0,2)
    dL_dcov3d[3] = TtD[2] * T_mat[1] + TtD[3] * T_mat[4];  // (1,1)
    dL_dcov3d[4] = TtD[2] * T_mat[2] + TtD[3] * T_mat[5];  // (1,2)
    dL_dcov3d[5] = TtD[4] * T_mat[2] + TtD[5] * T_mat[5];  // (2,2)
}

// ---------------------------------------------------------------------------
// 3. dL/d(cov3d) → dL/d(M)
// ---------------------------------------------------------------------------

/// @brief Propagate gradient from 3D covariance to the M = R*S matrix.
///
/// Forward: Σ = M @ M^T (symmetric)
/// Backward: dL/d(M) = (dL/dΣ + dL/dΣ^T) @ M = 2 * dL/dΣ_full @ M
/// where dL/dΣ_full is the full 3×3 (not just upper triangle).
///
/// @param dL_dcov3d Gradient w.r.t. 3D covariance [6] (upper triangle).
/// @param M         The M = R * diag(exp(log_scale)) matrix [9], row-major.
/// @param dL_dM     Output gradient w.r.t. M [9], row-major.
__device__ __forceinline__
void compute_dL_dM(
    const float dL_dcov3d[6],
    const float M[9],
    float dL_dM[9])
{
    // Expand to full symmetric 3×3
    // dL_dcov3d = [d00, d01, d02, d11, d12, d22]
    // For off-diagonal: dL/dΣ_ij gets multiplied by 2 since Σ_ij = Σ_ji
    // dL/dM = 2 * dΣ_full @ M, where dΣ_full has d01 in both (0,1) and (1,0)

    float d00 = dL_dcov3d[0];
    float d01 = dL_dcov3d[1];
    float d02 = dL_dcov3d[2];
    float d11 = dL_dcov3d[3];
    float d12 = dL_dcov3d[4];
    float d22 = dL_dcov3d[5];

    // dL/dM = 2 * [[d00, d01, d02], [d01, d11, d12], [d02, d12, d22]] @ M
    for (int i = 0; i < 3; ++i) {
        float row[3];
        if (i == 0)      { row[0] = d00; row[1] = d01; row[2] = d02; }
        else if (i == 1) { row[0] = d01; row[1] = d11; row[2] = d12; }
        else             { row[0] = d02; row[1] = d12; row[2] = d22; }

        for (int j = 0; j < 3; ++j) {
            dL_dM[i * 3 + j] = 2.0f * (row[0] * M[0 * 3 + j] +
                                         row[1] * M[1 * 3 + j] +
                                         row[2] * M[2 * 3 + j]);
        }
    }
}

// ---------------------------------------------------------------------------
// 4. dL/d(R) → dL/d(quaternion)
// ---------------------------------------------------------------------------

/// @brief Propagate gradient from the rotation matrix to quaternion (w,x,y,z).
///
/// Computes dL/d(w,x,y,z) from dL/d(R) using explicit derivatives of
/// R w.r.t. the *normalised* quaternion, plus the normalisation Jacobian.
///
/// @param rot       Input quaternion [4] = (w, x, y, z), unnormalised.
/// @param dL_dR     Gradient w.r.t. rotation matrix [9], row-major.
/// @param dL_dquat  Output gradient w.r.t. quaternion [4].
__device__ __forceinline__
void compute_dL_dquat(
    const float rot[4],
    const float dL_dR[9],
    float dL_dquat[4])
{
    // Normalise
    float w = rot[0], x = rot[1], y = rot[2], z = rot[3];
    float inv_norm = rsqrtf(w * w + x * x + y * y + z * z + 1e-12f);
    w *= inv_norm;
    x *= inv_norm;
    y *= inv_norm;
    z *= inv_norm;

    // R = [[1-2(yy+zz), 2(xy-wz),   2(xz+wy)  ],
    //      [2(xy+wz),   1-2(xx+zz), 2(yz-wx)  ],
    //      [2(xz-wy),   2(yz+wx),   1-2(xx+yy)]]
    //
    // dR/dw:  [[0,    -2z,  2y ],
    //          [2z,    0,  -2x ],
    //          [-2y,   2x,  0  ]]
    // dR/dx:  [[0,     2y,  2z ],
    //          [2y,   -4x,  -2w],
    //          [2z,    2w, -4x ]]
    // dR/dy:  [[-4y,   2x,  2w ],
    //          [2x,    0,   2z ],
    //          [-2w,   2z, -4y ]]
    // dR/dz:  [[-4z,  -2w,  2x ],
    //          [2w,   -4z,  2y ],
    //          [2x,    2y,  0  ]]

    // dL/dw = sum_ij dL/dR_ij * dR_ij/dw
    float dL_dw_norm = 2.0f * (
        -z * dL_dR[1] + y * dL_dR[2] +
         z * dL_dR[3] - x * dL_dR[5] +
        -y * dL_dR[6] + x * dL_dR[7]);

    float dL_dx_norm = 2.0f * (
         y * dL_dR[1] + z * dL_dR[2] +
         y * dL_dR[3] - 2.0f * x * dL_dR[4] - w * dL_dR[5] +
         z * dL_dR[6] + w * dL_dR[7] - 2.0f * x * dL_dR[8]);

    float dL_dy_norm = 2.0f * (
        -2.0f * y * dL_dR[0] + x * dL_dR[1] + w * dL_dR[2] +
         x * dL_dR[3] + z * dL_dR[5] +
        -w * dL_dR[6] + z * dL_dR[7] - 2.0f * y * dL_dR[8]);

    float dL_dz_norm = 2.0f * (
        -2.0f * z * dL_dR[0] - w * dL_dR[1] + x * dL_dR[2] +
         w * dL_dR[3] - 2.0f * z * dL_dR[4] + y * dL_dR[5] +
         x * dL_dR[6] + y * dL_dR[7]);

    // Chain through normalisation: d(q_norm)/d(q_raw) = (I - q_norm @ q_norm^T) / ||q||
    // dL/d(q_raw_i) = inv_norm * (dL/d(q_norm_i) - q_norm_i * dot(dL/d(q_norm), q_norm))
    float dot_val = dL_dw_norm * w + dL_dx_norm * x + dL_dy_norm * y + dL_dz_norm * z;

    dL_dquat[0] = inv_norm * (dL_dw_norm - w * dot_val);
    dL_dquat[1] = inv_norm * (dL_dx_norm - x * dot_val);
    dL_dquat[2] = inv_norm * (dL_dy_norm - y * dot_val);
    dL_dquat[3] = inv_norm * (dL_dz_norm - z * dot_val);
}

// ---------------------------------------------------------------------------
// 5. dL/d(cov2d) → dL/d(t_cam) through the Jacobian J
// ---------------------------------------------------------------------------

/// @brief Gradient of 2D covariance w.r.t. camera-space position t_cam,
///        through the dependence of the Jacobian J on t_cam.
///
/// The Jacobian J of perspective projection depends on t_cam:
///   J = [[fx/tz,  0,     -fx*tx/tz²],
///        [0,      fy/tz, -fy*ty/tz²]]
/// so cov_2d depends on t_cam through J in T = J * W.
///
/// @param dL_dcov2d Gradient w.r.t. 2D covariance [3] (a, b, c).
/// @param cov_3d    3D covariance [6] (upper triangle).
/// @param W         View rotation [9] (row-major).
/// @param t_cam     Camera-space position [3].
/// @param fx, fy    Focal lengths.
/// @param dL_dt_cam Output contribution to dL/d(t_cam) [3] (added, not overwritten).
__device__ __forceinline__
void compute_dL_dt_cam_from_cov(
    const float dL_dcov2d[3],
    const float cov_3d[6],
    const float W[9],
    const float t_cam[3],
    float fx, float fy,
    float dL_dt_cam[3])
{
    float tx = t_cam[0], ty = t_cam[1], tz = t_cam[2];
    float tz_inv = 1.0f / (tz + 1e-6f);
    float tz_inv2 = tz_inv * tz_inv;

    // J = [[fx/tz, 0, -fx*tx/tz²], [0, fy/tz, -fy*ty/tz²]]
    float J[6];
    J[0] = fx * tz_inv;
    J[1] = 0.0f;
    J[2] = -fx * tx * tz_inv2;
    J[3] = 0.0f;
    J[4] = fy * tz_inv;
    J[5] = -fy * ty * tz_inv2;

    // T = J * W
    float T[6];
    T[0] = J[0] * W[0] + J[2] * W[6];
    T[1] = J[0] * W[1] + J[2] * W[7];
    T[2] = J[0] * W[2] + J[2] * W[8];
    T[3] = J[4] * W[3] + J[5] * W[6];
    T[4] = J[4] * W[4] + J[5] * W[7];
    T[5] = J[4] * W[5] + J[5] * W[8];

    // Expand cov_3d to full symmetric
    float S00 = cov_3d[0], S01 = cov_3d[1], S02 = cov_3d[2];
    float S11 = cov_3d[3], S12 = cov_3d[4], S22 = cov_3d[5];

    // cov_2d = T @ Σ @ T^T. We need dL/d(J) which chains through T = J * W.
    // dL/d(T) comes from: cov_2d_ij = sum_kl T_ik * Sigma_kl * T_jl
    // dL/d(T_mn) = sum_ij dL/d(cov2d_ij) * (delta_im * sum_l Sigma_nl * T_jl +
    //                                         delta_jm * sum_k T_ik * Sigma_kn)
    // For symmetric cov_2d with (a,b,c) storage:
    // dL/d(T) row m = 2 * dL_dcov2d_full[m,:] @ T @ Sigma^T  ... wait let me be more careful.

    // Let's compute dL/d(T) directly:
    // cov_2d = T @ Σ @ T^T
    // dL/d(T) = dL/d(cov_2d_full) @ T @ Σ + dL/d(cov_2d_full)^T @ T @ Σ
    // Since cov_2d and dL/dcov_2d are symmetric:
    // dL/d(T) = 2 * dL_d(cov_2d_full) @ T @ Σ

    // First: Σ @ T^T (3×3 @ 3×2 → 3×2)... actually let's go:
    // T @ Σ (2×3 @ 3×3 → 2×3)
    float TS[6];
    TS[0] = T[0] * S00 + T[1] * S01 + T[2] * S02;
    TS[1] = T[0] * S01 + T[1] * S11 + T[2] * S12;
    TS[2] = T[0] * S02 + T[1] * S12 + T[2] * S22;
    TS[3] = T[3] * S00 + T[4] * S01 + T[5] * S02;
    TS[4] = T[3] * S01 + T[4] * S11 + T[5] * S12;
    TS[5] = T[3] * S02 + T[4] * S12 + T[5] * S22;

    // dL/d(cov_2d) full 2×2: [[da, db], [db, dc]]
    float da = dL_dcov2d[0], db = dL_dcov2d[1], dc = dL_dcov2d[2];

    // dL/d(T) = 2 * [[da, db], [db, dc]] @ [[TS0, TS1, TS2], [TS3, TS4, TS5]]
    float dL_dT[6];
    dL_dT[0] = 2.0f * (da * TS[0] + db * TS[3]);
    dL_dT[1] = 2.0f * (da * TS[1] + db * TS[4]);
    dL_dT[2] = 2.0f * (da * TS[2] + db * TS[5]);
    dL_dT[3] = 2.0f * (db * TS[0] + dc * TS[3]);
    dL_dT[4] = 2.0f * (db * TS[1] + dc * TS[4]);
    dL_dT[5] = 2.0f * (db * TS[2] + dc * TS[5]);

    // T = J * W, so dL/d(J) = dL/d(T) @ W^T
    float dL_dJ[6]; // 2×3
    dL_dJ[0] = dL_dT[0] * W[0] + dL_dT[1] * W[1] + dL_dT[2] * W[2];
    dL_dJ[1] = dL_dT[0] * W[3] + dL_dT[1] * W[4] + dL_dT[2] * W[5];
    dL_dJ[2] = dL_dT[0] * W[6] + dL_dT[1] * W[7] + dL_dT[2] * W[8];
    dL_dJ[3] = dL_dT[3] * W[0] + dL_dT[4] * W[1] + dL_dT[5] * W[2];
    dL_dJ[4] = dL_dT[3] * W[3] + dL_dT[4] * W[4] + dL_dT[5] * W[5];
    dL_dJ[5] = dL_dT[3] * W[6] + dL_dT[4] * W[7] + dL_dT[5] * W[8];

    // J depends on t_cam:
    // J[0,0] = fx/tz       → dJ00/dtx = 0,  dJ00/dty = 0,  dJ00/dtz = -fx/tz²
    // J[0,2] = -fx*tx/tz²  → dJ02/dtx = -fx/tz²,  dJ02/dty = 0,  dJ02/dtz = 2*fx*tx/tz³
    // J[1,1] = fy/tz       → dJ11/dtx = 0,  dJ11/dty = 0,  dJ11/dtz = -fy/tz²
    // J[1,2] = -fy*ty/tz²  → dJ12/dtx = 0,  dJ12/dty = -fy/tz²,  dJ12/dtz = 2*fy*ty/tz³

    float tz_inv3 = tz_inv2 * tz_inv;

    // dL/d(tx) = dL/dJ[0,2] * dJ02/dtx = dL_dJ[2] * (-fx/tz²)
    dL_dt_cam[0] += dL_dJ[2] * (-fx * tz_inv2);

    // dL/d(ty) = dL/dJ[1,2] * dJ12/dty = dL_dJ[5] * (-fy/tz²)
    dL_dt_cam[1] += dL_dJ[5] * (-fy * tz_inv2);

    // dL/d(tz) = dL/dJ[0,0] * (-fx/tz²) + dL/dJ[0,2] * (2*fx*tx/tz³)
    //          + dL/dJ[1,1] * (-fy/tz²) + dL/dJ[1,2] * (2*fy*ty/tz³)
    dL_dt_cam[2] += dL_dJ[0] * (-fx * tz_inv2)
                  + dL_dJ[2] * (2.0f * fx * tx * tz_inv3)
                  + dL_dJ[4] * (-fy * tz_inv2)
                  + dL_dJ[5] * (2.0f * fy * ty * tz_inv3);
}

} // namespace cugs
