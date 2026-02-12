/// @file projection_backward.cu
/// @brief CUDA kernel for the backward pass through the projection stage.
///
/// One thread per Gaussian, 256 threads/block. Each thread recomputes the
/// forward intermediates (t_cam, R, M, cov_3d, J, T, cov_2d, cov_2d_inv)
/// then applies the chain rule to produce gradients w.r.t. all learnable
/// Gaussian parameters.
///
/// Reference: Kerbl et al. "3D Gaussian Splatting" (SIGGRAPH 2023), supplementary.

#include "rasterizer/projection_backward.hpp"
#include "rasterizer/projection.cuh"
#include "rasterizer/backward.cuh"
#include "core/sh_backward.hpp"
#include "utils/cuda_utils.cuh"

#include <torch/torch.h>
#include <cuda_runtime.h>

namespace cugs {

/// @brief Per-Gaussian backward projection kernel.
///
/// Grid:  ((N + 255) / 256) blocks × 256 threads.
/// Each thread processes one Gaussian, recomputing all forward intermediates.
__global__ void k_project_backward(
    int n,
    const float* __restrict__ positions,
    const float* __restrict__ rotations,
    const float* __restrict__ scales,
    const float* __restrict__ opacities,
    const float* __restrict__ view_matrix,
    float fx, float fy, float cx, float cy,
    float scale_mod,
    const int* __restrict__ radii,
    const float* __restrict__ dL_dmeans_2d_in,
    const float* __restrict__ dL_dcov_2d_inv_in,
    const float* __restrict__ dL_dopacity_act_in,
    float* __restrict__ dL_dpositions_out,
    float* __restrict__ dL_drotations_out,
    float* __restrict__ dL_dscales_out,
    float* __restrict__ dL_dopacities_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Skip culled Gaussians — all gradients remain zero
    if (radii[idx] <= 0) return;

    // -----------------------------------------------------------------------
    // Recompute forward intermediates
    // -----------------------------------------------------------------------

    // Load position
    float px = positions[idx * 3 + 0];
    float py = positions[idx * 3 + 1];
    float pz = positions[idx * 3 + 2];

    // View rotation (upper-left 3×3, row-major)
    float W[9];
    W[0] = view_matrix[0]; W[1] = view_matrix[1]; W[2] = view_matrix[2];
    W[3] = view_matrix[4]; W[4] = view_matrix[5]; W[5] = view_matrix[6];
    W[6] = view_matrix[8]; W[7] = view_matrix[9]; W[8] = view_matrix[10];

    // Camera-space position
    float t_cam[3];
    t_cam[0] = W[0] * px + W[1] * py + W[2] * pz + view_matrix[3];
    t_cam[1] = W[3] * px + W[4] * py + W[5] * pz + view_matrix[7];
    t_cam[2] = W[6] * px + W[7] * py + W[8] * pz + view_matrix[11];

    // Load scale and rotation
    float log_scale[3] = {
        scales[idx * 3 + 0] + logf(scale_mod + 1e-8f),
        scales[idx * 3 + 1] + logf(scale_mod + 1e-8f),
        scales[idx * 3 + 2] + logf(scale_mod + 1e-8f)
    };
    float rot[4] = {
        rotations[idx * 4 + 0],
        rotations[idx * 4 + 1],
        rotations[idx * 4 + 2],
        rotations[idx * 4 + 3]
    };

    // Recompute 3D covariance
    float cov_3d[6];
    compute_cov_3d(log_scale, rot, cov_3d);

    // Recompute 2D covariance
    float cov2d[3];
    compute_cov_2d(cov_3d, W, t_cam, fx, fy, cov2d);

    // Recompute inverse
    float cov2d_inv[3];
    float det = compute_cov_2d_inverse(cov2d, cov2d_inv);
    if (det <= 0.0f) return;

    // Recompute T = J * W
    float tz_inv = 1.0f / (t_cam[2] + 1e-6f);
    float tz_inv2 = tz_inv * tz_inv;

    float J[6]; // 2×3
    J[0] = fx * tz_inv;
    J[1] = 0.0f;
    J[2] = -fx * t_cam[0] * tz_inv2;
    J[3] = 0.0f;
    J[4] = fy * tz_inv;
    J[5] = -fy * t_cam[1] * tz_inv2;

    float T_mat[6]; // 2×3
    T_mat[0] = J[0] * W[0] + J[2] * W[6];
    T_mat[1] = J[0] * W[1] + J[2] * W[7];
    T_mat[2] = J[0] * W[2] + J[2] * W[8];
    T_mat[3] = J[4] * W[3] + J[5] * W[6];
    T_mat[4] = J[4] * W[4] + J[5] * W[7];
    T_mat[5] = J[4] * W[5] + J[5] * W[8];

    // Recompute M = R * S
    float sx = expf(log_scale[0]);
    float sy = expf(log_scale[1]);
    float sz = expf(log_scale[2]);

    float R[9];
    quat_to_rotation(rot[0], rot[1], rot[2], rot[3], R);

    float M[9];
    M[0] = R[0] * sx;  M[1] = R[1] * sy;  M[2] = R[2] * sz;
    M[3] = R[3] * sx;  M[4] = R[4] * sy;  M[5] = R[5] * sz;
    M[6] = R[6] * sx;  M[7] = R[7] * sy;  M[8] = R[8] * sz;

    // -----------------------------------------------------------------------
    // Load incoming gradients
    // -----------------------------------------------------------------------
    float dL_dcov2d_inv[3] = {
        dL_dcov_2d_inv_in[idx * 3 + 0],
        dL_dcov_2d_inv_in[idx * 3 + 1],
        dL_dcov_2d_inv_in[idx * 3 + 2]
    };
    float dL_dmean2d[2] = {
        dL_dmeans_2d_in[idx * 2 + 0],
        dL_dmeans_2d_in[idx * 2 + 1]
    };
    float dL_dopa_act = dL_dopacity_act_in[idx];

    // -----------------------------------------------------------------------
    // 1. dL/d(cov_2d_inv) → dL/d(cov_2d)
    // -----------------------------------------------------------------------
    float dL_dcov2d[3];
    compute_dL_dcov2d_from_dL_dcov2d_inv(cov2d_inv, dL_dcov2d_inv, dL_dcov2d);

    // -----------------------------------------------------------------------
    // 2. dL/d(cov_2d) → dL/d(cov_3d)
    // -----------------------------------------------------------------------
    float dL_dcov3d[6];
    compute_dL_dcov3d(T_mat, dL_dcov2d, dL_dcov3d);

    // -----------------------------------------------------------------------
    // 3. dL/d(cov_3d) → dL/d(M)
    // -----------------------------------------------------------------------
    float dL_dM[9];
    compute_dL_dM(dL_dcov3d, M, dL_dM);

    // -----------------------------------------------------------------------
    // 4. dL/d(M) → dL/d(R) and dL/d(scale)
    //    M = R * diag(s), so:
    //    dL/d(R_ij) = dL/d(M_ij) * s_j
    //    dL/d(s_j) = sum_i dL/d(M_ij) * R_ij
    //    For log-space: dL/d(log_scale_j) = dL/d(s_j) * s_j
    // -----------------------------------------------------------------------
    float dL_dR[9];
    dL_dR[0] = dL_dM[0] * sx;  dL_dR[1] = dL_dM[1] * sy;  dL_dR[2] = dL_dM[2] * sz;
    dL_dR[3] = dL_dM[3] * sx;  dL_dR[4] = dL_dM[4] * sy;  dL_dR[5] = dL_dM[5] * sz;
    dL_dR[6] = dL_dM[6] * sx;  dL_dR[7] = dL_dM[7] * sy;  dL_dR[8] = dL_dM[8] * sz;

    // dL/d(scale_j) = sum_i dL/d(M_ij) * R_ij
    float dL_ds[3];
    dL_ds[0] = dL_dM[0] * R[0] + dL_dM[3] * R[3] + dL_dM[6] * R[6];
    dL_ds[1] = dL_dM[1] * R[1] + dL_dM[4] * R[4] + dL_dM[7] * R[7];
    dL_ds[2] = dL_dM[2] * R[2] + dL_dM[5] * R[5] + dL_dM[8] * R[8];

    // Chain through exp: dL/d(log_scale) = dL/d(s) * s
    float dL_dlog_scale[3];
    dL_dlog_scale[0] = dL_ds[0] * sx;
    dL_dlog_scale[1] = dL_ds[1] * sy;
    dL_dlog_scale[2] = dL_ds[2] * sz;

    // -----------------------------------------------------------------------
    // 5. dL/d(R) → dL/d(quaternion)
    // -----------------------------------------------------------------------
    float dL_dquat[4];
    compute_dL_dquat(rot, dL_dR, dL_dquat);

    // -----------------------------------------------------------------------
    // 6. dL/d(means_2d) → dL/d(t_cam) → dL/d(position)
    //    means_2d = [fx * tx/tz + cx, fy * ty/tz + cy]
    //    d(mean_x)/d(tx) = fx/tz
    //    d(mean_x)/d(tz) = -fx*tx/tz²
    //    d(mean_y)/d(ty) = fy/tz
    //    d(mean_y)/d(tz) = -fy*ty/tz²
    // -----------------------------------------------------------------------
    float dL_dt_cam[3] = {0.0f, 0.0f, 0.0f};

    dL_dt_cam[0] += dL_dmean2d[0] * fx * tz_inv;
    dL_dt_cam[1] += dL_dmean2d[1] * fy * tz_inv;
    dL_dt_cam[2] += dL_dmean2d[0] * (-fx * t_cam[0] * tz_inv2)
                  + dL_dmean2d[1] * (-fy * t_cam[1] * tz_inv2);

    // -----------------------------------------------------------------------
    // 7. dL/d(cov_2d) → dL/d(t_cam) contribution through J's t_cam dependence
    // -----------------------------------------------------------------------
    compute_dL_dt_cam_from_cov(dL_dcov2d, cov_3d, W, t_cam, fx, fy, dL_dt_cam);

    // -----------------------------------------------------------------------
    // 8. dL/d(t_cam) → dL/d(position)
    //    t_cam = W @ pos + t  ⟹  dL/d(pos) = W^T @ dL/d(t_cam)
    // -----------------------------------------------------------------------
    float dL_dpos[3];
    dL_dpos[0] = W[0] * dL_dt_cam[0] + W[3] * dL_dt_cam[1] + W[6] * dL_dt_cam[2];
    dL_dpos[1] = W[1] * dL_dt_cam[0] + W[4] * dL_dt_cam[1] + W[7] * dL_dt_cam[2];
    dL_dpos[2] = W[2] * dL_dt_cam[0] + W[5] * dL_dt_cam[1] + W[8] * dL_dt_cam[2];

    // -----------------------------------------------------------------------
    // 9. dL/d(opacity_act) → dL/d(opacity_logit)
    //    opacity_act = sigmoid(logit)
    //    d(sigmoid)/d(logit) = sigmoid * (1 - sigmoid)
    // -----------------------------------------------------------------------
    float logit_o = opacities[idx];
    float sigmoid_o = 1.0f / (1.0f + expf(-logit_o));
    float dL_dlogit = dL_dopa_act * sigmoid_o * (1.0f - sigmoid_o);

    // -----------------------------------------------------------------------
    // Write outputs
    // -----------------------------------------------------------------------
    dL_dpositions_out[idx * 3 + 0] = dL_dpos[0];
    dL_dpositions_out[idx * 3 + 1] = dL_dpos[1];
    dL_dpositions_out[idx * 3 + 2] = dL_dpos[2];

    dL_drotations_out[idx * 4 + 0] = dL_dquat[0];
    dL_drotations_out[idx * 4 + 1] = dL_dquat[1];
    dL_drotations_out[idx * 4 + 2] = dL_dquat[2];
    dL_drotations_out[idx * 4 + 3] = dL_dquat[3];

    dL_dscales_out[idx * 3 + 0] = dL_dlog_scale[0];
    dL_dscales_out[idx * 3 + 1] = dL_dlog_scale[1];
    dL_dscales_out[idx * 3 + 2] = dL_dlog_scale[2];

    dL_dopacities_out[idx] = dL_dlogit;
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

ProjectionBackwardOutput project_backward(
    const torch::Tensor& dL_dmeans_2d,
    const torch::Tensor& dL_dcov_2d_inv,
    const torch::Tensor& dL_drgb,
    const torch::Tensor& dL_dopacity_act,
    const torch::Tensor& positions,
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_coeffs,
    const torch::Tensor& radii,
    const CameraInfo& camera,
    int active_sh_degree,
    float scale_modifier)
{
    TORCH_CHECK(positions.is_cuda(), "positions must be on CUDA");

    const int n = static_cast<int>(positions.size(0));
    auto device = positions.device();
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    // Allocate outputs
    auto dL_dpositions = torch::zeros({n, 3}, opts_f);
    auto dL_drotations = torch::zeros({n, 4}, opts_f);
    auto dL_dscales    = torch::zeros({n, 3}, opts_f);
    auto dL_dopacities = torch::zeros({n, 1}, opts_f);

    if (n == 0) {
        return {dL_dpositions, dL_drotations, dL_dscales, dL_dopacities,
                torch::zeros_like(sh_coeffs)};
    }

    // Build row-major view matrix
    Eigen::Matrix4f w2c = camera.world_to_camera();
    float view_matrix[16];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            view_matrix[r * 4 + c] = w2c(r, c);

    auto view_matrix_t = torch::from_blob(view_matrix, {16}, torch::kFloat32)
                             .to(device);

    // Ensure inputs are contiguous
    auto pos_c = positions.contiguous().to(torch::kFloat32);
    auto rot_c = rotations.contiguous().to(torch::kFloat32);
    auto scl_c = scales.contiguous().to(torch::kFloat32);
    auto opa_c = opacities.contiguous().to(torch::kFloat32);
    auto radii_c = radii.contiguous();
    auto dL_m2d_c = dL_dmeans_2d.contiguous();
    auto dL_cov_c = dL_dcov_2d_inv.contiguous();
    auto dL_opa_c = dL_dopacity_act.contiguous();

    constexpr int kBlockSize = 256;
    int num_blocks = (n + kBlockSize - 1) / kBlockSize;

    k_project_backward<<<num_blocks, kBlockSize>>>(
        n,
        pos_c.data_ptr<float>(),
        rot_c.data_ptr<float>(),
        scl_c.data_ptr<float>(),
        opa_c.data_ptr<float>(),
        view_matrix_t.data_ptr<float>(),
        camera.intrinsics.fx, camera.intrinsics.fy,
        camera.intrinsics.cx, camera.intrinsics.cy,
        scale_modifier,
        radii_c.data_ptr<int>(),
        dL_m2d_c.data_ptr<float>(),
        dL_cov_c.data_ptr<float>(),
        dL_opa_c.data_ptr<float>(),
        dL_dpositions.data_ptr<float>(),
        dL_drotations.data_ptr<float>(),
        dL_dscales.data_ptr<float>(),
        dL_dopacities.data_ptr<float>());

    CUDA_CHECK(cudaGetLastError());

    // -----------------------------------------------------------------------
    // SH backward: dL/d(rgb) → dL/d(sh_coeffs)
    // -----------------------------------------------------------------------
    Eigen::Vector3f cam_center = camera.camera_center();
    auto cam_center_t = torch::tensor(
        {cam_center.x(), cam_center.y(), cam_center.z()}, opts_f);

    auto directions = pos_c - cam_center_t.unsqueeze(0);
    auto norms = directions.norm(2, /*dim=*/1, /*keepdim=*/true).clamp_min(1e-8f);
    directions = directions / norms;

    auto dL_dsh_coeffs = evaluate_sh_backward_cuda(
        active_sh_degree, sh_coeffs, directions, dL_drgb);

    return {dL_dpositions, dL_drotations, dL_dscales, dL_dopacities, dL_dsh_coeffs};
}

} // namespace cugs
