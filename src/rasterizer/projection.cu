/// @file projection.cu
/// @brief CUDA kernel for projecting 3D Gaussians to 2D screen space.
///
/// Implements the EWA splatting projection pipeline:
///   1. Transform Gaussian center to camera space
///   2. Frustum-cull (near plane, behind camera)
///   3. Compute 3D covariance → project to 2D covariance
///   4. Compute pixel radius from eigenvalues
///   5. Count tiles touched by each Gaussian's bounding rect
///
/// References:
///   - Kerbl et al. "3D Gaussian Splatting" (SIGGRAPH 2023), Section 4
///   - Zwicker et al. "EWA Splatting" (IEEE TVCG 2002)

#include "rasterizer/projection.cuh"
#include "rasterizer/projection.hpp"
#include "core/sh.hpp"
#include "utils/cuda_utils.cuh"

#include <torch/torch.h>
#include <cuda_runtime.h>

namespace cugs {

// ---------------------------------------------------------------------------
// Tile constants — duplicated here to avoid including sorting.hpp in .cu
// ---------------------------------------------------------------------------
static constexpr int kTileSize = 16;

// ---------------------------------------------------------------------------
// Projection kernel
// ---------------------------------------------------------------------------

/// @brief Per-Gaussian projection kernel.
///
/// Grid:  ((N + 255) / 256) blocks × 256 threads.
/// Each thread processes one Gaussian.
///
/// @param n            Number of Gaussians.
/// @param positions    [N, 3] world-space positions.
/// @param rotations    [N, 4] quaternions (w,x,y,z).
/// @param scales       [N, 3] log-space scales.
/// @param opacities    [N, 1] logit-space opacities.
/// @param view_matrix  [16] column-major 4×4 world-to-camera matrix.
/// @param fx, fy       Focal lengths in pixels.
/// @param cx, cy       Principal point in pixels.
/// @param img_w, img_h Image dimensions.
/// @param scale_mod    Global scale modifier.
/// @param means_2d     Output [N, 2] screen-space positions (pixels).
/// @param depths       Output [N] view-space depth.
/// @param cov_2d_inv   Output [N, 3] inverse 2D covariance.
/// @param radii        Output [N] pixel radii (int32).
/// @param tiles_touched Output [N] number of tiles touched (int32).
/// @param opacities_act Output [N] sigmoid-activated opacities.
__global__ void k_project_gaussians(
    int n,
    const float* __restrict__ positions,
    const float* __restrict__ rotations,
    const float* __restrict__ scales,
    const float* __restrict__ opacities,
    const float* __restrict__ view_matrix,
    float fx, float fy, float cx, float cy,
    int img_w, int img_h,
    float scale_mod,
    float* __restrict__ means_2d,
    float* __restrict__ depths,
    float* __restrict__ cov_2d_inv,
    int* __restrict__ radii,
    int* __restrict__ tiles_touched,
    float* __restrict__ opacities_act)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Default: culled
    radii[idx] = 0;
    tiles_touched[idx] = 0;

    // -----------------------------------------------------------------------
    // 1. Load Gaussian parameters
    // -----------------------------------------------------------------------
    float px = positions[idx * 3 + 0];
    float py = positions[idx * 3 + 1];
    float pz = positions[idx * 3 + 2];

    // -----------------------------------------------------------------------
    // 2. Transform to camera space
    //    view_matrix is row-major 4×4: [R | t]
    //                                  [0 | 1]
    // -----------------------------------------------------------------------
    float W[9]; // View rotation (upper-left 3×3, row-major)
    W[0] = view_matrix[0]; W[1] = view_matrix[1]; W[2] = view_matrix[2];
    W[3] = view_matrix[4]; W[4] = view_matrix[5]; W[5] = view_matrix[6];
    W[6] = view_matrix[8]; W[7] = view_matrix[9]; W[8] = view_matrix[10];

    float t_cam[3]; // Point in camera space
    t_cam[0] = W[0] * px + W[1] * py + W[2] * pz + view_matrix[3];
    t_cam[1] = W[3] * px + W[4] * py + W[5] * pz + view_matrix[7];
    t_cam[2] = W[6] * px + W[7] * py + W[8] * pz + view_matrix[11];

    // -----------------------------------------------------------------------
    // 3. Frustum cull — behind camera or too close
    // -----------------------------------------------------------------------
    if (t_cam[2] <= 0.2f) return;

    // -----------------------------------------------------------------------
    // 4. Project to screen
    // -----------------------------------------------------------------------
    float x_screen = fx * t_cam[0] / t_cam[2] + cx;
    float y_screen = fy * t_cam[1] / t_cam[2] + cy;

    depths[idx] = t_cam[2];
    means_2d[idx * 2 + 0] = x_screen;
    means_2d[idx * 2 + 1] = y_screen;

    // -----------------------------------------------------------------------
    // 5. Apply sigmoid to logit-space opacity
    // -----------------------------------------------------------------------
    float logit_o = opacities[idx];
    float sigmoid_o = 1.0f / (1.0f + expf(-logit_o));
    opacities_act[idx] = sigmoid_o;

    // -----------------------------------------------------------------------
    // 6. Compute 3D covariance
    // -----------------------------------------------------------------------
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

    float cov_3d[6];
    compute_cov_3d(log_scale, rot, cov_3d);

    // -----------------------------------------------------------------------
    // 7. Project covariance to 2D
    // -----------------------------------------------------------------------
    float cov2d[3];
    compute_cov_2d(cov_3d, W, t_cam, fx, fy, cov2d);

    // -----------------------------------------------------------------------
    // 8. Compute inverse covariance (needed for alpha evaluation)
    // -----------------------------------------------------------------------
    float cov2d_inv[3];
    float det = compute_cov_2d_inverse(cov2d, cov2d_inv);
    if (det <= 0.0f) return;

    cov_2d_inv[idx * 3 + 0] = cov2d_inv[0];
    cov_2d_inv[idx * 3 + 1] = cov2d_inv[1];
    cov_2d_inv[idx * 3 + 2] = cov2d_inv[2];

    // -----------------------------------------------------------------------
    // 9. Compute pixel radius
    // -----------------------------------------------------------------------
    int radius = compute_radius(cov2d);
    if (radius <= 0) return;

    // Cap radius to prevent a single Gaussian from covering the entire image
    int max_dim = max(img_w, img_h);
    radius = min(radius, max_dim);
    radii[idx] = radius;

    // -----------------------------------------------------------------------
    // 10. Compute tiles touched (bounding rect in tile coordinates)
    // -----------------------------------------------------------------------
    int num_tiles_x = (img_w + kTileSize - 1) / kTileSize;
    int num_tiles_y = (img_h + kTileSize - 1) / kTileSize;

    // Bounding rectangle in pixel coordinates
    int rect_min_x = max(0, static_cast<int>(x_screen - radius));
    int rect_min_y = max(0, static_cast<int>(y_screen - radius));
    int rect_max_x = min(img_w, static_cast<int>(x_screen + radius + 1));
    int rect_max_y = min(img_h, static_cast<int>(y_screen + radius + 1));

    // Convert to tile coordinates
    int tile_min_x = rect_min_x / kTileSize;
    int tile_min_y = rect_min_y / kTileSize;
    int tile_max_x = min(num_tiles_x, (rect_max_x + kTileSize - 1) / kTileSize);
    int tile_max_y = min(num_tiles_y, (rect_max_y + kTileSize - 1) / kTileSize);

    int n_tiles = (tile_max_x - tile_min_x) * (tile_max_y - tile_min_y);
    tiles_touched[idx] = max(n_tiles, 0);
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

ProjectionOutput project_gaussians(
    const torch::Tensor& positions,
    const torch::Tensor& rotations,
    const torch::Tensor& scales,
    const torch::Tensor& opacities,
    const torch::Tensor& sh_coeffs,
    const CameraInfo& camera,
    int active_sh_degree,
    float scale_modifier)
{
    TORCH_CHECK(positions.is_cuda(), "positions must be on CUDA");
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3);

    const int n = static_cast<int>(positions.size(0));
    auto device = positions.device();
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(device);

    // Allocate outputs
    auto means_2d      = torch::zeros({n, 2}, opts_f);
    auto depths        = torch::zeros({n}, opts_f);
    auto cov_2d_inv    = torch::zeros({n, 3}, opts_f);
    auto radii_t       = torch::zeros({n}, opts_i);
    auto tiles_touched = torch::zeros({n}, opts_i);
    auto opacities_act = torch::zeros({n}, opts_f);

    if (n == 0) {
        return ProjectionOutput{
            means_2d, depths, cov_2d_inv, radii_t, tiles_touched,
            torch::zeros({0, 3}, opts_f), opacities_act};
    }

    // Build row-major 4×4 view matrix from CameraInfo (Eigen is column-major)
    Eigen::Matrix4f w2c = camera.world_to_camera();
    float view_matrix[16];
    // Eigen stores column-major, so w2c(row, col) is at col*4+row
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            view_matrix[r * 4 + c] = w2c(r, c);

    // Copy view matrix to device
    auto view_matrix_t = torch::from_blob(view_matrix, {16}, torch::kFloat32)
                             .to(device);

    // Ensure inputs are contiguous float32
    auto pos_c = positions.contiguous().to(torch::kFloat32);
    auto rot_c = rotations.contiguous().to(torch::kFloat32);
    auto scl_c = scales.contiguous().to(torch::kFloat32);
    auto opa_c = opacities.contiguous().to(torch::kFloat32);

    // Launch projection kernel
    constexpr int kBlockSize = 256;
    int num_blocks = (n + kBlockSize - 1) / kBlockSize;

    k_project_gaussians<<<num_blocks, kBlockSize>>>(
        n,
        pos_c.data_ptr<float>(),
        rot_c.data_ptr<float>(),
        scl_c.data_ptr<float>(),
        opa_c.data_ptr<float>(),
        view_matrix_t.data_ptr<float>(),
        camera.intrinsics.fx, camera.intrinsics.fy,
        camera.intrinsics.cx, camera.intrinsics.cy,
        camera.width, camera.height,
        scale_modifier,
        means_2d.data_ptr<float>(),
        depths.data_ptr<float>(),
        cov_2d_inv.data_ptr<float>(),
        radii_t.data_ptr<int>(),
        tiles_touched.data_ptr<int>(),
        opacities_act.data_ptr<float>());

    CUDA_CHECK(cudaGetLastError());

    // -----------------------------------------------------------------------
    // SH evaluation for view-dependent colors
    // -----------------------------------------------------------------------
    // Direction: camera center → each Gaussian (unnormalised, then normalise)
    Eigen::Vector3f cam_center = camera.camera_center();
    auto cam_center_t = torch::tensor(
        {cam_center.x(), cam_center.y(), cam_center.z()}, opts_f);

    // directions = normalize(positions - camera_center)  [N, 3]
    auto directions = pos_c - cam_center_t.unsqueeze(0);
    auto norms = directions.norm(2, /*dim=*/1, /*keepdim=*/true).clamp_min(1e-8f);
    directions = directions / norms;

    auto rgb = evaluate_sh_cuda(active_sh_degree, sh_coeffs, directions);
    // Clamp to [0, ∞) — SH can produce negative values
    rgb = rgb.clamp_min(0.0f);

    return ProjectionOutput{
        means_2d, depths, cov_2d_inv, radii_t, tiles_touched,
        rgb, opacities_act};
}

} // namespace cugs
