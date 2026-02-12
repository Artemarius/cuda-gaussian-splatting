/// @file rasterizer.cpp
/// @brief Host orchestration for the forward Gaussian splatting pipeline.
///
/// Coordinates the three stages:
///   1. Projection — 3D Gaussians → 2D screen space (includes SH evaluation)
///   2. Sorting    — tile-based radix sort by (tile_id, depth)
///   3. Rasterization — per-tile alpha-compositing to produce final image
///
/// This file includes only .hpp headers (no .cuh), so it compiles with MSVC.

#include "rasterizer/rasterizer.hpp"
#include "rasterizer/projection.hpp"
#include "rasterizer/sorting.hpp"
#include "rasterizer/forward.hpp"
#include "rasterizer/backward.hpp"
#include "rasterizer/projection_backward.hpp"

#include <spdlog/spdlog.h>

namespace cugs {

RenderOutput render(
    const GaussianModel& model,
    const CameraInfo& camera,
    const RenderSettings& settings)
{
    TORCH_CHECK(model.is_valid(), "GaussianModel is not valid");
    TORCH_CHECK(model.positions.is_cuda(), "GaussianModel must be on CUDA device");

    const int64_t n = model.num_gaussians();
    auto device = model.positions.device();
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(device);

    // Handle empty model
    if (n == 0) {
        auto color = torch::zeros({camera.height, camera.width, 3}, opts_f);
        color.select(2, 0).fill_(settings.background[0]);
        color.select(2, 1).fill_(settings.background[1]);
        color.select(2, 2).fill_(settings.background[2]);

        return RenderOutput{
            /*color=*/          color,
            /*final_T=*/        torch::ones({camera.height, camera.width}, opts_f),
            /*n_contrib=*/      torch::zeros({camera.height, camera.width}, opts_i),
            /*means_2d=*/       torch::empty({0, 2}, opts_f),
            /*depths=*/         torch::empty({0}, opts_f),
            /*cov_2d_inv=*/     torch::empty({0, 3}, opts_f),
            /*radii=*/          torch::empty({0}, opts_i),
            /*rgb=*/            torch::empty({0, 3}, opts_f),
            /*opacities_act=*/  torch::empty({0}, opts_f),
            /*gaussian_indices=*/ torch::empty({0}, opts_i),
            /*tile_ranges=*/    torch::empty({0, 2}, opts_i),
        };
    }

    // -----------------------------------------------------------------------
    // Stage 1: Projection (includes SH evaluation)
    // -----------------------------------------------------------------------
    int active_degree = std::min(settings.active_sh_degree, model.max_sh_degree());

    auto proj = project_gaussians(
        model.positions,
        model.rotations,
        model.scales,
        model.opacities,
        model.sh_coeffs,
        camera,
        active_degree,
        settings.scale_modifier);

    // -----------------------------------------------------------------------
    // Stage 2: Sorting
    // -----------------------------------------------------------------------
    auto sort = sort_gaussians(
        proj.means_2d,
        proj.depths,
        proj.radii,
        proj.tiles_touched,
        camera.width,
        camera.height);

    // -----------------------------------------------------------------------
    // Stage 3: Rasterization
    // -----------------------------------------------------------------------
    auto fwd = rasterize_forward(
        proj.means_2d,
        proj.cov_2d_inv,
        proj.rgb,
        proj.opacities_act,
        sort.tile_ranges,
        sort.gaussian_values_sorted,
        camera.width,
        camera.height,
        settings.background);

    // -----------------------------------------------------------------------
    // Pack output
    // -----------------------------------------------------------------------
    return RenderOutput{
        /*color=*/             fwd.color,
        /*final_T=*/           fwd.final_T,
        /*n_contrib=*/         fwd.n_contrib,
        /*means_2d=*/          proj.means_2d,
        /*depths=*/            proj.depths,
        /*cov_2d_inv=*/        proj.cov_2d_inv,
        /*radii=*/             proj.radii,
        /*rgb=*/               proj.rgb,
        /*opacities_act=*/     proj.opacities_act,
        /*gaussian_indices=*/  sort.gaussian_values_sorted,
        /*tile_ranges=*/       sort.tile_ranges,
    };
}

BackwardOutput render_backward(
    const torch::Tensor& dL_dcolor,
    const RenderOutput& render_out,
    const GaussianModel& model,
    const CameraInfo& camera,
    const RenderSettings& settings)
{
    TORCH_CHECK(dL_dcolor.is_cuda(), "dL_dcolor must be on CUDA device");
    TORCH_CHECK(dL_dcolor.dim() == 3 && dL_dcolor.size(2) == 3,
                "dL_dcolor must be [H, W, 3]");

    const int n = static_cast<int>(model.num_gaussians());
    auto device = dL_dcolor.device();
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    if (n == 0) {
        return BackwardOutput{
            torch::zeros({0, 3}, opts_f),
            torch::zeros({0, 4}, opts_f),
            torch::zeros({0, 3}, opts_f),
            torch::zeros({0, 1}, opts_f),
            torch::zeros_like(model.sh_coeffs),
        };
    }

    int active_degree = std::min(settings.active_sh_degree, model.max_sh_degree());

    // -----------------------------------------------------------------------
    // Stage 1: Rasterize backward (pixel gradients → per-Gaussian 2D grads)
    // -----------------------------------------------------------------------
    auto rast_bwd = rasterize_backward(
        dL_dcolor,
        render_out.means_2d,
        render_out.cov_2d_inv,
        render_out.rgb,
        render_out.opacities_act,
        render_out.tile_ranges,
        render_out.gaussian_indices,
        render_out.final_T,
        render_out.n_contrib,
        camera.width, camera.height,
        settings.background,
        n);

    // -----------------------------------------------------------------------
    // Stage 2: Projection backward (2D grads → 3D parameter grads)
    // -----------------------------------------------------------------------
    auto proj_bwd = project_backward(
        rast_bwd.dL_dmeans_2d,
        rast_bwd.dL_dcov_2d_inv,
        rast_bwd.dL_drgb,
        rast_bwd.dL_dopacity_act,
        model.positions,
        model.rotations,
        model.scales,
        model.opacities,
        model.sh_coeffs,
        render_out.radii,
        camera,
        active_degree,
        settings.scale_modifier);

    return BackwardOutput{
        proj_bwd.dL_dpositions,
        proj_bwd.dL_drotations,
        proj_bwd.dL_dscales,
        proj_bwd.dL_dopacities,
        proj_bwd.dL_dsh_coeffs,
    };
}

} // namespace cugs
