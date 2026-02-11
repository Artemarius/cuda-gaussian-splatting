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

} // namespace cugs
