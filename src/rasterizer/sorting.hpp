#pragma once

/// @file sorting.hpp
/// @brief Host-side declarations for tile-based Gaussian sorting.
///
/// After projection, each Gaussian may overlap multiple tiles. The sorting
/// stage creates (tile_id, depth) pairs for every (Gaussian, tile) combination,
/// sorts them by tile then by depth using CUB radix sort, and computes the
/// start/end range of sorted Gaussians per tile.

#include <torch/torch.h>

namespace cugs {

/// @brief Tile size in pixels. Each tile is kTileSize × kTileSize pixels.
constexpr int kTileSize = 16;

/// @brief Output of the tile-based sorting stage.
struct SortingOutput {
    torch::Tensor gaussian_keys_sorted;  ///< Sorted uint64 keys [P] (tile_id << 32 | depth_bits)
    torch::Tensor gaussian_values_sorted; ///< Sorted Gaussian indices [P] (int32)
    torch::Tensor tile_ranges;           ///< Per-tile [start, end) indices [num_tiles, 2] (int32)
    int total_pairs;                      ///< Total number of (tile, Gaussian) pairs
};

/// @brief Sort projected Gaussians into per-tile depth order.
///
/// Pipeline:
///   1. Exclusive prefix sum on tiles_touched → per-Gaussian offsets + total P
///   2. Fill (tile_id << 32 | depth_bits, gaussian_idx) pairs
///   3. CUB radix sort on uint64 keys
///   4. Detect tile boundaries → tile_ranges
///
/// @param means_2d       2D positions [N, 2], float32, CUDA.
/// @param depths         View-space depths [N], float32, CUDA.
/// @param radii          Pixel radii [N], int32, CUDA.
/// @param tiles_touched  Number of tiles per Gaussian [N], int32, CUDA.
/// @param img_w          Image width in pixels.
/// @param img_h          Image height in pixels.
/// @return Sorting results including sorted pairs and tile ranges.
SortingOutput sort_gaussians(
    const torch::Tensor& means_2d,
    const torch::Tensor& depths,
    const torch::Tensor& radii,
    const torch::Tensor& tiles_touched,
    int img_w, int img_h);

} // namespace cugs
