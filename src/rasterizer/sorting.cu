/// @file sorting.cu
/// @brief Tile-based Gaussian sorting using CUB radix sort.
///
/// After projection, each Gaussian covers a set of screen tiles. This module:
///   1. Computes a prefix sum over tiles_touched to get per-Gaussian write offsets
///   2. Writes (key, value) pairs: key = (tile_id << 32 | depth_bits), value = gaussian_idx
///   3. Sorts pairs by key using cub::DeviceRadixSort
///   4. Finds per-tile start/end ranges in the sorted array
///
/// Key encoding: uint64_t key = ((uint64_t)tile_id << 32) | __float_as_uint(depth)
/// Positive depth floats have IEEE754 bit patterns that sort correctly as unsigned ints.

#include "rasterizer/sorting.hpp"
#include "utils/cuda_utils.cuh"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <torch/torch.h>

namespace cugs {

// ---------------------------------------------------------------------------
// Kernel: fill sort key/value pairs
// ---------------------------------------------------------------------------

/// @brief Write (tile_id << 32 | depth_bits, gaussian_idx) pairs.
///
/// Grid: ((N + 255) / 256) blocks × 256 threads. One thread per Gaussian.
/// Each Gaussian writes tiles_touched[idx] pairs starting at offsets[idx].
__global__ void k_fill_sort_pairs(
    int n,
    const float* __restrict__ means_2d,     // [N, 2]
    const float* __restrict__ depths,        // [N]
    const int* __restrict__ radii,           // [N]
    const int* __restrict__ offsets,          // [N] — exclusive prefix sum of tiles_touched
    int img_w, int img_h,
    int num_tiles_x, int num_tiles_y,
    unsigned long long* __restrict__ keys_out,  // [P]
    int* __restrict__ values_out)               // [P]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int radius = radii[idx];
    if (radius <= 0) return;

    float x = means_2d[idx * 2 + 0];
    float y = means_2d[idx * 2 + 1];
    float depth = depths[idx];

    // Bounding rect in pixel coords → tile coords
    int rect_min_x = max(0, static_cast<int>(x - radius)) / kTileSize;
    int rect_min_y = max(0, static_cast<int>(y - radius)) / kTileSize;
    int rect_max_x = min(num_tiles_x,
        (min(img_w, static_cast<int>(x + radius + 1)) + kTileSize - 1) / kTileSize);
    int rect_max_y = min(num_tiles_y,
        (min(img_h, static_cast<int>(y + radius + 1)) + kTileSize - 1) / kTileSize);

    // Depth bits — IEEE754 float bits sort correctly for positive values
    unsigned int depth_bits = __float_as_uint(depth);

    int write_pos = offsets[idx];
    for (int ty = rect_min_y; ty < rect_max_y; ++ty) {
        for (int tx = rect_min_x; tx < rect_max_x; ++tx) {
            unsigned long long tile_id = static_cast<unsigned long long>(ty * num_tiles_x + tx);
            unsigned long long key = (tile_id << 32) | static_cast<unsigned long long>(depth_bits);
            keys_out[write_pos] = key;
            values_out[write_pos] = idx;
            write_pos++;
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: compute per-tile ranges in the sorted array
// ---------------------------------------------------------------------------

/// @brief Detect tile boundaries in sorted keys → tile_ranges[tile_id] = {start, end}.
///
/// Grid: ((P + 255) / 256) blocks × 256 threads. One thread per sorted pair.
/// Compares current tile_id with previous pair's tile_id to detect boundaries.
__global__ void k_compute_tile_ranges(
    int total_pairs,
    const unsigned long long* __restrict__ sorted_keys, // [P]
    int* __restrict__ tile_ranges)                       // [num_tiles, 2]
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    unsigned int cur_tile = static_cast<unsigned int>(sorted_keys[idx] >> 32);

    if (idx == 0) {
        // First pair: start of its tile
        tile_ranges[cur_tile * 2 + 0] = 0;
    } else {
        unsigned int prev_tile = static_cast<unsigned int>(sorted_keys[idx - 1] >> 32);
        if (cur_tile != prev_tile) {
            // End of previous tile
            tile_ranges[prev_tile * 2 + 1] = idx;
            // Start of current tile
            tile_ranges[cur_tile * 2 + 0] = idx;
        }
    }

    // Last pair: end of its tile
    if (idx == total_pairs - 1) {
        tile_ranges[cur_tile * 2 + 1] = total_pairs;
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------

SortingOutput sort_gaussians(
    const torch::Tensor& means_2d,
    const torch::Tensor& depths,
    const torch::Tensor& radii,
    const torch::Tensor& tiles_touched,
    int img_w, int img_h)
{
    TORCH_CHECK(means_2d.is_cuda(), "means_2d must be on CUDA");

    const int n = static_cast<int>(means_2d.size(0));
    auto device = means_2d.device();
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(device);

    int num_tiles_x = (img_w + kTileSize - 1) / kTileSize;
    int num_tiles_y = (img_h + kTileSize - 1) / kTileSize;
    int num_tiles = num_tiles_x * num_tiles_y;

    // Handle empty case
    if (n == 0) {
        return SortingOutput{
            torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(device)),
            torch::empty({0}, opts_i),
            torch::zeros({num_tiles, 2}, opts_i),
            0};
    }

    // -----------------------------------------------------------------------
    // 1. Prefix sum on tiles_touched → per-Gaussian offsets + total pairs
    // -----------------------------------------------------------------------
    // Cumulative sum gives inclusive prefix sum; shift right for exclusive
    auto cumsum = tiles_touched.to(torch::kInt32).cumsum(0, torch::kInt32);
    int total_pairs = cumsum[-1].item<int>();

    // Exclusive prefix sum: [0, cumsum[0], cumsum[1], ..., cumsum[N-2]]
    auto offsets = torch::zeros({n}, opts_i);
    if (n > 1) {
        offsets.slice(0, 1, n) = cumsum.slice(0, 0, n - 1);
    }

    if (total_pairs == 0) {
        return SortingOutput{
            torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(device)),
            torch::empty({0}, opts_i),
            torch::zeros({num_tiles, 2}, opts_i),
            0};
    }

    // -----------------------------------------------------------------------
    // 2. Fill sort pairs
    // -----------------------------------------------------------------------
    auto opts_ull = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto keys_unsorted = torch::zeros({total_pairs}, opts_ull);
    auto values_unsorted = torch::zeros({total_pairs}, opts_i);

    constexpr int kBlockSize = 256;
    int num_blocks = (n + kBlockSize - 1) / kBlockSize;

    k_fill_sort_pairs<<<num_blocks, kBlockSize>>>(
        n,
        means_2d.contiguous().data_ptr<float>(),
        depths.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int>(),
        offsets.data_ptr<int>(),
        img_w, img_h,
        num_tiles_x, num_tiles_y,
        reinterpret_cast<unsigned long long*>(keys_unsorted.data_ptr<int64_t>()),
        values_unsorted.data_ptr<int>());
    CUDA_CHECK(cudaGetLastError());

    // -----------------------------------------------------------------------
    // 3. CUB radix sort
    // -----------------------------------------------------------------------
    auto keys_sorted = torch::empty_like(keys_unsorted);
    auto values_sorted = torch::empty_like(values_unsorted);

    // Determine temporary storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        reinterpret_cast<unsigned long long*>(keys_unsorted.data_ptr<int64_t>()),
        reinterpret_cast<unsigned long long*>(keys_sorted.data_ptr<int64_t>()),
        reinterpret_cast<unsigned int*>(values_unsorted.data_ptr<int>()),
        reinterpret_cast<unsigned int*>(values_sorted.data_ptr<int>()),
        total_pairs);

    auto temp_storage = torch::empty(
        {static_cast<int64_t>(temp_storage_bytes)},
        torch::TensorOptions().dtype(torch::kUInt8).device(device));

    cub::DeviceRadixSort::SortPairs(
        temp_storage.data_ptr<uint8_t>(), temp_storage_bytes,
        reinterpret_cast<unsigned long long*>(keys_unsorted.data_ptr<int64_t>()),
        reinterpret_cast<unsigned long long*>(keys_sorted.data_ptr<int64_t>()),
        reinterpret_cast<unsigned int*>(values_unsorted.data_ptr<int>()),
        reinterpret_cast<unsigned int*>(values_sorted.data_ptr<int>()),
        total_pairs);
    CUDA_CHECK(cudaGetLastError());

    // -----------------------------------------------------------------------
    // 4. Compute tile ranges
    // -----------------------------------------------------------------------
    auto tile_ranges = torch::zeros({num_tiles, 2}, opts_i);

    int num_blocks_ranges = (total_pairs + kBlockSize - 1) / kBlockSize;
    k_compute_tile_ranges<<<num_blocks_ranges, kBlockSize>>>(
        total_pairs,
        reinterpret_cast<unsigned long long*>(keys_sorted.data_ptr<int64_t>()),
        tile_ranges.data_ptr<int>());
    CUDA_CHECK(cudaGetLastError());

    return SortingOutput{
        keys_sorted, values_sorted, tile_ranges, total_pairs};
}

} // namespace cugs
