#include "core/gaussian_init.hpp"
#include "core/sh.hpp"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace cugs {

namespace {

/// @brief Compute the mean distance to the k nearest neighbors for each point.
///
/// Uses brute-force O(N^2) pairwise distances. Acceptable for typical COLMAP
/// sparse point counts (~10K-100K points). For larger counts, a spatial
/// acceleration structure would be needed.
///
/// @param positions Float pointer to [N, 3] position data (row-major).
/// @param n Number of points.
/// @param k Number of nearest neighbors.
/// @return Vector of mean k-NN distances, one per point.
std::vector<float> compute_knn_mean_distances(const float* positions,
                                               int64_t n, int k) {
    std::vector<float> mean_dists(n, 0.0f);

    if (n <= 1) {
        // Can't compute neighbors with 0 or 1 point — use a default scale
        std::fill(mean_dists.begin(), mean_dists.end(), 1.0f);
        return mean_dists;
    }

    // Clamp k to at most n-1
    k = std::min(k, static_cast<int>(n - 1));

    // For each point, find k nearest neighbors via brute force
    std::vector<float> dists(n);

    for (int64_t i = 0; i < n; ++i) {
        const float px = positions[i * 3 + 0];
        const float py = positions[i * 3 + 1];
        const float pz = positions[i * 3 + 2];

        // Compute squared distances to all other points
        for (int64_t j = 0; j < n; ++j) {
            const float dx = positions[j * 3 + 0] - px;
            const float dy = positions[j * 3 + 1] - py;
            const float dz = positions[j * 3 + 2] - pz;
            dists[j] = dx * dx + dy * dy + dz * dz;
        }
        // Set self-distance to infinity so it's not selected
        dists[i] = std::numeric_limits<float>::max();

        // Partial sort to get k smallest
        std::nth_element(dists.begin(), dists.begin() + k, dists.end());

        // Mean of k smallest squared distances, then sqrt
        float sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            sum += std::sqrt(dists[j]);
        }
        mean_dists[i] = sum / static_cast<float>(k);
    }

    return mean_dists;
}

} // anonymous namespace

GaussianModel init_gaussians_from_sparse(
    std::span<const SparsePoint> points,
    int sh_degree,
    int k_neighbors) {

    TORCH_CHECK(sh_degree >= 0 && sh_degree <= kMaxSHDegree,
                "SH degree must be 0..", kMaxSHDegree, ", got ", sh_degree);

    const int64_t n = static_cast<int64_t>(points.size());
    const int num_coeffs = sh_coeff_count(sh_degree);

    spdlog::info("Initializing {} Gaussians from sparse points (SH degree {}, {} coeffs/channel)",
                 n, sh_degree, num_coeffs);

    GaussianModel model;

    if (n == 0) {
        model.positions  = torch::zeros({0, 3}, torch::kFloat32);
        model.sh_coeffs  = torch::zeros({0, 3, num_coeffs}, torch::kFloat32);
        model.opacities  = torch::zeros({0, 1}, torch::kFloat32);
        model.rotations  = torch::zeros({0, 4}, torch::kFloat32);
        model.scales     = torch::zeros({0, 3}, torch::kFloat32);
        return model;
    }

    // -- Positions: directly from sparse points --
    model.positions = torch::zeros({n, 3}, torch::kFloat32);
    auto pos_acc = model.positions.accessor<float, 2>();
    for (int64_t i = 0; i < n; ++i) {
        pos_acc[i][0] = points[i].position.x();
        pos_acc[i][1] = points[i].position.y();
        pos_acc[i][2] = points[i].position.z();
    }

    // -- SH coefficients: DC term from point color, higher bands zero --
    // The DC SH basis value is C0 = 0.28209479... (kSH_C0).
    // To convert RGB [0,1] to SH coefficient space:
    //   sh_dc = (color_float - 0.5) / C0
    // This ensures evaluate_sh with degree 0 returns: C0 * sh_dc + 0.5 = color_float
    model.sh_coeffs = torch::zeros({n, 3, num_coeffs}, torch::kFloat32);
    auto sh_acc = model.sh_coeffs.accessor<float, 3>();
    for (int64_t i = 0; i < n; ++i) {
        for (int ch = 0; ch < 3; ++ch) {
            float color_float = static_cast<float>(points[i].color[ch]) / 255.0f;
            sh_acc[i][ch][0] = (color_float - 0.5f) / kSH_C0;
        }
        // Higher-order coefficients remain zero
    }

    // -- Opacity: inverse_sigmoid(0.1) --
    // sigmoid(x) = 1 / (1 + exp(-x))
    // inverse_sigmoid(p) = log(p / (1 - p))
    // inverse_sigmoid(0.1) = log(0.1 / 0.9) ~ -2.1972
    constexpr float init_opacity_logit = -2.1972245773362196f;
    model.opacities = torch::full({n, 1}, init_opacity_logit, torch::kFloat32);

    // -- Rotation: identity quaternion [w=1, x=0, y=0, z=0] --
    model.rotations = torch::zeros({n, 4}, torch::kFloat32);
    model.rotations.index({torch::indexing::Slice(), 0}) = 1.0f;  // w = 1

    // -- Scale: log of mean distance to k nearest neighbors --
    auto knn_dists = compute_knn_mean_distances(
        model.positions.data_ptr<float>(), n, k_neighbors);

    model.scales = torch::zeros({n, 3}, torch::kFloat32);
    auto scale_acc = model.scales.accessor<float, 2>();
    for (int64_t i = 0; i < n; ++i) {
        // Isotropic initial scale; clamp distance to avoid log(0)
        float log_dist = std::log(std::max(knn_dists[i], 1e-7f));
        scale_acc[i][0] = log_dist;
        scale_acc[i][1] = log_dist;
        scale_acc[i][2] = log_dist;
    }

    spdlog::info("Gaussian init complete: {} points, scale range [{:.4f}, {:.4f}]",
                 n,
                 model.scales.min().item<float>(),
                 model.scales.max().item<float>());

    return model;
}

} // namespace cugs
