#pragma once

/// @file trainer.hpp
/// @brief Main training loop for 3D Gaussian Splatting.
///
/// Wires together the dataset, Gaussian model, rasterizer, loss computation,
/// and optimizer into a complete training pipeline.

#include "core/gaussian.hpp"
#include "data/dataset.hpp"
#include "optimizer/adam.hpp"
#include "optimizer/densification.hpp"
#include "training/lr_schedule.hpp"
#include "utils/memory_monitor.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <string>

namespace cugs {

/// @brief Convert a CPU Image to a CUDA float32 tensor [H, W, 3].
///
/// Handles both RGB (3-channel) and RGBA (4-channel) images by discarding
/// the alpha channel if present.
///
/// @param image CPU image with float [0,1] pixel data.
/// @param device Target CUDA device.
/// @return Tensor [H, W, 3] on the specified device.
torch::Tensor image_to_tensor(const Image& image, torch::Device device);

/// @brief Full training configuration.
struct TrainConfig {
    // Paths
    std::filesystem::path data_path;
    std::filesystem::path output_path = "output";

    // Training
    int max_iterations  = 30000;
    int resolution_scale = 1;
    int max_sh_degree   = 3;
    float lambda_ssim   = 0.2f;

    // Checkpointing & logging
    int save_every = 7000;
    int log_every  = 100;

    // Capacity limits
    int max_gaussians = 0;  // 0 = no limit

    // Rendering
    bool random_background = false;

    // Reproducibility
    uint64_t seed = 42;

    // Optimizer
    AdamConfig adam;

    // Densification
    DensificationConfig densification;
    bool no_densify = false;  ///< Disable densification entirely

    // Memory safety
    MemoryLimitConfig memory;
};

/// @brief Statistics for a single training iteration.
struct IterationStats {
    int iteration       = 0;
    float loss          = 0.0f;
    float l1            = 0.0f;
    float ssim          = 0.0f;
    int num_gaussians   = 0;
    int active_sh_degree = 0;
    float position_lr   = 0.0f;

    // Densification stats (zero when densification didn't run).
    int num_cloned  = 0;
    int num_split   = 0;
    int num_pruned  = 0;
    bool densified  = false;
};

/// @brief Main training loop for 3D Gaussian Splatting.
///
/// Loads a dataset, initializes Gaussians from sparse points, and runs
/// the training loop: sample image -> render -> loss -> backward -> step.
class Trainer {
public:
    /// @brief Construct a trainer from configuration.
    ///
    /// Loads the dataset, initializes Gaussians from sparse points,
    /// moves model to CUDA, and creates the optimizer.
    ///
    /// @param config Training configuration.
    explicit Trainer(const TrainConfig& config);

    /// @brief Run the full training loop.
    ///
    /// Iterates max_iterations times, logging periodically and saving
    /// checkpoints at configured intervals.
    void train();

    /// @brief Run a single training iteration.
    ///
    /// @param step Current iteration number (0-based).
    /// @return Statistics for this iteration.
    IterationStats train_step(int step);

    /// @brief Save a checkpoint (PLY file) at the given step.
    void save_checkpoint(int step);

    /// @brief Log current VRAM and RAM usage via spdlog.
    void log_vram_usage();

    /// @brief Access the Gaussian model (for testing).
    const GaussianModel& model() const { return model_; }

private:
    TrainConfig config_;
    Dataset dataset_;
    GaussianModel model_;
    std::unique_ptr<GaussianAdam> optimizer_;
    std::unique_ptr<DensificationController> densify_ctrl_;
    std::mt19937 rng_;

    float effective_vram_limit_ = 0.0f;
    int vram_critical_streak_ = 0;

    /// @brief Check VRAM/RAM safety. Returns false to signal abort.
    bool check_vram_safety(int step);
};

} // namespace cugs
