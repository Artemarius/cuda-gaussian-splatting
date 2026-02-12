/// @file train_main.cpp
/// @brief CLI entry point for 3D Gaussian Splatting training.
///
/// Usage:
///   train -d <dataset_path> [-o <output>] [-i <iterations>] [-r <resolution>]
///         [--sh-degree <0-3>] [--max-gaussians <N>] [--save-every <N>]
///         [--log-every <N>] [--lambda <0-1>] [--random-bg] [--seed <N>]

#include "training/trainer.hpp"

#include <spdlog/spdlog.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* program) {
    std::cout
        << "3D Gaussian Splatting Trainer\n"
        << "\n"
        << "Usage: " << program << " -d <dataset_path> [options]\n"
        << "\n"
        << "Required:\n"
        << "  -d, --data <path>         Path to COLMAP dataset directory\n"
        << "\n"
        << "Options:\n"
        << "  -o, --output <path>       Output directory (default: output)\n"
        << "  -i, --iterations <N>      Number of training iterations (default: 30000)\n"
        << "  -r, --resolution <N>      Image downscale factor (default: 1)\n"
        << "  --sh-degree <0-3>         Maximum SH degree (default: 3)\n"
        << "  --max-gaussians <N>       Cap on number of Gaussians (default: no limit)\n"
        << "  --save-every <N>          Checkpoint interval (default: 7000)\n"
        << "  --log-every <N>           Logging interval (default: 100)\n"
        << "  --lambda <0-1>            SSIM loss weight (default: 0.2)\n"
        << "  --random-bg               Use random background color each iteration\n"
        << "  --seed <N>                Random seed (default: 42)\n"
        << "\n"
        << "Densification:\n"
        << "  --densify-from <N>        Start densification at step N (default: 500)\n"
        << "  --densify-until <N>       Stop densification at step N (default: 15000)\n"
        << "  --densify-every <N>       Densify every N steps (default: 100)\n"
        << "  --grad-threshold <F>      Position gradient threshold (default: 0.0002)\n"
        << "  --no-densify              Disable densification entirely\n"
        << "\n"
        << "Memory:\n"
        << "  --vram-limit <MB>         Hard VRAM usage limit in MB (default: auto)\n"
        << "  -h, --help                Show this help message\n";
}

/// @brief Simple arg parser. Returns true if arg matches short or long form.
bool arg_matches(const char* arg, const char* short_form, const char* long_form) {
    return (short_form && std::strcmp(arg, short_form) == 0) ||
           (long_form && std::strcmp(arg, long_form) == 0);
}

} // namespace

int main(int argc, char* argv[]) {
    cugs::TrainConfig config;
    bool has_data = false;

    for (int i = 1; i < argc; ++i) {
        if (arg_matches(argv[i], "-h", "--help")) {
            print_usage(argv[0]);
            return 0;
        }
        if (arg_matches(argv[i], "-d", "--data") && i + 1 < argc) {
            config.data_path = argv[++i];
            has_data = true;
        } else if (arg_matches(argv[i], "-o", "--output") && i + 1 < argc) {
            config.output_path = argv[++i];
        } else if (arg_matches(argv[i], "-i", "--iterations") && i + 1 < argc) {
            config.max_iterations = std::atoi(argv[++i]);
        } else if (arg_matches(argv[i], "-r", "--resolution") && i + 1 < argc) {
            config.resolution_scale = std::atoi(argv[++i]);
        } else if (arg_matches(argv[i], nullptr, "--sh-degree") && i + 1 < argc) {
            config.max_sh_degree = std::atoi(argv[++i]);
        } else if (arg_matches(argv[i], nullptr, "--max-gaussians") && i + 1 < argc) {
            config.max_gaussians = std::atoi(argv[++i]);
        } else if (arg_matches(argv[i], nullptr, "--save-every") && i + 1 < argc) {
            config.save_every = std::atoi(argv[++i]);
        } else if (arg_matches(argv[i], nullptr, "--log-every") && i + 1 < argc) {
            config.log_every = std::atoi(argv[++i]);
        } else if (arg_matches(argv[i], nullptr, "--lambda") && i + 1 < argc) {
            config.lambda_ssim = static_cast<float>(std::atof(argv[++i]));
        } else if (arg_matches(argv[i], nullptr, "--random-bg")) {
            config.random_background = true;
        } else if (arg_matches(argv[i], nullptr, "--seed") && i + 1 < argc) {
            config.seed = static_cast<uint64_t>(std::atoll(argv[++i]));
        } else if (arg_matches(argv[i], nullptr, "--densify-from") && i + 1 < argc) {
            config.densification.densify_from = std::atoi(argv[++i]);
        } else if (arg_matches(argv[i], nullptr, "--densify-until") && i + 1 < argc) {
            config.densification.densify_until = std::atoi(argv[++i]);
        } else if (arg_matches(argv[i], nullptr, "--densify-every") && i + 1 < argc) {
            config.densification.densify_every = std::atoi(argv[++i]);
        } else if (arg_matches(argv[i], nullptr, "--grad-threshold") && i + 1 < argc) {
            config.densification.grad_threshold = static_cast<float>(std::atof(argv[++i]));
        } else if (arg_matches(argv[i], nullptr, "--no-densify")) {
            config.no_densify = true;
        } else if (arg_matches(argv[i], nullptr, "--vram-limit") && i + 1 < argc) {
            config.memory.vram_limit_mb = static_cast<float>(std::atof(argv[++i]));
        } else {
            spdlog::error("Unknown argument: {}", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!has_data) {
        spdlog::error("Dataset path is required (-d / --data)");
        print_usage(argv[0]);
        return 1;
    }

    // Validate CUDA.
    if (!torch::cuda::is_available()) {
        spdlog::error("CUDA is not available. This program requires a CUDA-capable GPU.");
        return 1;
    }
    { cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0); spdlog::info("CUDA device: {}", prop.name); }

    // Validate config.
    if (config.max_iterations <= 0) {
        spdlog::error("Iterations must be positive");
        return 1;
    }
    if (config.max_sh_degree < 0 || config.max_sh_degree > 3) {
        spdlog::error("SH degree must be 0..3");
        return 1;
    }

    try {
        cugs::Trainer trainer(config);
        trainer.train();
    } catch (const std::exception& e) {
        spdlog::error("Training failed: {}", e.what());
        return 1;
    }

    return 0;
}
