/// @file viewer_main.cpp
/// @brief CLI entry point for the real-time Gaussian splatting viewer.
///
/// Usage:
///   viewer <model.ply> [options]
///   viewer output/garden/point_cloud.ply --width 1920 --height 1080
///   viewer model.ply --background white --sh-degree 2

#include "viewer/viewer.hpp"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <torch/torch.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>

namespace {

void print_usage(const char* program) {
    std::cout
        << "3D Gaussian Splatting Real-Time Viewer\n"
        << "\n"
        << "Usage: " << program << " <model.ply> [options]\n"
        << "\n"
        << "Required:\n"
        << "  <model.ply>               Path to trained Gaussian model PLY file\n"
        << "\n"
        << "Options:\n"
        << "  --width <N>               Window width in pixels (default: 1280)\n"
        << "  --height <N>              Window height in pixels (default: 720)\n"
        << "  --background <color>      Background color: black or white (default: black)\n"
        << "  --sh-degree <0-3>         Active SH degree (default: model's max)\n"
        << "  --no-vsync                Disable vertical sync\n"
        << "  -h, --help                Show this help message\n"
        << "\n"
        << "Controls:\n"
        << "  Left drag                 Orbit camera\n"
        << "  Middle/Right drag         Pan camera\n"
        << "  Scroll wheel              Zoom in/out\n"
        << "  1/2/3                     RGB / Depth / Heatmap mode\n"
        << "  Esc                       Quit\n";
}

bool arg_matches(const char* arg, const char* short_form, const char* long_form) {
    return (short_form && std::strcmp(arg, short_form) == 0) ||
           (long_form && std::strcmp(arg, long_form) == 0);
}

} // namespace

int main(int argc, char* argv[]) {
    cugs::ViewerConfig config;
    std::filesystem::path ply_path;
    bool has_ply = false;

    for (int i = 1; i < argc; ++i) {
        if (arg_matches(argv[i], "-h", "--help")) {
            print_usage(argv[0]);
            return 0;
        }
        if (arg_matches(argv[i], nullptr, "--width") && i + 1 < argc) {
            config.width = std::atoi(argv[++i]);
        } else if (arg_matches(argv[i], nullptr, "--height") && i + 1 < argc) {
            config.height = std::atoi(argv[++i]);
        } else if (arg_matches(argv[i], nullptr, "--background") && i + 1 < argc) {
            ++i;
            if (std::strcmp(argv[i], "white") == 0) {
                config.background[0] = config.background[1] = config.background[2] = 1.0f;
            } else if (std::strcmp(argv[i], "black") == 0) {
                config.background[0] = config.background[1] = config.background[2] = 0.0f;
            } else {
                spdlog::error("Unknown background color '{}' — use 'black' or 'white'", argv[i]);
                return 1;
            }
        } else if (arg_matches(argv[i], nullptr, "--sh-degree") && i + 1 < argc) {
            config.sh_degree = std::atoi(argv[++i]);
        } else if (arg_matches(argv[i], nullptr, "--no-vsync")) {
            config.vsync = false;
        } else if (argv[i][0] == '-') {
            spdlog::error("Unknown option: {}", argv[i]);
            print_usage(argv[0]);
            return 1;
        } else {
            // Positional argument = PLY path
            ply_path = argv[i];
            has_ply = true;
        }
    }

    if (!has_ply) {
        spdlog::error("PLY model path is required");
        print_usage(argv[0]);
        return 1;
    }

    if (!std::filesystem::exists(ply_path)) {
        spdlog::error("PLY file not found: {}", ply_path.string());
        return 1;
    }

    // Validate CUDA
    if (!torch::cuda::is_available()) {
        spdlog::error("CUDA is not available. A CUDA-capable GPU is required.");
        return 1;
    }
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        spdlog::info("CUDA device: {}", prop.name);
    }

    // Validate config
    if (config.width <= 0 || config.height <= 0) {
        spdlog::error("Window dimensions must be positive");
        return 1;
    }
    if (config.sh_degree >= 0 && (config.sh_degree < 0 || config.sh_degree > 3)) {
        spdlog::error("SH degree must be 0..3");
        return 1;
    }

    try {
        cugs::Viewer viewer(ply_path, config);
        viewer.run();
    } catch (const std::exception& e) {
        spdlog::error("Viewer failed: {}", e.what());
        return 1;
    }

    return 0;
}
