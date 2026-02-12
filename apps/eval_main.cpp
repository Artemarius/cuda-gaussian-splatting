/// @file eval_main.cpp
/// @brief CLI entry point for evaluating a trained 3D Gaussian Splatting model.
///
/// Usage:
///   eval -m <model.ply> -d <dataset_path> [-o <output.json>] [-r <resolution>]
///        [--background <black|white>] [--sh-degree <0-3>]

#include "training/metrics.hpp"

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
        << "3D Gaussian Splatting Evaluator\n"
        << "\n"
        << "Usage: " << program << " -m <model.ply> -d <dataset_path> [options]\n"
        << "\n"
        << "Required:\n"
        << "  -m, --model <path>        Path to trained .ply model file\n"
        << "  -d, --data <path>         Path to COLMAP dataset directory\n"
        << "\n"
        << "Options:\n"
        << "  -o, --output <path>       Output JSON file (default: metrics.json next to model)\n"
        << "  -r, --resolution <N>      Image downscale factor (default: 1)\n"
        << "  --sh-degree <0-3>         SH degree override (default: from model)\n"
        << "  --background <color>      Background color: black or white (default: black)\n"
        << "  -h, --help                Show this help message\n";
}

/// @brief Simple arg parser. Returns true if arg matches short or long form.
bool arg_matches(const char* arg, const char* short_form, const char* long_form) {
    return (short_form && std::strcmp(arg, short_form) == 0) ||
           (long_form && std::strcmp(arg, long_form) == 0);
}

} // namespace

int main(int argc, char* argv[]) {
    std::filesystem::path model_path;
    std::filesystem::path data_path;
    std::filesystem::path output_path;
    int resolution_scale = 1;
    int sh_degree = -1;  // -1 = auto from model
    bool white_background = false;
    bool has_model = false;
    bool has_data = false;

    for (int i = 1; i < argc; ++i) {
        if (arg_matches(argv[i], "-h", "--help")) {
            print_usage(argv[0]);
            return 0;
        }
        if (arg_matches(argv[i], "-m", "--model") && i + 1 < argc) {
            model_path = argv[++i];
            has_model = true;
        } else if (arg_matches(argv[i], "-d", "--data") && i + 1 < argc) {
            data_path = argv[++i];
            has_data = true;
        } else if (arg_matches(argv[i], "-o", "--output") && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg_matches(argv[i], "-r", "--resolution") && i + 1 < argc) {
            resolution_scale = std::atoi(argv[++i]);
        } else if (arg_matches(argv[i], nullptr, "--sh-degree") && i + 1 < argc) {
            sh_degree = std::atoi(argv[++i]);
        } else if (arg_matches(argv[i], nullptr, "--background") && i + 1 < argc) {
            ++i;
            if (std::strcmp(argv[i], "white") == 0) {
                white_background = true;
            } else if (std::strcmp(argv[i], "black") != 0) {
                spdlog::error("Unknown background color: {} (use 'black' or 'white')", argv[i]);
                return 1;
            }
        } else {
            spdlog::error("Unknown argument: {}", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!has_model) {
        spdlog::error("Model path is required (-m / --model)");
        print_usage(argv[0]);
        return 1;
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
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        spdlog::info("CUDA device: {}", prop.name);
    }

    try {
        // Load model.
        spdlog::info("Loading model: {}", model_path.string());
        auto model = cugs::GaussianModel::load_ply(model_path);
        model.to_device(torch::kCUDA);
        spdlog::info("Model: {} Gaussians, SH degree {}",
                     model.num_gaussians(), model.max_sh_degree());

        // Determine SH degree.
        int active_sh = (sh_degree >= 0) ? sh_degree : model.max_sh_degree();
        if (active_sh > model.max_sh_degree()) {
            spdlog::warn("Requested SH degree {} exceeds model's max degree {}; clamping",
                         active_sh, model.max_sh_degree());
            active_sh = model.max_sh_degree();
        }

        // Load dataset.
        spdlog::info("Loading dataset: {}", data_path.string());
        cugs::Dataset dataset(data_path, resolution_scale);
        dataset.print_summary();

        if (dataset.num_test() == 0) {
            spdlog::error("Dataset has no test images — nothing to evaluate");
            return 1;
        }

        // Configure rendering.
        cugs::RenderSettings settings;
        settings.active_sh_degree = active_sh;
        if (white_background) {
            settings.background[0] = 1.0f;
            settings.background[1] = 1.0f;
            settings.background[2] = 1.0f;
        }

        // Evaluate.
        auto results = cugs::evaluate(model, dataset, settings);

        // Print summary table.
        std::cout << "\n";
        std::cout << "=== Evaluation Results ===\n";
        std::cout << "  Gaussians:   " << results.num_gaussians << "\n";
        std::cout << "  SH degree:   " << results.sh_degree << "\n";
        std::cout << "  Test images: " << results.per_image.size() << "\n";
        std::cout << "  Mean PSNR:   " << results.mean_psnr << " dB\n";
        std::cout << "  Mean SSIM:   " << results.mean_ssim << "\n";
        std::cout << "  Eval time:   " << results.eval_time_seconds << " s\n";
        std::cout << "\n";

        // Determine output path.
        if (output_path.empty()) {
            output_path = model_path.parent_path() / "metrics.json";
        }

        // Save JSON.
        results.save_json(output_path);

    } catch (const std::exception& e) {
        spdlog::error("Evaluation failed: {}", e.what());
        return 1;
    }

    return 0;
}
