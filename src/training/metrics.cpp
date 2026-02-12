/// @file metrics.cpp
/// @brief Evaluation metrics implementation (PSNR, SSIM).

#include "training/metrics.hpp"
#include "training/loss.hpp"
#include "training/trainer.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cmath>
#include <fstream>

namespace cugs {

// ---------------------------------------------------------------------------
// compute_psnr
// ---------------------------------------------------------------------------

float compute_psnr(const torch::Tensor& rendered, const torch::Tensor& target) {
    TORCH_CHECK(rendered.sizes() == target.sizes(),
                "PSNR: rendered and target must have same shape");
    TORCH_CHECK(rendered.dim() == 3 && rendered.size(2) == 3,
                "PSNR: expected [H, W, 3] tensors");

    float mse = (rendered - target).pow(2).mean().item<float>();

    // Clamp for near-identical images to avoid log10(0) = -inf.
    if (mse < 1e-10f) {
        return 100.0f;
    }

    return 10.0f * std::log10(1.0f / mse);
}

// ---------------------------------------------------------------------------
// compute_ssim
// ---------------------------------------------------------------------------

float compute_ssim(const torch::Tensor& rendered, const torch::Tensor& target) {
    // Reuse the SSIM implementation from loss.hpp.
    // ssim() returns a per-pixel map [H, W]; take the mean.
    auto ssim_map = cugs::ssim(rendered, target);
    return ssim_map.mean().item<float>();
}

// ---------------------------------------------------------------------------
// EvalResults::to_json
// ---------------------------------------------------------------------------

std::string EvalResults::to_json() const {
    nlohmann::json j;
    j["mean_psnr"] = mean_psnr;
    j["mean_ssim"] = mean_ssim;
    j["num_gaussians"] = num_gaussians;
    j["sh_degree"] = sh_degree;
    j["eval_time_seconds"] = eval_time_seconds;
    j["num_test_images"] = static_cast<int>(per_image.size());

    nlohmann::json images = nlohmann::json::array();
    for (const auto& im : per_image) {
        nlohmann::json entry;
        entry["image_name"] = im.image_name;
        entry["psnr"] = im.psnr;
        entry["ssim"] = im.ssim;
        images.push_back(entry);
    }
    j["per_image"] = images;

    return j.dump(2);
}

// ---------------------------------------------------------------------------
// EvalResults::save_json
// ---------------------------------------------------------------------------

void EvalResults::save_json(const std::filesystem::path& path) const {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        spdlog::error("Failed to open {} for writing", path.string());
        return;
    }
    ofs << to_json() << "\n";
    spdlog::info("Saved evaluation results to {}", path.string());
}

// ---------------------------------------------------------------------------
// evaluate
// ---------------------------------------------------------------------------

EvalResults evaluate(
    const GaussianModel& model,
    const Dataset& dataset,
    const RenderSettings& settings) {

    const size_t num_test = dataset.num_test();
    if (num_test == 0) {
        spdlog::warn("No test images in dataset — nothing to evaluate");
        return {};
    }

    spdlog::info("Evaluating on {} test views...", num_test);

    EvalResults results;
    results.num_gaussians = static_cast<int>(model.num_gaussians());
    results.sh_degree = settings.active_sh_degree;

    auto t_start = std::chrono::steady_clock::now();

    torch::NoGradGuard no_grad;

    float sum_psnr = 0.0f;
    float sum_ssim = 0.0f;

    for (size_t i = 0; i < num_test; ++i) {
        const auto& camera = dataset.test_cameras()[i];
        Image cpu_image = dataset.load_test_image(i);

        // Resize if the loaded image doesn't match camera dimensions
        // (same pattern as trainer.cpp).
        if (cpu_image.width != camera.width ||
            cpu_image.height != camera.height) {
            cpu_image = resize_image(cpu_image, camera.width, camera.height);
        }

        auto target = image_to_tensor(cpu_image, torch::kCUDA);

        // Render.
        auto render_out = render(model, camera, settings);

        // Compute metrics.
        float psnr = compute_psnr(render_out.color, target);
        float ssim_val = compute_ssim(render_out.color, target);

        ImageMetrics im;
        im.image_name = camera.image_name;
        im.psnr = psnr;
        im.ssim = ssim_val;
        results.per_image.push_back(im);

        sum_psnr += psnr;
        sum_ssim += ssim_val;

        spdlog::info("  [{:>3}/{}] {} — PSNR: {:.2f} dB, SSIM: {:.4f}",
                     i + 1, num_test, camera.image_name, psnr, ssim_val);
    }

    results.mean_psnr = sum_psnr / static_cast<float>(num_test);
    results.mean_ssim = sum_ssim / static_cast<float>(num_test);

    auto elapsed = std::chrono::steady_clock::now() - t_start;
    results.eval_time_seconds =
        std::chrono::duration<float>(elapsed).count();

    spdlog::info("Evaluation complete: mean PSNR={:.2f} dB, mean SSIM={:.4f} "
                 "({:.1f}s, {} views)",
                 results.mean_psnr, results.mean_ssim,
                 results.eval_time_seconds, num_test);

    return results;
}

} // namespace cugs
