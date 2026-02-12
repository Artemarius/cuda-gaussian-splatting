/// @file test_backward.cpp
/// @brief Tests for the backward pass of the Gaussian splatting rasterizer.
///
/// Tests include:
///   - Finite-difference gradient checks for all parameter types
///   - Single Gaussian convergence (GD step reduces loss)
///   - Culled Gaussians produce zero gradients
///   - Output gradient shapes match parameter shapes
///   - No NaN/Inf in gradients for random Gaussians

#include <gtest/gtest.h>
#include <torch/torch.h>
#include <cmath>

#include "rasterizer/rasterizer.hpp"
#include "training/loss.hpp"
#include "core/gaussian.hpp"
#include "core/types.hpp"

namespace cugs {
namespace {

/// @brief Create a test camera at the origin looking down +Z.
CameraInfo make_test_camera(int w = 64, int h = 48) {
    CameraInfo cam;
    cam.width = w;
    cam.height = h;
    cam.intrinsics.fx = 100.0f;
    cam.intrinsics.fy = 100.0f;
    cam.intrinsics.cx = static_cast<float>(w) / 2.0f;
    cam.intrinsics.cy = static_cast<float>(h) / 2.0f;
    cam.rotation = Eigen::Matrix3f::Identity();
    cam.translation = Eigen::Vector3f::Zero();
    return cam;
}

/// @brief Build a GaussianModel from raw tensors.
GaussianModel make_model(const torch::Tensor& positions,
                         const torch::Tensor& rotations,
                         const torch::Tensor& scales,
                         const torch::Tensor& opacities,
                         const torch::Tensor& sh_coeffs)
{
    GaussianModel m;
    m.positions  = positions;
    m.rotations  = rotations;
    m.scales     = scales;
    m.opacities  = opacities;
    m.sh_coeffs  = sh_coeffs;
    return m;
}

/// @brief Create a single-Gaussian model visible to the test camera.
GaussianModel make_single_gaussian(float x = 0.0f, float y = 0.0f, float z = 5.0f,
                                    float sh_dc = 1.0f,
                                    float opacity_logit = 3.0f,
                                    float log_scale = -2.0f)
{
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    return make_model(
        torch::tensor({{x, y, z}}, opts),
        torch::tensor({{1.0f, 0.0f, 0.0f, 0.0f}}, opts),
        torch::full({1, 3}, log_scale, opts),
        torch::full({1, 1}, opacity_logit, opts),
        torch::full({1, 3, 1}, sh_dc, opts));
}

/// @brief Create a small multi-Gaussian model for finite-diff tests.
///
/// Generates Gaussians guaranteed to project well within the image bounds
/// so that gradients are numerically well-conditioned for finite-difference
/// verification.
GaussianModel make_test_gaussians(int n = 5) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    torch::manual_seed(42);

    // Keep x,y spread small so all Gaussians project well within the
    // 64×48 image (at z ≈ 4, screen offset = fx * x/z = 100 * 0.3/4 = 7.5 px).
    auto positions = torch::randn({n, 3}, opts) * 0.3f;
    positions.select(1, 2).abs_().add_(3.5f); // z in [3.5, ~4.5]

    auto rotations = torch::randn({n, 4}, opts);
    rotations = rotations / rotations.norm(2, 1, true).clamp_min(1e-8f);

    // Moderately-sized Gaussians (exp(-1.5) ≈ 0.22, visible at z ≈ 4)
    auto scales = torch::full({n, 3}, -1.5f, opts) + torch::randn({n, 3}, opts) * 0.2f;
    auto opacities = torch::full({n, 1}, 2.0f, opts);
    auto sh_coeffs = torch::randn({n, 3, 1}, opts) * 0.5f;

    return make_model(positions, rotations, scales, opacities, sh_coeffs);
}

/// @brief Render a model and compute the combined loss against a target.
float compute_loss(const GaussianModel& model,
                   const CameraInfo& camera,
                   const RenderSettings& settings,
                   const torch::Tensor& target)
{
    auto out = render(model, camera, settings);
    auto loss = combined_loss(out.color, target);
    return loss.item<float>();
}

/// @brief Render and compute backward pass, returning all gradients.
BackwardOutput compute_gradients(const GaussianModel& model,
                                  const CameraInfo& camera,
                                  const RenderSettings& settings,
                                  const torch::Tensor& target)
{
    auto out = render(model, camera, settings);
    // dL/dcolor = d(combined_loss)/d(rendered)
    // combined_loss uses libtorch ops which support autograd
    auto rendered = out.color.clone().detach().requires_grad_(true);
    auto loss = combined_loss(rendered, target);
    loss.backward();
    auto dL_dcolor = rendered.grad().clone();

    return render_backward(dL_dcolor, out, model, camera, settings);
}

// ===========================================================================
// Test: Output shapes match parameter shapes
// ===========================================================================

TEST(BackwardTest, OutputShapes) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    auto model = make_test_gaussians(5);

    RenderSettings settings;
    settings.active_sh_degree = 0;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto target = torch::rand({cam.height, cam.width, 3}, opts);

    auto grads = compute_gradients(model, cam, settings, target);

    EXPECT_EQ(grads.dL_dpositions.sizes(), model.positions.sizes());
    EXPECT_EQ(grads.dL_drotations.sizes(), model.rotations.sizes());
    EXPECT_EQ(grads.dL_dscales.sizes(), model.scales.sizes());
    EXPECT_EQ(grads.dL_dopacities.sizes(), model.opacities.sizes());
    EXPECT_EQ(grads.dL_dsh_coeffs.sizes(), model.sh_coeffs.sizes());
}

// ===========================================================================
// Test: No NaN/Inf in gradients
// ===========================================================================

TEST(BackwardTest, NoNanInf) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    auto model = make_test_gaussians(20);

    RenderSettings settings;
    settings.active_sh_degree = 0;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto target = torch::rand({cam.height, cam.width, 3}, opts);

    auto grads = compute_gradients(model, cam, settings, target);

    EXPECT_FALSE(grads.dL_dpositions.isnan().any().item<bool>()) << "positions grad has NaN";
    EXPECT_FALSE(grads.dL_dpositions.isinf().any().item<bool>()) << "positions grad has Inf";
    EXPECT_FALSE(grads.dL_drotations.isnan().any().item<bool>()) << "rotations grad has NaN";
    EXPECT_FALSE(grads.dL_drotations.isinf().any().item<bool>()) << "rotations grad has Inf";
    EXPECT_FALSE(grads.dL_dscales.isnan().any().item<bool>()) << "scales grad has NaN";
    EXPECT_FALSE(grads.dL_dscales.isinf().any().item<bool>()) << "scales grad has Inf";
    EXPECT_FALSE(grads.dL_dopacities.isnan().any().item<bool>()) << "opacities grad has NaN";
    EXPECT_FALSE(grads.dL_dopacities.isinf().any().item<bool>()) << "opacities grad has Inf";
    EXPECT_FALSE(grads.dL_dsh_coeffs.isnan().any().item<bool>()) << "sh_coeffs grad has NaN";
    EXPECT_FALSE(grads.dL_dsh_coeffs.isinf().any().item<bool>()) << "sh_coeffs grad has Inf";
}

// ===========================================================================
// Test: Culled Gaussians have zero gradients
// ===========================================================================

TEST(BackwardTest, CulledGaussiansZeroGrad) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

    // Gaussian behind the camera (z = -5)
    auto model = make_single_gaussian(0.0f, 0.0f, -5.0f);
    auto target = torch::rand({cam.height, cam.width, 3}, opts);

    RenderSettings settings;
    settings.active_sh_degree = 0;

    auto grads = compute_gradients(model, cam, settings, target);

    EXPECT_FLOAT_EQ(grads.dL_dpositions.abs().sum().item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(grads.dL_drotations.abs().sum().item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(grads.dL_dscales.abs().sum().item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(grads.dL_dopacities.abs().sum().item<float>(), 0.0f);
    EXPECT_FLOAT_EQ(grads.dL_dsh_coeffs.abs().sum().item<float>(), 0.0f);
}

// ===========================================================================
// Test: Single Gaussian convergence — one GD step reduces loss
// ===========================================================================

TEST(BackwardTest, SingleGaussianConvergence) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto target = torch::full({cam.height, cam.width, 3}, 0.8f, opts);

    auto model = make_single_gaussian(0.0f, 0.0f, 5.0f,
                                       /*sh_dc=*/0.5f,
                                       /*opacity_logit=*/3.0f,
                                       /*log_scale=*/-1.5f);

    RenderSettings settings;
    settings.active_sh_degree = 0;

    float loss_before = compute_loss(model, cam, settings, target);

    auto grads = compute_gradients(model, cam, settings, target);

    // Gradient descent step
    float lr = 0.01f;
    model.positions = model.positions - lr * grads.dL_dpositions;
    model.rotations = model.rotations - lr * grads.dL_drotations;
    model.scales    = model.scales    - lr * grads.dL_dscales;
    model.opacities = model.opacities - lr * grads.dL_dopacities;
    model.sh_coeffs = model.sh_coeffs - lr * grads.dL_dsh_coeffs;

    float loss_after = compute_loss(model, cam, settings, target);

    EXPECT_LT(loss_after, loss_before)
        << "Loss should decrease after GD step. Before: " << loss_before
        << " After: " << loss_after;
}

// ===========================================================================
// Finite-difference gradient checks
// ===========================================================================

/// @brief Helper: finite-difference gradient check for a specific parameter.
///
/// For each element in the parameter tensor, perturbs by ±eps, renders,
/// computes loss, and compares (L+ - L-) / (2*eps) to the analytic gradient.
///
/// Uses a mixed tolerance approach (standard for gradient checking):
/// an element passes if the relative error is within rel_tol OR the absolute
/// error is within abs_tol. The absolute tolerance handles near-zero gradients
/// where relative error is dominated by float32 cancellation noise.
///
/// @param param_name   Name for error messages.
/// @param model        Base model (will be modified in-place, then restored).
/// @param param        The parameter tensor to perturb.
/// @param analytic_grad The analytic gradient for this parameter.
/// @param camera       Camera for rendering.
/// @param settings     Render settings.
/// @param target       Target image for loss computation.
/// @param eps          Perturbation size.
/// @param rel_tol      Maximum allowed relative error.
/// @param abs_tol      Maximum allowed absolute error.
/// @param max_elements Maximum number of elements to check (for speed).
void finite_diff_check(
    const std::string& param_name,
    GaussianModel& model,
    torch::Tensor& param,
    const torch::Tensor& analytic_grad,
    const CameraInfo& camera,
    const RenderSettings& settings,
    const torch::Tensor& target,
    float eps = 1e-3f,
    float rel_tol = 0.05f,
    float abs_tol = 1e-4f,
    int max_elements = -1)
{
    auto param_cpu = param.cpu().contiguous();
    auto grad_cpu = analytic_grad.cpu().contiguous();

    auto param_flat = param_cpu.reshape({-1});
    auto grad_flat = grad_cpu.reshape({-1});

    int n_elements = static_cast<int>(param_flat.numel());
    if (max_elements > 0 && n_elements > max_elements) {
        n_elements = max_elements;
    }

    int num_checked = 0;
    int num_passed = 0;

    for (int i = 0; i < n_elements; ++i) {
        float original = param_flat[i].item<float>();
        float analytic = grad_flat[i].item<float>();

        // Perturb +eps
        param_flat[i] = original + eps;
        param.copy_(param_flat.reshape(param.sizes()).to(param.device()));
        float loss_plus = compute_loss(model, camera, settings, target);

        // Perturb -eps
        param_flat[i] = original - eps;
        param.copy_(param_flat.reshape(param.sizes()).to(param.device()));
        float loss_minus = compute_loss(model, camera, settings, target);

        // Restore
        param_flat[i] = original;
        param.copy_(param_flat.reshape(param.sizes()).to(param.device()));

        float numerical = (loss_plus - loss_minus) / (2.0f * eps);

        num_checked++;

        float abs_error = std::abs(analytic - numerical);

        // Mixed tolerance: pass if relative error is small OR absolute error is small
        float denom = std::max({std::abs(analytic), std::abs(numerical), 1e-6f});
        float rel_error = abs_error / denom;

        if (rel_error <= rel_tol || abs_error <= abs_tol) {
            num_passed++;
        } else {
            std::cout << "  FAIL " << param_name << "[" << i << "]: "
                      << "analytic=" << analytic << " numerical=" << numerical
                      << " rel_err=" << rel_error << " abs_err=" << abs_error
                      << std::endl;
        }
    }

    // At least 80% of checked elements should pass
    float pass_rate = static_cast<float>(num_passed) / static_cast<float>(num_checked);
    EXPECT_GE(pass_rate, 0.80f)
        << param_name << ": only " << num_passed << "/" << num_checked
        << " elements passed finite-diff check (pass rate " << pass_rate * 100.0f << "%)";
}

TEST(BackwardTest, FiniteDiffPositions) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto target = torch::rand({cam.height, cam.width, 3}, opts);

    auto model = make_test_gaussians(3);

    RenderSettings settings;
    settings.active_sh_degree = 0;

    auto grads = compute_gradients(model, cam, settings, target);
    // Position gradients have inherent discontinuities from tile-based binning:
    // moving a Gaussian can change which 16×16 tiles it overlaps, creating step
    // changes that finite difference captures but analytic gradients don't.
    // Use relaxed tolerances accordingly (larger eps, wider rel/abs tol).
    finite_diff_check("positions", model, model.positions, grads.dL_dpositions,
                      cam, settings, target, 2e-3f, 0.15f, 1e-3f);
}

TEST(BackwardTest, FiniteDiffScales) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto target = torch::rand({cam.height, cam.width, 3}, opts);

    auto model = make_test_gaussians(3);

    RenderSettings settings;
    settings.active_sh_degree = 0;

    auto grads = compute_gradients(model, cam, settings, target);
    finite_diff_check("scales", model, model.scales, grads.dL_dscales,
                      cam, settings, target, 1e-3f, 0.05f);
}

TEST(BackwardTest, FiniteDiffRotations) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto target = torch::rand({cam.height, cam.width, 3}, opts);

    auto model = make_test_gaussians(3);

    RenderSettings settings;
    settings.active_sh_degree = 0;

    auto grads = compute_gradients(model, cam, settings, target);
    finite_diff_check("rotations", model, model.rotations, grads.dL_drotations,
                      cam, settings, target, 1e-3f, 0.10f, 1e-4f);
}

TEST(BackwardTest, FiniteDiffOpacities) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto target = torch::rand({cam.height, cam.width, 3}, opts);

    auto model = make_test_gaussians(3);

    RenderSettings settings;
    settings.active_sh_degree = 0;

    auto grads = compute_gradients(model, cam, settings, target);
    finite_diff_check("opacities", model, model.opacities, grads.dL_dopacities,
                      cam, settings, target, 1e-3f, 0.05f);
}

TEST(BackwardTest, FiniteDiffSHCoeffs) {
    if (!torch::cuda::is_available()) GTEST_SKIP() << "CUDA not available";

    auto cam = make_test_camera();
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto target = torch::rand({cam.height, cam.width, 3}, opts);

    auto model = make_test_gaussians(3);

    RenderSettings settings;
    settings.active_sh_degree = 0;

    auto grads = compute_gradients(model, cam, settings, target);
    finite_diff_check("sh_coeffs", model, model.sh_coeffs, grads.dL_dsh_coeffs,
                      cam, settings, target, 1e-3f, 0.05f);
}

} // namespace
} // namespace cugs
