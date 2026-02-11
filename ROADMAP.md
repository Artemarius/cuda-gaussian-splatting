# ROADMAP.md — cuda-gaussian-splatting

## Development Environment

- **OS**: Windows 10 Pro 22H2 (no cross-platform)
- **Compiler**: MSVC (Visual Studio 2022)
- **GPU**: RTX 3060 6GB VRAM (compute capability 8.6)
- **CUDA**: 12.6+ (to be installed — ensure VS integration during setup)
- **C++ Standard**: `/std:c++latest` — target C++20 as baseline, use C++23 features where MSVC supports them (`std::expected`, `std::format` work; some ranges features are spotty)
- **Build**: CMake 3.24+ with both VS generator (for debugging) and Ninja (for release)

### 6GB VRAM Constraints

The RTX 3060 6GB is sufficient for development but will hit limits on large scenes. Plan for this from day one:

- Start with smaller scenes: Truck, Train (Tanks & Temples) before Garden (Mip-NeRF 360)
- Add `--max-gaussians` CLI flag early
- Instrument VRAM usage with `cudaMemGetInfo` — add a simple budget tracker utility
- Consider SH degree 2 (9 coefficients) during development instead of degree 3 (16) to save memory
- Implement aggressive pruning early
- Garden at full quality (30K iterations, SH degree 3) can use 8-12GB — may need to reduce resolution or Gaussian count

---

## Phase 0: Project Scaffold

**Goal**: Empty project that compiles, links CUDA, and runs a trivial kernel.

### Tasks

- [x] Initialize git repo with `.gitignore` (build dirs, libtorch, `.vs/`, `*.ply`, datasets)
- [x] Create `CMakeLists.txt` with CUDA language enabled, MSVC flags, libtorch integration
- [x] Create `CMakePresets.json` with two presets:
  - `dev` — VS 2022 generator, Debug/RelWithDebInfo, for IDE debugging
  - `release` — Ninja generator, Release, for fast builds
- [x] Set up `vcpkg.json` manifest for: Eigen3, spdlog, gtest, nlohmann-json, glfw3, imgui
- [x] Download and configure libtorch (C++ CUDA-enabled Windows distribution) in `external/libtorch/`
  - **Watch out**: libtorch uses `/MD` — ensure the whole project uses `/MD` consistently
  - Set `CMAKE_PREFIX_PATH` to libtorch directory
- [x] Create directory structure:
  ```
  src/core/  src/data/  src/rasterizer/  src/optimizer/
  src/training/  src/viewer/  src/utils/
  apps/  tests/  benchmarks/  docs/
  ```
- [x] Write `src/utils/cuda_utils.cuh` with:
  - `CUDA_CHECK()` macro wrapping `cudaGetLastError`
  - `CUDA_SYNC_CHECK()` for debug builds
  - VRAM usage query helper (`vram_used_mb()`, `vram_free_mb()`)
- [x] Write a trivial `apps/hello_cuda.cpp` + `src/utils/cuda_info.cu` that launches a kernel and prints GPU info
- [x] Verify: `cmake --preset dev`, open `.sln`, build, run, see GPU info printed
- [x] Verify: `cmake --preset release && cmake --build build-release`
- [x] Write first Google Test: `tests/test_cuda_utils.cpp` — verify GPU is accessible

### Key CMake Notes (Windows/MSVC)

```cmake
# CUDA architecture for RTX 3060
set(CMAKE_CUDA_ARCHITECTURES 86)

# C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 20)  # NVCC doesn't support C++23 fully

# MSVC flags
add_compile_options(/W4 /utf-8)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(/O2 /DNDEBUG)
endif()
```

### Definition of Done

A clean build on Windows with CUDA kernel execution and unit test passing.

---

## Phase 1: COLMAP Data Loading

**Goal**: Parse COLMAP sparse reconstruction, load images, verify correctness visually.

### Background

COLMAP outputs binary files: `cameras.bin`, `images.bin`, `points3D.bin`. These contain camera intrinsics (focal length, principal point, distortion), camera extrinsics (rotation quaternion + translation per image), and sparse 3D points with colors.

### Tasks

- [x] `src/data/colmap_loader.hpp/.cpp` — binary parser for COLMAP format
  - Parse `cameras.bin`: camera model ID, width, height, params (fx, fy, cx, cy, distortion)
  - Parse `images.bin`: image ID, quaternion (COLMAP uses wxyz), translation, camera ID, filename
  - Parse `points3D.bin`: XYZ position, RGB color, reprojection error, track (which images see it)
  - Support at least SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL camera models
  - **Note**: COLMAP stores world-to-camera rotation and translation. The camera center in world coords is `C = -R^T * t`
- [x] `src/data/image_io.hpp/.cpp` — image loading via stb_image
  - Load images, convert to float [0,1], handle RGB/RGBA
  - Resize support for memory-constrained development
- [x] `src/data/dataset.hpp/.cpp` — dataset abstraction
  - Combines cameras, images, sparse points
  - Train/test split (COLMAP convention: every 8th image for test)
  - Provides iterators for training loop
- [x] `src/core/types.hpp` — define core structs:
  ```cpp
  struct Camera {
      int width, height;
      float fx, fy, cx, cy;
      // distortion params if needed
      Eigen::Matrix4f world_to_camera;  // extrinsics
  };
  ```
- [x] Download a small test dataset (e.g., Truck from Tanks & Temples)
- [x] **Sanity check**: write a test that loads the dataset and verifies:
  - Number of cameras/images matches expected
  - Sparse points are in a reasonable bounding box
  - Camera centers (computed from extrinsics) form a reasonable pattern around the scene
- [x] (Optional) Write a quick point cloud dumper to PLY for visual verification in MeshLab/CloudCompare

### Tests

- `tests/test_colmap_loader.cpp` — parse a known small COLMAP output, verify values
- `tests/test_dataset.cpp` — verify train/test split, image loading

### Definition of Done

Can load a COLMAP dataset, print summary statistics, and verify camera geometry is correct.

---

## Phase 2: Gaussian Initialization & Core Types

**Goal**: Initialize 3D Gaussians from sparse SfM points with all required parameters.

### Background

Each Gaussian has: position (3), color as SH coefficients (3 × 16 for degree 3, or 3 × 9 for degree 2), opacity (1), rotation as quaternion (4), scale (3). Total: ~62 floats per Gaussian at SH degree 3.

The initial Gaussians come from COLMAP sparse points. Initial values:
- Position: from point3D XYZ
- Color (SH DC term): from point3D RGB, converted to SH coefficient space (color / 0.2820947917... - 0.5)
- Higher SH bands: zero
- Opacity: inverse sigmoid of 0.1 (≈ -2.197)
- Scale: log of mean distance to k-nearest neighbors (paper uses k=3)
- Rotation: identity quaternion [1, 0, 0, 0]

### Tasks

- [x] `src/core/gaussian.hpp` — Gaussian parameter storage
  - Design decision: Structure of Arrays (SoA) for GPU efficiency, not AoS
  - Store all parameters as contiguous `torch::Tensor` buffers:
    ```cpp
    struct GaussianModel {
        torch::Tensor positions;     // [N, 3]
        torch::Tensor sh_coeffs;     // [N, C, (D+1)²] where C=3 (RGB), D=SH degree
        torch::Tensor opacities;     // [N, 1] (logit space)
        torch::Tensor rotations;     // [N, 4] (quaternions, wxyz)
        torch::Tensor scales;        // [N, 3] (log space)
    };
    ```
  - All parameters in their activation-function-friendly spaces (log scale, logit opacity, raw quaternion)
  - Methods: `num_gaussians()`, `to_device()`, `save_ply()`, `load_ply()`
- [x] `src/core/sh.hpp/.cu` — Spherical Harmonics evaluation
  - SH basis functions up to degree 3
  - `evaluate_sh(degree, sh_coeffs, direction)` → RGB color
  - CPU reference implementation first, then CUDA kernel
  - **Reference**: Appendix A of the original paper, or Ramamoorthi & Hanrahan 2001
- [x] `src/core/gaussian_init.cpp` — initialization from sparse points
  - k-NN for scale initialization (use a simple brute-force or spatial hash for now)
  - SH coefficient initialization from point colors
  - Reasonable defaults for opacity and rotation
- [x] `src/utils/ply_io.hpp/.cpp` — PLY writer/reader for Gaussian models
  - Match the format used by the reference implementation for compatibility with existing viewers

### Tests

- `tests/test_gaussian_model.cpp` — create, save, reload, verify roundtrip
- `tests/test_sh.cpp` — verify SH evaluation against known values (e.g., constant color should be DC-only)
- `tests/test_gaussian_init.cpp` — initialize from mock sparse points, verify reasonable scale/opacity

### Definition of Done

Can initialize Gaussians from COLMAP data, save to PLY, and load in an external viewer.

---

## Phase 3: Forward Rasterizer (CUDA)

**Goal**: Render an image from Gaussians. This is the hardest and most important phase.

### Background

The forward pass:
1. Project each 3D Gaussian to 2D (compute 2D mean and 2D covariance)
2. Compute the bounding rect of each 2D Gaussian (e.g., 3σ ellipse)
3. Assign Gaussians to screen tiles (e.g., 16×16 pixels)
4. Sort Gaussians per tile by depth
5. For each pixel, alpha-composite the Gaussians front-to-back

### Sub-phases

#### Phase 3a: Projection (CUDA kernel)

- [x] `src/rasterizer/projection.cuh` — `__device__` math helpers:
  - `quat_to_rotation()` — quaternion to 3×3 rotation matrix
  - `compute_cov_3d()` — 3D covariance from rotation + log-scale: `Σ = R S Sᵀ Rᵀ`
  - `compute_cov_2d()` — project 3D covariance to 2D via Jacobian: `Σ' = J W Σ Wᵀ Jᵀ + 0.3I`
  - `compute_radius()` — pixel radius from 2D covariance eigenvalues (3σ)
  - `compute_cov_2d_inverse()` — inverse of 2×2 symmetric matrix
- [x] `src/rasterizer/projection.cu` — `k_project_gaussians` kernel + host launcher
  - One thread per Gaussian (256 threads/block)
  - Transform to camera space, frustum cull, project to screen, compute cov, radius, tile count
  - SH evaluation via existing `evaluate_sh_cuda()` for view-dependent colors
  - Output: means_2d, depths, cov_2d_inv, radii, tiles_touched, rgb, sigmoid opacities
- [x] `src/rasterizer/projection.hpp` — host-side `ProjectionOutput` struct and `project_gaussians()` declaration

**Math reference** (equation numbers from Kerbl et al.):
- 3D covariance: Eq. 6 — `Σ = R S Sᵀ Rᵀ`
- 2D covariance: Eq. 5 — `Σ' = J W Σ Wᵀ Jᵀ` (adds 0.3 to diagonal for numerical stability / anti-aliasing)
- Jacobian J: from Zwicker et al., EWA Splatting — Jacobian of the local affine approximation of the projective transform

#### Phase 3b: Tile Assignment & Sorting

- [x] `src/rasterizer/sorting.hpp` — `SortingOutput` struct, `sort_gaussians()` declaration, `kTileSize=16`
- [x] `src/rasterizer/sorting.cu` — three kernels + CUB sort:
  - Prefix sum on `tiles_touched` → per-Gaussian offsets + total pair count P
  - `k_fill_sort_pairs` — writes `(tile_id << 32 | depth_bits, gaussian_idx)` pairs
  - `cub::DeviceRadixSort::SortPairs` on uint64 keys
  - `k_compute_tile_ranges` — detects tile boundaries → `tile_ranges[num_tiles, 2]`

#### Phase 3c: Rasterization (Alpha Compositing)

- [x] `src/rasterizer/forward.cuh` — `SharedGaussian` struct (40 bytes), block constants
- [x] `src/rasterizer/forward.cu` — `k_rasterize_forward` kernel + host launcher
  - Grid: `dim3(num_tiles_x, num_tiles_y)`, Block: `dim3(16, 16)` = 256 threads
  - Shared memory: batch of 256 `SharedGaussian` entries (10 KB per block)
  - Per pixel: accumulate `color += rgb * alpha * T; T *= (1 - alpha)`
  - Early termination when `T < 1/255`
  - Tracks `n_contrib` per pixel (for backward pass)
  - Final blend: `color += T * background`
- [x] `src/rasterizer/forward.hpp` — `ForwardOutput` struct, `rasterize_forward()` declaration

#### Phase 3d: Host API

- [x] `src/rasterizer/rasterizer.hpp` — public `RenderSettings`, `RenderOutput` structs, `render()` declaration
- [x] `src/rasterizer/rasterizer.cpp` — host orchestration:
  - Converts CameraInfo to raw float view matrix (Eigen → raw array, avoids Eigen in CUDA)
  - Calls `project_gaussians()` (includes SH eval)
  - Calls `sort_gaussians()`
  - Calls `rasterize_forward()`
  - Packs all intermediates into `RenderOutput` (retained for backward pass)

### Validation Strategy

- [x] **Gradient-free visual test**: random Gaussians render without NaN/Inf, produce colored output
- [x] **Single Gaussian test**: place one Gaussian at known position, verify 2D mean/depth/color are correct
- [x] **Depth ordering test**: two overlapping Gaussians at different depths, front Gaussian dominates
- [ ] **Render the initial sparse Gaussians**: load a dataset, initialize Gaussians, render from training camera — should see a noisy but recognizable point cloud rendering
- [ ] Compare output against the reference implementation on the same input (if possible)

### Performance Notes

- Expect ~1-5 ms per frame for moderate scenes on RTX 3060
- Profile with Nsight Compute once functional
- The sorting step is often the bottleneck for large Gaussian counts

### Tests

- [x] `tests/test_projection.cpp` — 7 tests: single Gaussian in front, behind-camera culling, off-center projection, batch no-NaN, anisotropic, empty input, scale modifier
- [x] `tests/test_rasterizer.cpp` — 7 tests: empty scene background, single Gaussian center pixel, depth ordering, background blending, random no-NaN, transmittance/contrib, output shapes

### Architecture Note — .hpp vs .cuh split

MSVC compiles `.cpp` files and cannot handle `__device__` code. Each module uses:
- `.hpp` — host-only declarations (included by `.cpp` and `.cu`)
- `.cuh` — `__device__` helpers (included only by `.cu` files)
- `.cu` — kernel implementations + host launchers

`rasterizer.cpp` includes only `.hpp` headers, compiling cleanly with MSVC.

### Definition of Done

Can render a recognizable (if untrained) image from initial Gaussians at interactive framerates.

---

## Phase 4: Loss Computation

**Goal**: Compute L1 + SSIM loss between rendered and ground truth images.

### Background

The training loss is: `L = (1 - λ) L₁ + λ L_SSIM`, where λ = 0.2 (paper default).

### Tasks

- [ ] `src/training/loss.hpp/.cu`
  - **L1 loss**: trivial, per-pixel absolute difference, mean reduction
  - **SSIM loss**: `1 - SSIM(rendered, target)`
    - SSIM uses 11×11 Gaussian-weighted window (σ = 1.5)
    - Compute per-channel: luminance, contrast, structure terms
    - Can implement as 2D convolution with Gaussian kernel (separable for efficiency)
    - Use CUDA for the convolution, or libtorch's `conv2d` as initial implementation
  - Combined loss with configurable λ
- [ ] Verify SSIM against a known implementation (e.g., compute on identical images → should be 1.0)

### Tests

- `tests/test_loss.cpp` — L1 of identical images = 0, SSIM of identical images = 1.0, known difference values

### Definition of Done

Loss computation works and matches expected values on test cases.

---

## Phase 5: Backward Rasterizer (CUDA)

**Goal**: Compute gradients of the rendering loss with respect to all Gaussian parameters.

### Background

This is the second-hardest part after the forward pass. The backward pass traverses the same tile structure in reverse, computing gradients via the chain rule. For each pixel, gradients flow from `dL/dColor` back to each contributing Gaussian's parameters.

### Tasks

- [ ] `src/rasterizer/backward.cuh/.cu`
  - `k_rasterize_backward` kernel:
    - Same tile structure as forward, but traverse Gaussians back-to-front
    - For each pixel, starting from the last contributing Gaussian:
      - Receive `dL/d(pixel_color)` from loss
      - Compute `dL/d(gaussian_color)`, `dL/d(gaussian_opacity)`, `dL/d(gaussian_2d_mean)`, `dL/d(gaussian_2d_cov)`
      - Accumulate transmittance in reverse
    - Atomic add to accumulate gradients across pixels (multiple pixels contribute to each Gaussian)
  - `k_project_backward` kernel:
    - Backpropagate from 2D parameters to 3D parameters
    - `dL/d(position_3d)` from `dL/d(mean_2d)` via projection Jacobian
    - `dL/d(quaternion)` and `dL/d(scale)` from `dL/d(covariance_2d)` via the chain of covariance computation
    - `dL/d(sh_coefficients)` from `dL/d(color)` via SH evaluation Jacobian
    - `dL/d(opacity_logit)` from `dL/d(opacity)` via sigmoid derivative
  - All gradients are analytic — no autograd, no finite differences in the CUDA kernels

### Validation Strategy

- [ ] **Gradient checking**: for a small number of Gaussians, compare analytic gradients to finite difference approximation
  - Perturb each parameter by ε, re-render, compute `(L(θ+ε) - L(θ-ε)) / (2ε)`
  - Compare to analytic gradient — relative error should be < 1e-3 for float32
  - This is slow but essential for correctness
- [ ] **Convergence test**: optimize a single Gaussian to match a simple target (e.g., colored circle) — verify loss decreases monotonically with gradient descent

### Math Reference

Key gradient derivations (all in Kerbl et al., Appendix):
- dL/dΣ' (2D covariance gradient) → dL/dΣ (3D) → dL/dR, dL/dS
- dL/dμ' (2D mean gradient) → dL/dμ (3D position)
- Quaternion gradients require the derivative of R(q) w.r.t. q components

### Tests

- `tests/test_backward.cpp` — finite difference gradient check for each parameter type

### Definition of Done

Analytic gradients match finite differences to reasonable precision. A simple optimization test converges.

---

## Phase 6: Basic Training Loop

**Goal**: Train Gaussians on a real scene — get first converging results.

### Background

Training loop: for each iteration, randomly sample a training image, render from that camera, compute loss, backpropagate, update parameters. The original paper trains for 30,000 iterations.

### Tasks

- [ ] `src/training/lr_schedule.hpp` — learning rate schedules
  - Position: exponential decay from 1.6e-4 to 1.6e-6 (paper values)
  - SH: constant 2.5e-3, higher bands start at iteration 1000 (progressive activation)
  - Opacity: constant 0.05
  - Scale: constant 5e-3
  - Rotation: constant 1e-3
  - Implement as a simple function: `float get_lr(param_type, iteration)`
- [ ] `src/optimizer/adam.hpp/.cpp` — initial Adam optimizer using libtorch
  - Use `torch::optim::Adam` for first implementation (correct but slow)
  - Per-parameter-group learning rates
  - This gets replaced by fused CUDA Adam in Phase 8, but we need a working baseline first
- [ ] `src/training/trainer.hpp/.cpp` — main training loop
  - Iteration: sample image → render → loss → backward → optimizer step
  - Log loss every N iterations (spdlog)
  - Save checkpoint every N iterations (PLY + optimizer state)
  - Print VRAM usage periodically
  - CLI args: dataset path, output path, num iterations, eval flag
- [ ] `apps/train_main.cpp` — entry point
  - Parse CLI arguments
  - Set up dataset, model, optimizer, trainer
  - Run training loop
- [ ] **Progressive SH**: start with degree 0 (DC only), increase to 1 at iteration 1000, to 2 at 2000, to 3 at 3000

### First Training Run

- Train on Truck scene (smaller than Garden)
- Start with just 1000 iterations and monitor loss curve
- Expect: loss should decrease noticeably within the first few hundred iterations
- If loss plateaus immediately: likely gradient bug. Go back to Phase 5 validation.
- If loss diverges: likely learning rate too high or numerical instability in projection

### Tests

- `tests/test_training.cpp` — run 100 iterations on a tiny synthetic scene, verify loss decreases

### Definition of Done

Can train on a real dataset. Loss decreases. Rendered images from training views are recognizably similar to ground truth (even if blurry/noisy at this stage).

---

## Phase 7: Adaptive Density Control

**Goal**: Clone, split, and prune Gaussians during training to improve quality.

### Background

The original paper's densification strategy (every 100 iterations, starting at 500, stopping at 15000):
- **Clone**: Gaussians with large accumulated position gradients AND small scale → duplicate with slight offset
- **Split**: Gaussians with large accumulated position gradients AND large scale → split into two smaller Gaussians
- **Prune**: remove Gaussians with very low opacity (< threshold after sigmoid)
- **Reset opacity**: every 3000 iterations, reset all opacities to a low value (paper: sigmoid⁻¹(0.01))

### Tasks

- [ ] `src/optimizer/densification.hpp/.cpp` — original densification strategy
  - Accumulate position gradient magnitudes over iterations between densification steps
  - Clone/split/prune logic
  - Handle parameter tensor resizing (add/remove rows) — this is fiddly with libtorch
  - Reset optimizer state (Adam moments) for new Gaussians
  - **Memory-aware**: check VRAM before adding Gaussians, skip densification if near limit
- [ ] Tune thresholds for 6GB VRAM:
  - May need lower `max_gaussians` cap
  - More aggressive pruning
  - Start densification earlier or stop earlier

### MCMC Densification (Kheradmand et al.)

- [ ] `src/optimizer/mcmc.hpp/.cpp` — MCMC-based densification (Phase 10, but plan the interface now)
  - Alternative to clone/split/prune
  - Treats Gaussians as particles in an MCMC chain
  - Relocate low-opacity Gaussians instead of pruning + creating new ones
  - More memory-stable (fixed Gaussian count)
  - **This is better for 6GB VRAM** since it doesn't grow the Gaussian count

### Tests

- `tests/test_densification.cpp` — verify clone/split/prune on mock data, check parameter counts

### Definition of Done

Training with densification produces noticeably sharper results than without. Gaussian count grows and stabilizes.

---

## Phase 8: Fused Adam Optimizer (CUDA)

**Goal**: Replace libtorch Adam with a single fused CUDA kernel for 2x+ speedup.

### Background

The standard Adam update requires multiple kernel launches per parameter: compute gradient, update first moment, update second moment, compute bias correction, apply update. A fused kernel does all of this in a single launch, eliminating kernel launch overhead and improving memory locality.

### Tasks

- [ ] `src/optimizer/fused_adam.cuh/.cu`
  - Single kernel: reads gradient, updates m (first moment), v (second moment), applies bias-corrected update
  - Handle per-parameter learning rates
  - Template on parameter count per element (3 for position, 4 for quaternion, etc.)
  - Epsilon, beta1, beta2 as kernel parameters
- [ ] Benchmark against libtorch Adam — expect 1.5-2.5x speedup on the optimizer step
- [ ] Ensure training results are numerically equivalent (within floating point tolerance)

### Tests

- `tests/test_fused_adam.cpp` — compare one step of fused Adam vs libtorch Adam, verify numerical equivalence

### Definition of Done

Fused Adam produces identical training results with measurable speedup.

---

## Phase 9: Evaluation & Quality Metrics

**Goal**: Compute PSNR, SSIM, and (optionally) LPIPS on test views. Compare to paper results.

### Tasks

- [ ] `src/training/metrics.hpp/.cpp`
  - PSNR: `10 * log10(1.0 / MSE)` — trivial
  - SSIM: reuse from loss computation
  - LPIPS: optional, requires a pretrained VGG — may skip or implement later via libtorch model loading
- [ ] Evaluation script in `apps/eval_main.cpp`:
  - Load trained model
  - Render all test views
  - Compute per-image and mean metrics
  - Output results as JSON
- [ ] Run on Truck and/or Train datasets, compare to paper Table 1:
  - Truck: PSNR ~25.2, SSIM ~0.88 (Mip-NeRF 360 dataset)
  - Don't expect exact match on 6GB GPU if Gaussian count is limited

### Definition of Done

Can evaluate trained models and produce quality metrics. Results are in the right ballpark.

---

## Phase 10: MCMC Densification

**Goal**: Implement the MCMC-based densification strategy from Kheradmand et al.

### Background

Instead of clone/split/prune, MCMC treats optimization as sampling from a posterior. Low-contribution Gaussians are relocated rather than destroyed. Key advantage: fixed memory footprint — important for the 6GB constraint.

### Tasks

- [ ] `src/optimizer/mcmc.hpp/.cpp`
  - Stochastic relocation of low-opacity Gaussians to high-gradient regions
  - Noise injection for exploration
  - Temperature scheduling
- [ ] Compare MCMC vs original densification:
  - Memory usage over training
  - Final quality metrics
  - Training time
- [ ] Make densification strategy a CLI flag: `--densification {default, mcmc}`

### Definition of Done

MCMC densification works and produces comparable quality with more stable memory usage.

---

## Phase 11: Real-Time Viewer

**Goal**: Interactive OpenGL viewer for trained models.

### Note

This can be developed in parallel once Phase 2 (PLY I/O) is done. It shares no code with the training pipeline except the PLY format and Gaussian data structures.

### Tasks

- [ ] `src/viewer/viewer.hpp/.cpp`
  - GLFW window creation, OpenGL context (4.3+ for compute shaders, or just use fragment shaders)
  - Load trained `.ply` model
  - Camera controls: orbit, pan, zoom (mouse + keyboard)
  - Render loop: project Gaussians → sort → splat (can be a simplified version of the rasterizer, or a compute shader reimplementation)
- [ ] Render modes:
  - RGB (default)
  - Depth visualization
  - Gaussian count per pixel (heatmap)
- [ ] Dear ImGui overlay:
  - FPS counter
  - Camera parameters
  - Number of Gaussians
  - Render mode selector
  - SH degree selector
- [ ] `apps/viewer_main.cpp` — entry point
  - `./build/viewer output/garden/point_cloud.ply`

### Performance Target

- ≥30 FPS at 1080p for typical trained scenes (~1M Gaussians)
- On RTX 3060 this should be achievable with the tile-based approach

### Definition of Done

Can interactively view trained models at 30+ FPS with smooth camera controls.

---

## Phase 12: Polish & Documentation

**Goal**: Clean up, document, benchmark, write blog post.

### Tasks

- [ ] Code cleanup: consistent formatting, remove dead code, resolve TODOs
- [ ] Doxygen comments on all public APIs
- [ ] Performance benchmarks:
  - Training time per iteration breakdown (render, loss, backward, optimizer, densification)
  - Peak VRAM usage per scene
  - Rendering FPS at various resolutions
  - Comparison to reference implementation numbers (from their paper/repo)
- [ ] Update README with actual results:
  - Quality metrics table
  - Training time comparison
  - Example rendered images (before/after training)
- [ ] Blog post explaining the implementation journey:
  - Key challenges and solutions
  - Performance optimization story
  - Math explanations with diagrams
  - What was learned about ML training dynamics

### Definition of Done

Repository is clean, documented, and showcases the work effectively.

---

## Quick Reference: Key Paper Equations

For convenience when implementing — all from Kerbl et al. 2023 unless noted:

| What | Equation | Notes |
|---|---|---|
| 3D Covariance | `Σ = R S Sᵀ Rᵀ` | R from quaternion, S = diag(scale) |
| 2D Covariance | `Σ' = J W Σ Wᵀ Jᵀ` | J = Jacobian of projective transform |
| Jacobian J | See Zwicker et al. Eq. 29 | Local affine approx of perspective |
| SH evaluation | `c(d) = Σₗ Σₘ cₗₘ Yₗₘ(d)` | d = view direction, cₗₘ = coefficients |
| Alpha compositing | `C = Σᵢ cᵢ αᵢ Πⱼ<ᵢ (1-αⱼ)` | Front-to-back accumulation |
| Gaussian weight | `α = opacity × exp(-½ Δᵀ Σ'⁻¹ Δ)` | Δ = pixel - mean_2d |
| Training loss | `L = (1-λ) L₁ + λ L_SSIM` | λ = 0.2 |

---

## Dependencies Checklist

| Dependency | Install Method | Notes |
|---|---|---|
| CUDA Toolkit 12.6+ | NVIDIA installer | Enable VS integration |
| libtorch (C++, CUDA) | Manual download | `external/libtorch/`, `/MD` linking |
| Eigen3 | vcpkg | Header-only |
| spdlog | vcpkg | Logging |
| Google Test | vcpkg | Testing |
| nlohmann/json | vcpkg | Config files |
| GLFW | vcpkg | Viewer window |
| Dear ImGui | vcpkg or FetchContent | Viewer UI |
| stb_image | FetchContent or vendored | Single header |

---

## File Naming Convention

- Headers: `.hpp` (C++), `.cuh` (CUDA device code)
- Source: `.cpp` (C++), `.cu` (CUDA)
- Kernel functions: prefix `k_` (e.g., `k_rasterize_forward`)
- Test files: `test_<module>.cpp`
