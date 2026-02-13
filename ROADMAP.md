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

- [x] `src/training/loss.hpp` — public API: `l1_loss`, `ssim`, `ssim_loss`, `combined_loss`
  - All functions accept `[H, W, 3]` float32 CUDA tensors (matching `RenderOutput::color` layout)
  - `ssim()` returns per-pixel map `[H, W]` for reuse in Phase 9 evaluation metrics
- [x] `src/training/loss.cpp` — implementation using libtorch ops (`.cpp`, not `.cu`)
  - **L1 loss**: `(rendered - target).abs().mean()`
  - **SSIM** (Wang et al. 2004): Gaussian-weighted 11×11 window (σ=1.5), `conv2d` with `groups=3` dispatches to cuDNN on CUDA tensors
  - **Combined loss**: `(1 - λ) * L1 + λ * (1 - mean SSIM)`
  - Gaussian kernel cached as static to avoid recreating per call
- [x] Verify SSIM against known properties: identical images → 1.0, very different images → low, symmetry
- [x] CMake: added `cugs_training` library target, linked to `cugs_utils`

### Implementation Notes

- Used `.cpp` not `.cu` — libtorch's `conv2d` handles GPU dispatch automatically, no custom kernels needed
- `l1_loss` calls require `::cugs::` qualification to avoid ADL ambiguity with `at::l1_loss` from libtorch
- Input validation via `TORCH_CHECK()` for shape, dtype, device

### Tests

- [x] `tests/test_loss.cpp` — 10 tests:
  - L1: identical=0, known difference=0.5, non-negative
  - SSIM: identical≈1.0, different<0.1, symmetry, range [-1,1]
  - Combined: identical≈0, closer images → lower loss
  - Input validation: mismatched shapes, wrong channels, CPU tensors, wrong dtype, even window_size

### Definition of Done

Loss computation works and matches expected values on test cases.

---

## Phase 5: Backward Rasterizer (CUDA)

**Goal**: Compute gradients of the rendering loss with respect to all Gaussian parameters.

### Background

This is the second-hardest part after the forward pass. The backward pass traverses the same tile structure in reverse, computing gradients via the chain rule. For each pixel, gradients flow from `dL/dColor` back to each contributing Gaussian's parameters.

### Data Flow

```
loss.backward()  -->  dL/dColor [H, W, 3]
       |
  k_rasterize_backward  (per-tile, back-to-front compositing in reverse)
       |
       +--> dL/d(rgb) [N,3], dL/d(opacity_act) [N], dL/d(means_2d) [N,2], dL/d(cov_2d_inv) [N,3]
       |
  k_project_backward  (one thread per Gaussian, chain rule through projection)
       |
       +--> dL/d(positions) [N,3], dL/d(rotations) [N,4], dL/d(scales) [N,3], dL/d(opacities) [N,1]
       |
  k_evaluate_sh_backward  (one thread per Gaussian, linear in coefficients)
       |
       +--> dL/d(sh_coeffs) [N,3,C]
```

### Tasks

- [x] `src/core/sh_backward.hpp/.cu` — SH backward kernel
  - One thread per Gaussian, computes `dL/d(c_k) = dL/d(color) * Y_k(dir) * gate`
  - ReLU gate: zero gradient when forward SH output was clamped to 0
- [x] `src/rasterizer/backward.cuh` — five `__device__` helper functions:
  1. `compute_dL_dcov2d_from_dL_dcov2d_inv` — matrix inverse derivative for symmetric 2×2
  2. `compute_dL_dcov3d` — propagates via `T^T @ dL @ T`
  3. `compute_dL_dM` — propagates via `2 * dΣ_full @ M`
  4. `compute_dL_dquat` — explicit quaternion-rotation Jacobian with normalization
  5. `compute_dL_dt_cam_from_cov` — gradient through Jacobian J's t_cam dependence
- [x] `src/rasterizer/backward.hpp/.cu` — rasterize backward kernel
  - `k_rasterize_backward`: same tile grid as forward, back-to-front traversal
  - Reconstructs transmittance via `T /= max(1 - alpha, 1e-5)` going backwards
  - Uses `S_after` accumulator for tracking contributions after current Gaussian
  - Scatters gradients to per-Gaussian accumulators via `atomicAdd`
- [x] `src/rasterizer/projection_backward.hpp/.cu` — projection backward kernel
  - `k_project_backward`: one thread per Gaussian, recomputes all forward intermediates
  - Chains: `dL/d(cov2d_inv) → cov2d → cov3d → M → R/scale → quat/log_scale`
  - Position gradient through both means_2d path and covariance J-dependence path
  - Sigmoid derivative for `opacity_act → logit` chain
  - Host launcher calls `evaluate_sh_backward_cuda()` for SH gradients
- [x] `src/rasterizer/rasterizer.hpp/.cpp` — added `BackwardOutput` struct and `render_backward()` API
- [x] CMake: added `backward.cu`, `projection_backward.cu` to `cugs_rasterizer`; `sh_backward.cu` to `cugs_utils`

### Key Design Decisions

1. **Transmittance recomputation** (not storage): back-to-front reconstruction from `final_T` by dividing out `(1-alpha)`, clamped to `max(1-alpha, 1e-5)` for stability
2. **Recomputation in projection backward**: recompute all intermediates per-Gaussian rather than storing them — saves ~176 bytes/Gaussian (significant at 1M Gaussians on 6GB VRAM)
3. **No SH direction-through-position gradients**: direction treated as constant in backward (matches reference impl)
4. **Direct `atomicAdd(float*)`**: hardware-accelerated on sm_86, no warp-level reduction (defer to Phase 12)
5. **Clamp handling**: zero gradient when forward alpha was clamped to 0.99, below 1/255 (skipped), or SH was negative (clamped to 0)

### Issues Encountered

See `docs/issues.md` for detailed descriptions and solutions. Key issues:
- Symmetric 2×2 matrix off-diagonal gradient convention (combined vs single-entry format)
- Backward contributor count logic (tracking actual vs all Gaussians in range)
- Tile boundary discontinuities in position gradient finite-difference tests

### Tests

- [x] `tests/test_backward.cpp` — 9 tests:
  - OutputShapes, NoNanInf, CulledGaussiansZeroGrad — structural correctness
  - SingleGaussianConvergence — GD step reduces loss
  - FiniteDiff{Positions, Scales, Rotations, Opacities, SHCoeffs} — analytic vs numerical gradients
  - Uses mixed tolerance (absolute + relative) for robust finite-difference verification

### Definition of Done

Analytic gradients match finite differences to reasonable precision. A simple optimization test converges. All 9 tests pass.

---

## Phase 6: Basic Training Loop

**Goal**: Train Gaussians on a real scene — get first converging results.

### Background

Training loop: for each iteration, randomly sample a training image, render from that camera, compute loss, backpropagate, update parameters. The original paper trains for 30,000 iterations.

### Tasks

- [x] `src/training/lr_schedule.hpp` — learning rate schedules (header-only)
  - Position: exponential decay from 1.6e-4 to 1.6e-6 (log-linear interpolation)
  - SH: constant 2.5e-3, higher bands start at iteration 1000 (progressive activation)
  - Opacity: constant 0.05
  - Scale: constant 5e-3
  - Rotation: constant 1e-3
  - `ParamGroup` enum, `PositionLRConfig` struct, `position_lr()`, `active_sh_degree_for_step()`
- [x] `src/optimizer/adam.hpp/.cpp` — Adam optimizer using libtorch
  - `GaussianAdam` wraps `torch::optim::Adam` with 5 parameter groups
  - `apply_gradients(BackwardOutput)` bridges custom CUDA backward → libtorch optimizer
  - `update_lr(step)` applies exponential decay schedule to position group
  - Per-group learning rates with paper-default eps=1e-15
  - This gets replaced by fused CUDA Adam in Phase 8, but we need a working baseline first
- [x] `src/training/trainer.hpp/.cpp` — main training loop
  - `image_to_tensor()` converts CPU Image → CUDA tensor, handles RGBA→RGB
  - `Trainer` class: loads Dataset, initializes Gaussians, creates optimizer
  - `train_step()`: sample image → render → autograd dL/dcolor → custom backward → inject gradients → Adam step
  - Periodic logging (loss, L1, SSIM, num_gaussians, SH degree, position LR, it/s)
  - VRAM monitoring via `cudaMemGetInfo`
  - Checkpoint saving as PLY files
  - `max_gaussians` cap for VRAM-constrained GPUs
- [x] `apps/train_main.cpp` — CLI entry point
  - Full argument parsing: -d/--data, -o/--output, -i/--iterations, -r/--resolution, --sh-degree, --max-gaussians, --save-every, --log-every, --lambda, --random-bg, --seed
  - CUDA validation, config validation
- [x] **Progressive SH**: `active_sh_degree_for_step(step, max_degree)` — degree 0 at start, increases every 1000 iterations up to max_degree
- [x] CMake: `cugs_training` now links `cugs_data` and `cugs_rasterizer`; `train` executable and `test_training` test added

### Key Design Decisions

1. **Hybrid gradient flow**: render() (custom CUDA) → autograd dL/dcolor (libtorch) → render_backward() (custom CUDA) → inject .grad() → Adam step (libtorch). This avoids reimplementing Adam while keeping rasterizer performance.
2. **No SH coefficient masking**: progressive SH controls `active_sh_degree` in RenderSettings but doesn't zero out higher-degree coefficients. The forward rasterizer already respects the active degree.
3. **Lazy image loading**: Dataset loads images on-demand each iteration to minimize peak memory.
4. **Image-camera resolution reconciliation**: `train_step()` resizes the loaded image to match camera dimensions if they differ. This handles the common case where COLMAP's `cameras.bin` records the original capture resolution but the actual image files on disk are already downscaled.

### Tests

- [x] `tests/test_training.cpp` — 12 tests:
  - LR schedule: initial/final/beyond/monotonic/midpoint, active SH degree clamping
  - Image-to-tensor: RGB and RGBA conversion, pixel value verification
  - Adam: per-group LRs, position LR decay over training
  - Convergence: 20 Gaussians, perturb SH, 100 Adam iterations → loss decreases >10%

### First Training Run (Truck scene, RTX 3060 Laptop GPU)

```
build\train -d data\tandt\truck -o output\truck -i 1000 -r 4 --max-gaussians 50000
```

| Metric | Value |
|---|---|
| Scene | Tanks & Temples Truck (219 train / 32 test views) |
| Resolution | 489x272 (1/4 scale) |
| Gaussians | 50,000 (capped from 136k sparse points) |
| VRAM | ~1 GB / 6 GB |
| Speed | 0.4 it/s (2585s total) |
| Loss (step 0) | 0.285 (L1=0.205, SSIM=0.39) |
| Loss (step 400) | 0.125 (L1=0.086, SSIM=0.72) — best observed |
| Loss (step 999) | 0.295 (L1=0.230, SSIM=0.44) — per-view variance |

**Observations**:
- Loss clearly decreases overall — gradient flow is correct end-to-end
- Per-iteration loss fluctuates because each step samples a different training view
- 0.4 it/s is slow; bottlenecks: libtorch Adam overhead, per-iteration disk I/O, no image caching. Fused CUDA Adam (Phase 8) and image caching will help.
- SH degree stays at 0 (progressive activation starts degree 1 at step 1000)
- Gaussian init takes ~4 min for 136k points due to O(n²) kNN — optimization target for later
- Checkpoint PLY has correct structure but no vertex colors — SH coefficients stored as `f_dc_*` properties, need a Gaussian splatting viewer (Phase 11) to visualize correctly. Standard PLY viewers show uncolored points.
- Scene is sparse without densification — quality improves dramatically in Phase 7

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

- [x] `src/optimizer/densification.hpp` — `DensificationConfig`, `DensificationStats`, `DensificationController` class
  - `DensificationConfig`: schedule params (from/until/every), thresholds (grad, opacity, percent_dense, max_screen_size), capacity (max_gaussians, min_vram_headroom_mb)
  - `DensificationController`: accumulates per-Gaussian gradient norms, runs clone/split/prune cycle
- [x] `src/optimizer/densification.cpp` — full implementation using libtorch tensor ops
  - **Clone**: duplicate small Gaussians (`max(exp(scale)) < percent_dense * scene_extent`) with high avg gradient
  - **Split**: replace large Gaussians with 2 children (scale reduced by `log(1.6)`, positions jittered by `randn * exp(new_scale)`)
  - **Prune**: remove Gaussians with `sigmoid(opacity) < threshold` or `screen_radius > max_screen_size`; remove originals that were split
  - **Opacity reset**: set all opacities to `inverse_sigmoid(0.01) ≈ -4.595` every 3000 steps
  - **VRAM guard**: skip densification if `cudaMemGetInfo` reports free VRAM below headroom
  - Budget-limited cloning/splitting: respects `max_gaussians` cap, selects highest-gradient candidates when budget is tight
  - Lazy accumulator init/reset after densification handles changing N gracefully
- [x] `src/training/trainer.hpp` — added `DensificationConfig densification` and `bool no_densify` to `TrainConfig`; added densification stats to `IterationStats`; added `DensificationController` member to `Trainer`
- [x] `src/training/trainer.cpp` — integrated densification hooks into `train_step()`:
  - Accumulate gradients every iteration after backward pass
  - Run densify on schedule; rebuild `GaussianAdam` when model size changes (Adam moments invalid)
  - Run opacity reset on schedule; log densification events
- [x] `apps/train_main.cpp` — added CLI flags: `--densify-from`, `--densify-until`, `--densify-every`, `--grad-threshold`, `--no-densify`
- [x] CMake: added `densification.cpp` to `cugs_training`, added `test_densification` test target
- [x] Tune thresholds for 6GB VRAM:
  - Implemented five-part memory safety system (see `docs/issues.md` Issue 14):
    1. Non-throwing `VramInfo` query in `cuda_utils.cuh`
    2. `MemoryLimitConfig` + `memory_monitor.hpp` with auto/manual VRAM limit, RAM monitoring
    3. Per-iteration VRAM safety check with critical-streak abort
    4. Budget-aware clone/split in densification (reduces count when VRAM tight)
    5. `emptyCache()` after densification to reclaim freed memory
  - `--vram-limit <MB>` CLI flag for manual override
  - Default: auto-reserves 600 MB for display driver on WDDM GPUs

### Memory Safety Test Runs (Truck scene, RTX 3060 6GB Laptop GPU)

```
build\train -d data\tandt\truck -o output\truck_memsafe -i 1000 -r 4 --max-gaussians 50000 --log-every 10
```

| Test | VRAM Limit | Result |
|---|---|---|
| `--vram-limit 4000` | 4000 MB (user-set) | Trained to step 68, graceful abort after 5 consecutive critical readings (budget: -52 MB). Checkpoint saved. No freeze. |
| Auto mode (no flag) | 5544 MB (auto: 6144 - 600) | Trained to step 385 before VRAM crept to -210 MB budget. CUDA caching allocator gradually consumed memory. Streak began, run interrupted by user. |

**Startup memory**: 1046 / 6144 MB VRAM, ~49 GB / 65 GB RAM available.

**Observations**:
- 50k Gaussians at SH degree 3 with Adam state is near the 6 GB limit even without densification
- The CUDA caching allocator accumulates fragmented memory over iterations, causing steady VRAM growth
- `emptyCache()` reclaims memory after densification but cannot address per-iteration allocator growth
- Recommended 6 GB settings: `--max-gaussians 30000` or `--sh-degree 2` with auto VRAM limit
- Gaussian init still takes ~4 min for 136k points (O(n²) kNN) before capping to 50k

### Key Design Decisions

1. **Optimizer reconstruction after densification**: destroy and rebuild `GaussianAdam` when N changes. Adam moments become invalid after tensor resize. Momentum rebuilds within ~100 iterations.
2. **No custom CUDA kernels**: all densification logic uses libtorch tensor ops (`cat`, `index`, `norm`, `sigmoid`). Fast enough since densification runs only every 100 iterations.
3. **Split removes originals via prune mask**: split appends 2 children, then the prune step removes the original. No complex index bookkeeping.
4. **Gradient accumulation only for visible Gaussians**: `radii > 0` mask prevents invisible Gaussians from accumulating zero gradients, which would dilute their average.
5. **Lazy accumulator initialization**: `grad_accum_` created on first `accumulate_gradients()` call and re-created after densification. Handles N changes gracefully.

### MCMC Densification (Kheradmand et al.)

- [x] `src/optimizer/mcmc_densification.hpp/.cpp` — implemented in Phase 10
  - Alternative to clone/split/prune with fixed Gaussian count
  - Relocate low-opacity Gaussians to high-opacity regions
  - Per-iteration noise injection for exploration
  - Opacity/scale regularization
  - **Ideal for 6GB VRAM** since N stays constant throughout training

### Tests

- [x] `tests/test_densification.cpp` — 10 tests:
  - ShouldDensifyBoundaries: schedule start/end/frequency/exclusions
  - ShouldResetOpacity: opacity reset schedule (3000, 6000, etc.)
  - AccumulatesGradients: gradient accumulation doesn't crash, densify runs
  - InvisibleGaussiansNotAccumulated: zero-radius Gaussians produce no clones/splits
  - HighGradSmallScaleGetsCloned: clone path triggers, N increases, model valid
  - HighGradLargeScaleGetsSplit: split path triggers, model valid
  - LowOpacityGetsPruned: 5 low-opacity removed, 5 high-opacity kept
  - OpacityResetSetsLowValue: all opacities set to -4.595
  - ModelRemainsValidAfterFullCycle: mixed clone/split/prune, model.is_valid()
  - MaxGaussiansRespected: hard cap enforced during densification

### Definition of Done

Training with densification produces noticeably sharper results than without. Gaussian count grows and stabilizes.

---

## Phase 8: Fused Adam Optimizer (CUDA)

**Goal**: Replace libtorch Adam with a single fused CUDA kernel for 2x+ speedup.

### Background

The standard Adam update requires multiple kernel launches per parameter: compute gradient, update first moment, update second moment, compute bias correction, apply update. A fused kernel does all of this in a single launch, eliminating kernel launch overhead and improving memory locality.

### Tasks

- [x] `src/optimizer/fused_adam.hpp` — FusedAdam class declaration with identical API to GaussianAdam
  - Self-managed Adam state: m_[5], v_[5] (moment tensors), grads_[5] (gradient refs), learning_rates_[5], step_count_
  - No dependency on torch::optim::Adam
- [x] `src/optimizer/fused_adam.cu` — k_fused_adam kernel + class implementation
  - Single kernel per parameter group: reads gradient, updates m/v in-place, writes bias-corrected param update
  - 256 threads/block, one thread per float element
  - Bias correction factors pre-computed on host in double precision for early-step accuracy
  - launch_kernel() private helper: validates tensors, ensures contiguity, computes grid dims, CUDA_CHECK()
- [x] Swapped trainer to use FusedAdam (trainer.hpp + trainer.cpp)
- [x] Ensure training results are numerically equivalent (within floating point tolerance)
- [ ] Benchmark against libtorch Adam — expect 1.5-2.5x speedup on the optimizer step

### Tests

- `tests/test_fused_adam.cpp` — compare one step of fused Adam vs libtorch Adam, verify numerical equivalence

### Definition of Done

Fused Adam produces identical training results with measurable speedup.

---

## Phase 9: Evaluation & Quality Metrics

**Goal**: Compute PSNR, SSIM, and (optionally) LPIPS on test views. Compare to paper results.

### Tasks

- [x] `src/training/metrics.hpp/.cpp`
  - PSNR: `10 * log10(1.0 / MSE)`, clamped to 100 dB for identical images
  - SSIM: reuses `cugs::ssim()` from `loss.hpp`, returns `.mean().item<float>()`
  - LPIPS: skipped — requires pretrained VGG network, adds complexity. PSNR + SSIM are the primary metrics in the paper's comparisons.
  - `EvalResults` struct with `to_json()` / `save_json()` using nlohmann/json
  - `evaluate()` function: iterates test cameras, renders with `NoGradGuard`, computes per-image and mean metrics
- [x] Evaluation CLI in `apps/eval_main.cpp`:
  - Load trained model via `GaussianModel::load_ply()`
  - Load dataset, configure `RenderSettings` (SH degree from model or CLI, background color)
  - Render all test views, compute PSNR/SSIM per image
  - Print summary table to console
  - Save results as JSON
  - CLI: `-m/--model`, `-d/--data`, `-o/--output`, `-r/--resolution`, `--sh-degree`, `--background`
- [x] `tests/test_metrics.cpp` — 8 tests:
  - PSNRIdentical, PSNRKnownDifference, PSNRSymmetric, PSNRRange
  - SSIMIdentical, SSIMDifferent, SSIMSymmetric
  - EvalResultsJsonRoundtrip
- [x] CMake: added `metrics.cpp` to `cugs_training`, `eval` executable, `test_metrics` test target
- [ ] Run on Truck and/or Train datasets, compare to paper Table 1:
  - Truck: PSNR ~25.2, SSIM ~0.88 (Mip-NeRF 360 dataset)
  - Don't expect exact match on 6GB GPU if Gaussian count is limited

### Key Design Decisions

1. **No LPIPS**: requires a pretrained VGG network and libtorch model loading, which adds complexity. PSNR + SSIM are sufficient for initial comparisons against the paper.
2. **Reuses existing SSIM**: `compute_ssim()` wraps `cugs::ssim()` from `loss.hpp` — no code duplication.
3. **Same image resize pattern as trainer**: handles COLMAP camera/image resolution mismatch.
4. **NoGradGuard**: all evaluation runs without autograd for speed and memory efficiency.

### Definition of Done

Can evaluate trained models and produce quality metrics. Results are in the right ballpark.

---

## Phase 10: MCMC Densification

**Goal**: Implement the MCMC-based densification strategy from Kheradmand et al. (NeurIPS 2024) as an alternative to clone/split/prune.

### Background

Instead of clone/split/prune (which grows/shrinks N), MCMC densification maintains a **fixed Gaussian count** by relocating "dead" (low-opacity) Gaussians to high-opacity regions. Per-iteration noise injection guides exploration, and opacity/scale regularization prevents degenerate solutions. Key advantage: constant VRAM footprint — ideal for the 6GB constraint.

### Tasks

- [x] `src/optimizer/mcmc_densification.hpp` — `MCMCConfig`, `MCMCStats`, `MCMCController` class
  - `MCMCConfig`: schedule (relocate_from/until/every), relocation (dead_opacity_threshold=0.005, relocate_cap=0.05), noise (noise_lr_init=5e5, noise_lr_final=1e3, noise_gate_k=100, noise_gate_t=0.995), regularization (lambda_opacity=0.01, lambda_scale=0.01), VRAM fields
  - `MCMCController`: `should_relocate()`, `relocate()`, `inject_noise()`, `compute_regularization()`, `noise_lr()`
- [x] `src/optimizer/mcmc_densification.cpp` — controller implementation using libtorch tensor ops
  - **Relocation**: `dead_mask = sigmoid(opa) < threshold` → sample alive sources via `multinomial(opacity_weights)` → copy SH/rotation from source, jitter position by `randn * scene_extent * 0.01`, set small scale + low opacity. N stays constant.
  - **Noise injection** (under `NoGradGuard`): `gate = sigmoid(-k * (sigmoid(opa) - t))`, `noise = noise_lr * exp(scales) * gate * randn_like(positions)`, `positions += noise`. Low-opacity Gaussians get more noise.
  - **Regularization**: `lambda_o * mean(sigmoid(opa)) + lambda_s * mean(exp(scales))` computed on autograd-tracked clones; gradients injected into `BackwardOutput`.
  - **Noise LR**: log-linear decay from init to final (same formula as `position_lr()`).
- [x] `src/training/trainer.hpp` — added `bool mcmc_densify`, `MCMCConfig mcmc`, `MCMCController` ptr, MCMC stats to `IterationStats`
- [x] `src/training/trainer.cpp` — MCMC integration in constructor + `train_step()`:
  - Step 7b: regularization gradient injection after `render_backward`, before `apply_gradients`
  - Step 9: MCMC branch — noise injection every iteration, relocation on schedule. No gradient accumulation, no opacity reset, no optimizer rebuild (N constant).
- [x] `apps/train_main.cpp` — CLI flags: `--mcmc`, `--mcmc-relocate-every`, `--mcmc-lambda-opacity`, `--mcmc-lambda-scale`, `--mcmc-noise-lr`. Validated `--mcmc` + `--no-densify` mutual exclusion.
- [x] CMake: added `mcmc_densification.cpp` to `cugs_training`, `test_mcmc` to tests
- [ ] Compare MCMC vs ADC on Truck scene:
  - Memory usage over training (should be constant for MCMC)
  - Final quality metrics (PSNR, SSIM)
  - Training time

### Key Design Decisions

1. **No abstract base class**: MCMCController and DensificationController have different APIs (MCMC needs `inject_noise()` every iter, `compute_regularization()`). Two separate `unique_ptr`s in Trainer, switched by `mcmc_densify` flag.
2. **No optimizer rebuild on relocate**: N stays constant, so Adam moments remain valid. Stale values for relocated indices adapt within ~10 steps via exponential averaging.
3. **Regularization via autograd on cloned tensors**: Creates detached copies with `requires_grad_(true)`, computes reg loss, calls `.backward()`, extracts `.grad()` and adds to `BackwardOutput`. Clean injection without modifying the custom CUDA backward pass.
4. **All libtorch tensor ops**: No custom CUDA kernels for MCMC — relocation runs every ~100 iterations, noise injection is a few elementwise ops per iteration.

### Tests

- [x] `tests/test_mcmc.cpp` — 11 tests:
  - ShouldRelocateBoundaries: schedule start/end/frequency/exclusions
  - NoiseLRInitAndFinal: init value at step 0, final at/beyond max_steps
  - NoiseLRMonotonicDecay: strictly decreasing over training
  - RelocationFixesDeadGaussians: dead positions change, alive untouched, N constant
  - RelocateCapRespected: num_relocated <= cap * N
  - RelocationWithNoDeadIsNoop: all-alive model unchanged
  - NoiseGateSelectivity: low-opacity Gaussians displaced >2x more than high-opacity
  - NoiseInjectionModifiesPositions: positions change, model valid
  - RegularizationIsPositiveScalar: positive value, correct gradient shapes, non-zero grads
  - ModelValidAfterFullCycle: noise + regularization + relocation → valid model, constant N
  - ConstantNAcrossMultipleRelocations: 5 relocation steps, N unchanged each time

### Definition of Done

MCMC densification works and produces comparable quality with constant memory usage.

---

## Phase 11: Real-Time Viewer ✅

**Goal**: Interactive OpenGL viewer for trained models.

### Architecture

Reuses the existing CUDA rasterizer (`cugs::render()`) for GPU-accelerated splatting, then transfers the resulting `[H,W,3]` float tensor to CPU and uploads to an OpenGL texture for display as a fullscreen quad. This guarantees identical visual output to training.

### Tasks

- [x] `src/viewer/camera_controller.hpp` — orbit camera controller
  - Spherical coordinate system (azimuth, elevation, radius) around target point
  - `rotate()`, `pan()`, `zoom()` methods for mouse input
  - `build_camera()` produces `CameraInfo` with world-to-camera extrinsics + pinhole intrinsics
  - Auto-initializes from Gaussian bounding box, gimbal lock clamping
- [x] `src/viewer/viewer.hpp` — viewer class declaration
  - `ViewerConfig` struct (width, height, background, sh_degree, vsync)
  - `RenderMode` enum (RGB, Depth, Heatmap)
- [x] `src/viewer/viewer.cpp` — viewer implementation
  - GLFW window + OpenGL 2.1 context (no GLAD/GLEW — manual GL extension loading via `glfwGetProcAddress`)
  - GLSL 120 fullscreen quad shader for texture display
  - CUDA rasterizer integration: `render()` → CPU transfer → `glTexSubImage2D`
  - Three render modes: RGB, depth (1-final_T with turbo colormap), heatmap (n_contrib normalized)
  - ImGui overlay: FPS, Gaussian count, VRAM usage, render mode selector, SH degree slider, FOV control
  - Mouse input: left-drag=orbit, middle/right-drag=pan, scroll=zoom
  - Keyboard: 1/2/3 = render modes, Esc = quit
  - `#define NOMINMAX` before all includes to avoid Windows min/max macro conflicts
- [x] `apps/viewer_main.cpp` — CLI entry point
  - `viewer <model.ply> [--width N] [--height N] [--background black|white] [--sh-degree 0-3] [--no-vsync]`
  - CUDA validation, model loading, viewer launch
- [x] CMake integration
  - `cugs_viewer` library linking `cugs_rasterizer`, `cugs_data`, `glfw`, `imgui::imgui`, `OpenGL::GL`
  - `viewer` executable in `apps/CMakeLists.txt`
  - `find_package(imgui CONFIG REQUIRED)` + `find_package(OpenGL REQUIRED)` added to root CMakeLists.txt

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
