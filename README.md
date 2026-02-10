# cuda-gaussian-splatting

A from-scratch implementation of [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) in C++23 and CUDA, with a custom differentiable rasterizer, fused Adam optimizer, and real-time interactive viewer.

## Motivation

The original 3D Gaussian Splatting paper achieves remarkable real-time rendering quality, but the reference implementation relies heavily on Python and PyTorch. This project reimplements the entire pipeline — training, optimization, and rendering — in pure C++/CUDA to understand the algorithm at the lowest level: the math behind EWA splatting, the CUDA kernels for differentiable rasterization, and the optimization dynamics that make it converge.

## What's Implemented

**Differentiable Rasterizer (CUDA)**
- Forward pass: 3D Gaussian projection, 2D covariance via Jacobian approximation, tile-based sorting, alpha compositing
- Backward pass: analytic gradients w.r.t. all Gaussian parameters (position, covariance, opacity, spherical harmonics)
- Tile-based parallel rasterization for scalability

**Training Pipeline**
- COLMAP sparse reconstruction loader (cameras, images, points3D)
- Loss: L1 + SSIM (structural similarity)
- Fused Adam optimizer as a single CUDA kernel (avoids multiple libtorch kernel launches)
- Adaptive density control: clone, split, prune based on gradient accumulation
- MCMC densification strategy (Kheradmand et al., NeurIPS 2024)

**Real-Time Viewer**
- OpenGL-based interactive viewer with camera controls
- Multiple render modes: RGB, depth, combined
- Load and visualize trained .ply models

## The Math

The core of Gaussian Splatting involves several interconnected mathematical components:

- **3D → 2D projection**: each 3D Gaussian with covariance Σ is projected to a 2D Gaussian with covariance Σ' = J W Σ Wᵀ Jᵀ, where W is the view transform and J is the Jacobian of the projective mapping
- **Spherical Harmonics**: up to degree 3 (16 coefficients per channel) encode view-dependent color
- **Alpha compositing**: front-to-back blending with transmittance tracking, C = Σᵢ cᵢ αᵢ Πⱼ₌₁ⁱ⁻¹ (1 - αⱼ)
- **Covariance parameterization**: rotation (quaternion) + scale → 3D covariance via Σ = R S Sᵀ Rᵀ

All gradient computations for the backward pass are derived analytically and implemented as custom CUDA kernels.

## Building

### Prerequisites
- NVIDIA GPU with compute capability 7.5+ (RTX 2080 or newer)
- CUDA Toolkit 12.0+
- C++23 compiler (GCC 12+, Clang 15+)
- CMake 3.24+

### Build
```bash
# Clone
git clone https://github.com/<username>/cuda-gaussian-splatting.git
cd cuda-gaussian-splatting

# Download libtorch (CUDA-enabled C++ distribution)
wget https://download.pytorch.org/libtorch/cu128/libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.7.0+cu128.zip -d external/

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build -- -j$(nproc)
```

### Train
```bash
# Prepare data: run COLMAP on your images, or download a standard dataset
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip tandt_db.zip -d data/

# Train
./build/train -d data/garden -o output/garden -i 30000 --eval
```

### View
```bash
./build/viewer output/garden/point_cloud.ply
```

## Project Structure

```
src/
  core/        — Gaussian data structures, SH coefficients
  data/        — COLMAP loader, camera models, image I/O
  rasterizer/  — CUDA forward/backward kernels
  optimizer/   — Fused Adam (CUDA), densification strategies
  training/    — Training loop, loss, LR scheduling
  viewer/      — OpenGL real-time viewer
  utils/       — Timer, logging, PLY I/O, math helpers
tests/         — Unit tests (Google Test)
benchmarks/    — Performance benchmarks
```

## Dependencies

| Library | Purpose |
|---|---|
| libtorch | Tensor operations, autograd for non-kernel paths |
| Eigen3 | CPU-side linear algebra, camera math |
| GLFW + OpenGL + Dear ImGui | Interactive viewer |
| stb_image | Image I/O |
| nlohmann/json | Configuration |
| Google Test | Unit tests |
| spdlog | Logging |

## References

1. Kerbl et al., [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), SIGGRAPH 2023
2. Kheradmand et al., [3D Gaussian Splatting as Markov Chain Monte Carlo](https://arxiv.org/abs/2404.09591), NeurIPS 2024
3. Ye et al., [gsplat: An Open-Source Library for Gaussian Splatting](https://arxiv.org/abs/2409.06765), 2024
4. Zwicker et al., [EWA Splatting](https://www.cs.umd.edu/~zwicker/publications/EWASplatting-TVCG02.pdf), IEEE TVCG 2002

## License

MIT
