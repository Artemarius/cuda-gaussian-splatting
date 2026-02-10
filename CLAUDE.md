# CLAUDE.md

## Architecture

```
src/
  core/        — Gaussian data structures (means, quaternions, scales, opacity, SH coefficients)
  data/        — COLMAP dataset loader, camera models, image I/O
  rasterizer/  — CUDA forward/backward pass kernels for splatting
  optimizer/   — Fused Adam optimizer (CUDA), densification/pruning strategies (default + MCMC)
  training/    — Training loop, loss computation (L1 + SSIM), learning rate scheduling
  viewer/      — Real-time OpenGL viewer with interactive camera
  utils/       — Timer, logging, PLY I/O, math helpers
tests/         — Google Test unit tests
benchmarks/    — Performance benchmarks
```

## Technical Decisions

- **C++23 standard** — use modern features: std::expected, std::format, structured bindings, concepts, ranges where appropriate
- **CUDA 12+** — target compute capability 7.5+, use cooperative groups, modern CUDA idioms
- **libtorch C++ API** for autograd/tensor operations where it simplifies the training loop, but custom CUDA kernels for all performance-critical paths (rasterization forward/backward, SH evaluation, covariance computation)
- **CMake** build system with vcpkg or FetchContent for dependencies
- **No Python** — the entire pipeline is C++/CUDA
- **MIT License**

## Code Style & Conventions

- Google C++ Style Guide as baseline, with modifications:
  - `snake_case` for functions and variables, `PascalCase` for types/classes
  - RAII everywhere, no raw `new`/`delete`
  - Use `std::span`, `std::string_view` where appropriate
  - Prefer `constexpr` and `consteval` where possible
- CUDA kernels: prefix with `k_` (e.g., `k_rasterize_forward`), document grid/block dimensions
- All public APIs must have doxygen-style doc comments
- Every module must have unit tests

## Build & Run

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build -- -j$(nproc)
./build/train -d data/garden -o output/garden -i 30000
./build/viewer output/garden/point_cloud.ply
```

## Dependencies

- libtorch (C++ distribution, CUDA-enabled)
- Eigen3 (linear algebra)
- OpenGL + GLFW + Dear ImGui (viewer)
- stb_image (image I/O)
- nlohmann/json (config)
- Google Test (testing)
- spdlog (logging)

## Priorities

1. **Correctness** — match the original paper's quality metrics (PSNR, SSIM, LPIPS) on standard datasets (Mip-NeRF 360, Tanks and Temples)
2. **Readability** — code clarity matters as much as performance; favor explicit over clever
3. **Performance** — once correct, optimize CUDA kernels and benchmark against the reference implementation
4. **Documentation** — every non-trivial algorithm should reference the relevant paper/equation

## Key References

- Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering" (SIGGRAPH 2023)
- Kheradmand et al., "3D Gaussian Splatting as Markov Chain Monte Carlo" (NeurIPS 2024) — for MCMC densification
- Ye et al., "gsplat: An Open-Source Library for Gaussian Splatting" — for efficient CUDA rasterizer design
