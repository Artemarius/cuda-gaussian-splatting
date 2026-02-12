#pragma once

/// @file memory_monitor.hpp
/// @brief GPU and system memory monitoring utilities for safe training.
///
/// Provides VRAM budget tracking, system RAM queries, and memory-aware
/// limits to prevent CUDA OOM on WDDM GPUs (where OOM can freeze the
/// display driver and require a power cycle).

#include "utils/cuda_utils.cuh"

#include <spdlog/spdlog.h>

#include <cmath>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace cugs {

/// @brief Configuration for GPU/RAM memory safety limits.
struct MemoryLimitConfig {
    /// Hard VRAM limit in MB (0 = auto: total minus safety_margin).
    float vram_limit_mb = 0.0f;

    /// Reserved VRAM for display driver when auto-computing limit (MB).
    /// On WDDM GPUs (Windows), DWM needs VRAM for desktop compositing.
    float vram_safety_margin_mb = 600.0f;

    /// Critical VRAM threshold (MB free). Below this, increment streak counter.
    float vram_critical_mb = 200.0f;

    /// Consecutive critical checks before aborting training.
    int vram_critical_count = 5;

    /// Warn when system RAM drops below this (MB).
    float ram_warning_mb = 1024.0f;
};

/// @brief Compute the effective VRAM limit from config.
///
/// If `config.vram_limit_mb` is set (> 0), use it directly.
/// Otherwise, auto-compute as `total_vram - safety_margin`.
///
/// @param config Memory limit configuration.
/// @return Effective VRAM limit in MB, or 0 if VRAM query fails.
inline float compute_effective_vram_limit(const MemoryLimitConfig& config) {
    if (config.vram_limit_mb > 0.0f) {
        return config.vram_limit_mb;
    }
    auto info = vram_info_mb();
    if (!info.valid()) return 0.0f;
    return info.total_mb - config.vram_safety_margin_mb;
}

/// @brief Returns how much VRAM budget is available before hitting the limit.
///
/// @param effective_limit The effective VRAM limit (from compute_effective_vram_limit).
/// @return Available budget in MB, or -1 if VRAM query fails.
inline float vram_budget_available_mb(float effective_limit) {
    auto info = vram_info_mb();
    if (!info.valid()) return -1.0f;
    return effective_limit - info.used_mb();
}

/// @brief Returns available system RAM in MB.
inline float system_ram_available_mb() {
#ifdef _WIN32
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(mem_info);
    if (GlobalMemoryStatusEx(&mem_info)) {
        return static_cast<float>(mem_info.ullAvailPhys) / (1024.0f * 1024.0f);
    }
#endif
    return -1.0f;  // Unknown on non-Windows or on failure.
}

/// @brief Returns total system RAM in MB.
inline float system_ram_total_mb() {
#ifdef _WIN32
    MEMORYSTATUSEX mem_info;
    mem_info.dwLength = sizeof(mem_info);
    if (GlobalMemoryStatusEx(&mem_info)) {
        return static_cast<float>(mem_info.ullTotalPhys) / (1024.0f * 1024.0f);
    }
#endif
    return -1.0f;
}

/// @brief Log VRAM and system RAM status.
///
/// @param effective_limit The effective VRAM limit in MB.
inline void log_memory_status(float effective_limit) {
    auto info = vram_info_mb();
    if (info.valid()) {
        float budget = effective_limit - info.used_mb();
        spdlog::info("VRAM: {:.0f} / {:.0f} MB used | budget: {:.0f} MB | limit: {:.0f} MB",
                     info.used_mb(), info.total_mb, budget, effective_limit);
    }

    float ram = system_ram_available_mb();
    float ram_total = system_ram_total_mb();
    if (ram >= 0.0f) {
        spdlog::info("RAM:  {:.0f} / {:.0f} MB available",
                     ram, ram_total);
    }
}

/// @brief Estimate VRAM needed to densify N Gaussians at a given SH degree.
///
/// Each Gaussian requires storage for: positions (3), scales (3),
/// opacities (1), rotations (4), SH coefficients (3 * (deg+1)^2).
/// All stored as float32 (4 bytes each).
///
/// During densification, both old and new tensors coexist in memory
/// (torch::cat creates copies), so the estimate covers the new allocation.
///
/// @param num_gaussians Number of new Gaussians to be created.
/// @param sh_degree Current SH degree.
/// @return Estimated VRAM in MB.
inline float estimate_densification_vram_mb(int num_gaussians, int sh_degree) {
    int sh_coeffs = (sh_degree + 1) * (sh_degree + 1);
    // positions(3) + scales(3) + opacities(1) + rotations(4) + sh(3*C)
    int floats_per_gaussian = 3 + 3 + 1 + 4 + 3 * sh_coeffs;
    float bytes = static_cast<float>(num_gaussians) *
                  static_cast<float>(floats_per_gaussian) * 4.0f;
    return bytes / (1024.0f * 1024.0f);
}

} // namespace cugs
