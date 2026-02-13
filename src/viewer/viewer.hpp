#pragma once

/// @file viewer.hpp
/// @brief Real-time Gaussian splatting viewer using CUDA rasterizer + OpenGL display.
///
/// Loads a trained .ply model, renders it with the CUDA rasterizer, and displays
/// the result as a fullscreen textured quad via OpenGL. Supports interactive orbit
/// camera controls and an ImGui debug overlay.

#include "core/gaussian.hpp"
#include "rasterizer/rasterizer.hpp"
#include "viewer/camera_controller.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <string>

// Forward declarations to avoid pulling GL/GLFW into every TU
struct GLFWwindow;

namespace cugs {

/// @brief Render mode for the viewer display.
enum class RenderMode : int {
    kRGB = 0,      ///< Standard RGB color output
    kDepth = 1,    ///< Depth visualization (1 - final_T as opacity proxy)
    kHeatmap = 2,  ///< Per-pixel Gaussian contribution count heatmap
};

/// @brief Configuration for the viewer window and rendering.
struct ViewerConfig {
    int width = 1280;
    int height = 720;
    float background[3] = {0.0f, 0.0f, 0.0f};
    int sh_degree = -1;  ///< -1 = use model's max SH degree
    bool vsync = true;
};

/// @brief Real-time interactive Gaussian splatting viewer.
///
/// Loads a PLY model, creates a GLFW window with OpenGL context, and renders
/// the Gaussians using the existing CUDA rasterizer. The rendered image is
/// transferred to an OpenGL texture for display as a fullscreen quad.
class Viewer {
public:
    /// @brief Construct a viewer for the given PLY model.
    /// @param ply_path Path to the trained Gaussian model (.ply).
    /// @param config   Viewer configuration (window size, background, etc.).
    Viewer(std::filesystem::path ply_path, ViewerConfig config);
    ~Viewer();

    // Non-copyable, non-movable
    Viewer(const Viewer&) = delete;
    Viewer& operator=(const Viewer&) = delete;

    /// @brief Run the viewer main loop (blocking).
    void run();

private:
    void init_window();
    void init_gl_resources();
    void init_imgui();
    void cleanup();
    void process_input();
    void check_dirty();
    void render_frame();
    void draw_scene();
    void draw_imgui();

    /// @brief Upload a [H,W,4] float CPU tensor to the GL texture.
    void upload_texture(const torch::Tensor& image);

    /// @brief Upload a [H,W,4] float GPU tensor via CUDA-GL interop PBO.
    void upload_texture_interop(const torch::Tensor& rgba_gpu);

    /// @brief Initialize PBO and register with CUDA for GL interop.
    void init_cuda_gl_interop();

    /// @brief Resize PBO to match new dimensions.
    void resize_pbo(int w, int h);

    /// @brief Release CUDA-GL interop resources.
    void cleanup_cuda_gl_interop();

    /// @brief Apply a colormap to a single-channel [H,W] tensor.
    /// @return [H,W,3] float tensor with colormap applied.
    static torch::Tensor apply_turbo_colormap(const torch::Tensor& values);
    static torch::Tensor apply_heat_colormap(const torch::Tensor& values);

    // --- State ---
    std::filesystem::path ply_path_;
    ViewerConfig config_;
    GaussianModel model_;
    CameraController camera_;
    RenderSettings render_settings_;
    RenderMode render_mode_ = RenderMode::kRGB;

    // Window/GL
    GLFWwindow* window_ = nullptr;
    unsigned int texture_id_ = 0;
    int texture_width_ = 0;   ///< Actual GL texture dimensions (may differ from framebuffer)
    int texture_height_ = 0;
    unsigned int vao_ = 0;
    unsigned int vbo_ = 0;
    unsigned int shader_program_ = 0;
    int framebuffer_width_ = 0;
    int framebuffer_height_ = 0;

    // CUDA-GL interop (PBO)
    unsigned int pbo_id_ = 0;
    size_t pbo_size_ = 0;
    cudaGraphicsResource_t cuda_pbo_resource_ = nullptr;
    bool interop_available_ = false;

    // Dirty tracking — skip rendering when nothing changed
    bool needs_render_ = true;
    uint64_t last_camera_version_ = 0;
    RenderSettings last_render_settings_{};
    RenderMode last_render_mode_ = RenderMode::kRGB;
    int last_render_width_ = 0;
    int last_render_height_ = 0;
    bool camera_moved_this_frame_ = false;
    bool last_was_interactive_ = false;

    // Frame timing
    double last_frame_time_ = 0.0;
    float frame_dt_ = 0.0f;
    float fps_ = 0.0f;
    int frame_count_ = 0;
    int render_width_ = 0;   ///< Actual resolution used for last render
    int render_height_ = 0;

    // Mouse state
    bool mouse_left_down_ = false;
    bool mouse_middle_down_ = false;
    bool mouse_right_down_ = false;
    double mouse_last_x_ = 0.0;
    double mouse_last_y_ = 0.0;
};

} // namespace cugs
