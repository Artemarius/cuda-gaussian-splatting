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
    void render_frame();
    void draw_imgui();

    /// @brief Upload a [H,W,3] float tensor to the GL texture.
    void upload_texture(const torch::Tensor& image);

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
    unsigned int vao_ = 0;
    unsigned int vbo_ = 0;
    unsigned int shader_program_ = 0;
    int framebuffer_width_ = 0;
    int framebuffer_height_ = 0;

    // Frame timing
    double last_frame_time_ = 0.0;
    float frame_dt_ = 0.0f;
    float fps_ = 0.0f;
    int frame_count_ = 0;

    // Mouse state
    bool mouse_left_down_ = false;
    bool mouse_middle_down_ = false;
    bool mouse_right_down_ = false;
    double mouse_last_x_ = 0.0;
    double mouse_last_y_ = 0.0;
};

} // namespace cugs
