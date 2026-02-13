/// @file viewer.cpp
/// @brief Real-time Gaussian splatting viewer implementation.
///
/// Renders trained Gaussian models using the CUDA rasterizer and displays
/// results via an OpenGL fullscreen textured quad with ImGui overlay.

// Must be before any Windows header (including those pulled by torch/torch.h)
#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#endif

#include "viewer/viewer.hpp"
#include "utils/cuda_utils.cuh"

#include <spdlog/spdlog.h>
#include <torch/torch.h>

// OpenGL / GLFW / ImGui — only included in this TU
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace cugs {

// ---------------------------------------------------------------------------
// OpenGL 2.0+ function loading (Windows gl.h only provides GL 1.1)
// ---------------------------------------------------------------------------

// GL types and constants not in Windows gl.h
using GLchar = char;
#ifndef GL_FRAGMENT_SHADER
#define GL_FRAGMENT_SHADER        0x8B30
#define GL_VERTEX_SHADER          0x8B31
#define GL_COMPILE_STATUS         0x8B81
#define GL_LINK_STATUS            0x8B82
#define GL_INFO_LOG_LENGTH        0x8B84
#define GL_ARRAY_BUFFER           0x8892
#define GL_STATIC_DRAW            0x88E4
#define GL_TEXTURE0               0x84C0
#define GL_CLAMP_TO_EDGE          0x812F
#define GL_RGB32F                 0x8815
#define GL_RGBA32F                0x8814
#define GL_PIXEL_UNPACK_BUFFER    0x88EC
#define GL_STREAM_DRAW            0x88E0
#endif

// Function pointer types
using PFNGLCREATESHADERPROC       = GLuint (*)(GLenum type);
using PFNGLSHADERSOURCEPROC       = void (*)(GLuint shader, GLsizei count, const GLchar** string, const GLint* length);
using PFNGLCOMPILESHADERPROC      = void (*)(GLuint shader);
using PFNGLGETSHADERIVPROC        = void (*)(GLuint shader, GLenum pname, GLint* params);
using PFNGLGETSHADERINFOLOGPROC   = void (*)(GLuint shader, GLsizei maxLength, GLsizei* length, GLchar* infoLog);
using PFNGLDELETESHADERPROC       = void (*)(GLuint shader);
using PFNGLCREATEPROGRAMPROC      = GLuint (*)();
using PFNGLATTACHSHADERPROC       = void (*)(GLuint program, GLuint shader);
using PFNGLLINKPROGRAMPROC        = void (*)(GLuint program);
using PFNGLGETPROGRAMIVPROC       = void (*)(GLuint program, GLenum pname, GLint* params);
using PFNGLGETPROGRAMINFOLOGPROC  = void (*)(GLuint program, GLsizei maxLength, GLsizei* length, GLchar* infoLog);
using PFNGLDELETEPROGRAMPROC      = void (*)(GLuint program);
using PFNGLUSEPROGRAMPROC         = void (*)(GLuint program);
using PFNGLGETUNIFORMLOCATIONPROC = GLint (*)(GLuint program, const GLchar* name);
using PFNGLUNIFORM1IPROC          = void (*)(GLint location, GLint v0);
using PFNGLGETATTRIBLOCATIONPROC  = GLint (*)(GLuint program, const GLchar* name);
using PFNGLENABLEVERTEXATTRIBARRAYPROC  = void (*)(GLuint index);
using PFNGLDISABLEVERTEXATTRIBARRAYPROC = void (*)(GLuint index);
using PFNGLVERTEXATTRIBPOINTERPROC     = void (*)(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer);
using PFNGLGENBUFFERSPROC         = void (*)(GLsizei n, GLuint* buffers);
using PFNGLDELETEBUFFERSPROC      = void (*)(GLsizei n, const GLuint* buffers);
using PFNGLBINDBUFFERPROC         = void (*)(GLenum target, GLuint buffer);
using PFNGLBUFFERDATAPROC         = void (*)(GLenum target, ptrdiff_t size, const void* data, GLenum usage);
using PFNGLACTIVETEXTUREPROC      = void (*)(GLenum texture);

// Function pointers (loaded at init time)
static PFNGLCREATESHADERPROC       gl_CreateShader = nullptr;
static PFNGLSHADERSOURCEPROC       gl_ShaderSource = nullptr;
static PFNGLCOMPILESHADERPROC      gl_CompileShader = nullptr;
static PFNGLGETSHADERIVPROC        gl_GetShaderiv = nullptr;
static PFNGLGETSHADERINFOLOGPROC   gl_GetShaderInfoLog = nullptr;
static PFNGLDELETESHADERPROC       gl_DeleteShader = nullptr;
static PFNGLCREATEPROGRAMPROC      gl_CreateProgram = nullptr;
static PFNGLATTACHSHADERPROC       gl_AttachShader = nullptr;
static PFNGLLINKPROGRAMPROC        gl_LinkProgram = nullptr;
static PFNGLGETPROGRAMIVPROC       gl_GetProgramiv = nullptr;
static PFNGLGETPROGRAMINFOLOGPROC  gl_GetProgramInfoLog = nullptr;
static PFNGLDELETEPROGRAMPROC      gl_DeleteProgram = nullptr;
static PFNGLUSEPROGRAMPROC         gl_UseProgram = nullptr;
static PFNGLGETUNIFORMLOCATIONPROC gl_GetUniformLocation = nullptr;
static PFNGLUNIFORM1IPROC          gl_Uniform1i = nullptr;
static PFNGLGETATTRIBLOCATIONPROC  gl_GetAttribLocation = nullptr;
static PFNGLENABLEVERTEXATTRIBARRAYPROC  gl_EnableVertexAttribArray = nullptr;
static PFNGLDISABLEVERTEXATTRIBARRAYPROC gl_DisableVertexAttribArray = nullptr;
static PFNGLVERTEXATTRIBPOINTERPROC     gl_VertexAttribPointer = nullptr;
static PFNGLGENBUFFERSPROC         gl_GenBuffers = nullptr;
static PFNGLDELETEBUFFERSPROC      gl_DeleteBuffers = nullptr;
static PFNGLBINDBUFFERPROC         gl_BindBuffer = nullptr;
static PFNGLBUFFERDATAPROC         gl_BufferData = nullptr;
static PFNGLACTIVETEXTUREPROC      gl_ActiveTexture = nullptr;

/// @brief Load all GL 2.0+ functions we need. Must be called after context creation.
static void load_gl_functions() {
    auto load = [](const char* name) -> void* {
        void* p = reinterpret_cast<void*>(glfwGetProcAddress(name));
        if (!p) throw std::runtime_error(std::string("Failed to load GL function: ") + name);
        return p;
    };

    gl_CreateShader      = reinterpret_cast<PFNGLCREATESHADERPROC>(load("glCreateShader"));
    gl_ShaderSource      = reinterpret_cast<PFNGLSHADERSOURCEPROC>(load("glShaderSource"));
    gl_CompileShader     = reinterpret_cast<PFNGLCOMPILESHADERPROC>(load("glCompileShader"));
    gl_GetShaderiv       = reinterpret_cast<PFNGLGETSHADERIVPROC>(load("glGetShaderiv"));
    gl_GetShaderInfoLog  = reinterpret_cast<PFNGLGETSHADERINFOLOGPROC>(load("glGetShaderInfoLog"));
    gl_DeleteShader      = reinterpret_cast<PFNGLDELETESHADERPROC>(load("glDeleteShader"));
    gl_CreateProgram     = reinterpret_cast<PFNGLCREATEPROGRAMPROC>(load("glCreateProgram"));
    gl_AttachShader      = reinterpret_cast<PFNGLATTACHSHADERPROC>(load("glAttachShader"));
    gl_LinkProgram       = reinterpret_cast<PFNGLLINKPROGRAMPROC>(load("glLinkProgram"));
    gl_GetProgramiv      = reinterpret_cast<PFNGLGETPROGRAMIVPROC>(load("glGetProgramiv"));
    gl_GetProgramInfoLog = reinterpret_cast<PFNGLGETPROGRAMINFOLOGPROC>(load("glGetProgramInfoLog"));
    gl_DeleteProgram     = reinterpret_cast<PFNGLDELETEPROGRAMPROC>(load("glDeleteProgram"));
    gl_UseProgram        = reinterpret_cast<PFNGLUSEPROGRAMPROC>(load("glUseProgram"));
    gl_GetUniformLocation = reinterpret_cast<PFNGLGETUNIFORMLOCATIONPROC>(load("glGetUniformLocation"));
    gl_Uniform1i         = reinterpret_cast<PFNGLUNIFORM1IPROC>(load("glUniform1i"));
    gl_GetAttribLocation = reinterpret_cast<PFNGLGETATTRIBLOCATIONPROC>(load("glGetAttribLocation"));
    gl_EnableVertexAttribArray  = reinterpret_cast<PFNGLENABLEVERTEXATTRIBARRAYPROC>(load("glEnableVertexAttribArray"));
    gl_DisableVertexAttribArray = reinterpret_cast<PFNGLDISABLEVERTEXATTRIBARRAYPROC>(load("glDisableVertexAttribArray"));
    gl_VertexAttribPointer      = reinterpret_cast<PFNGLVERTEXATTRIBPOINTERPROC>(load("glVertexAttribPointer"));
    gl_GenBuffers        = reinterpret_cast<PFNGLGENBUFFERSPROC>(load("glGenBuffers"));
    gl_DeleteBuffers     = reinterpret_cast<PFNGLDELETEBUFFERSPROC>(load("glDeleteBuffers"));
    gl_BindBuffer        = reinterpret_cast<PFNGLBINDBUFFERPROC>(load("glBindBuffer"));
    gl_BufferData        = reinterpret_cast<PFNGLBUFFERDATAPROC>(load("glBufferData"));
    gl_ActiveTexture     = reinterpret_cast<PFNGLACTIVETEXTUREPROC>(load("glActiveTexture"));
}

// ---------------------------------------------------------------------------
// Fullscreen quad shader (OpenGL 2.1 / GLSL 120 for maximum compatibility)
// ---------------------------------------------------------------------------

static const char* kVertexShaderSrc = R"glsl(
#version 120
attribute vec2 a_pos;
attribute vec2 a_uv;
varying vec2 v_uv;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
    v_uv = a_uv;
}
)glsl";

static const char* kFragmentShaderSrc = R"glsl(
#version 120
varying vec2 v_uv;
uniform sampler2D u_texture;
void main() {
    gl_FragColor = texture2D(u_texture, v_uv);
}
)glsl";

// ---------------------------------------------------------------------------
// GL helpers
// ---------------------------------------------------------------------------

namespace {

/// @brief Compile a shader and return its ID. Throws on failure.
GLuint compile_shader(GLenum type, const char* source) {
    GLuint shader = gl_CreateShader(type);
    gl_ShaderSource(shader, 1, &source, nullptr);
    gl_CompileShader(shader);

    GLint success = 0;
    gl_GetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        gl_GetShaderInfoLog(shader, sizeof(log), nullptr, log);
        gl_DeleteShader(shader);
        throw std::runtime_error(std::string("Shader compile error: ") + log);
    }
    return shader;
}

/// @brief Link a vertex + fragment shader into a program.
GLuint link_program(GLuint vert, GLuint frag) {
    GLuint program = gl_CreateProgram();
    gl_AttachShader(program, vert);
    gl_AttachShader(program, frag);
    gl_LinkProgram(program);

    GLint success = 0;
    gl_GetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        gl_GetProgramInfoLog(program, sizeof(log), nullptr, log);
        gl_DeleteProgram(program);
        throw std::runtime_error(std::string("Shader link error: ") + log);
    }
    return program;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Viewer construction / destruction
// ---------------------------------------------------------------------------

Viewer::Viewer(std::filesystem::path ply_path, ViewerConfig config)
    : ply_path_(std::move(ply_path))
    , config_(config) {

    // Load model
    spdlog::info("Loading model from {}", ply_path_.string());
    model_ = GaussianModel::load_ply(ply_path_);
    spdlog::info("Loaded {} Gaussians (max SH degree {})",
                 model_.num_gaussians(), model_.max_sh_degree());

    // Move to GPU
    model_.to_device(torch::Device(torch::kCUDA, 0));

    // Initialize render settings
    render_settings_.active_sh_degree =
        (config_.sh_degree >= 0) ? std::min(config_.sh_degree, model_.max_sh_degree())
                                 : model_.max_sh_degree();
    std::memcpy(render_settings_.background, config_.background, sizeof(float) * 3);

    // Initialize camera from Gaussian positions (robust median-based viewpoint)
    auto positions_cpu = model_.positions.to(torch::kCPU);
    camera_.reset_from_positions(positions_cpu);
    spdlog::info("Camera: target=({:.2f}, {:.2f}, {:.2f}), radius={:.2f}, az={:.1f}, el={:.1f}",
                 camera_.target().x(), camera_.target().y(), camera_.target().z(),
                 camera_.radius(), camera_.azimuth(), camera_.elevation());
}

Viewer::~Viewer() {
    cleanup();
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

void Viewer::init_window() {
    glfwSetErrorCallback([](int error, const char* desc) {
        spdlog::error("GLFW error {}: {}", error, desc);
    });

    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    // Request OpenGL 2.1 (compatible profile) — sufficient for textured quad
    // and works without GLAD/GLEW on Windows.
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window_ = glfwCreateWindow(config_.width, config_.height,
                                "Gaussian Splatting Viewer", nullptr, nullptr);
    if (!window_) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(config_.vsync ? 1 : 0);

    // Store `this` for GLFW callbacks
    glfwSetWindowUserPointer(window_, this);

    // Framebuffer size (may differ from window size on HiDPI)
    glfwGetFramebufferSize(window_, &framebuffer_width_, &framebuffer_height_);

    // Scroll callback
    glfwSetScrollCallback(window_, [](GLFWwindow* w, double /*xoff*/, double yoff) {
        // Don't process if ImGui wants the input
        if (ImGui::GetIO().WantCaptureMouse) return;
        auto* self = static_cast<Viewer*>(glfwGetWindowUserPointer(w));
        self->camera_.zoom(static_cast<float>(yoff));
    });

    // Framebuffer resize callback
    glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int width, int height) {
        auto* self = static_cast<Viewer*>(glfwGetWindowUserPointer(w));
        self->framebuffer_width_ = width;
        self->framebuffer_height_ = height;
        glViewport(0, 0, width, height);
    });
}

void Viewer::init_gl_resources() {
    // Load GL 2.0+ extension functions
    load_gl_functions();

    // Pixel upload settings — use alignment of 1 for safety and RGBA format
    // to avoid driver bugs with 3-component float textures (GL_RGB32F).
    // Many NVIDIA drivers on Windows internally convert GL_RGB to GL_RGBA
    // with incorrect stride calculations, causing visible vertical seams.
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Create texture for rendered image (RGBA for driver compatibility)
    glGenTextures(1, &texture_id_);
    glBindTexture(GL_TEXTURE_2D, texture_id_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // Allocate initial texture storage
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F,
                 framebuffer_width_, framebuffer_height_,
                 0, GL_RGBA, GL_FLOAT, nullptr);
    texture_width_ = framebuffer_width_;
    texture_height_ = framebuffer_height_;
    glBindTexture(GL_TEXTURE_2D, 0);

    // Compile shaders
    GLuint vert = compile_shader(GL_VERTEX_SHADER, kVertexShaderSrc);
    GLuint frag = compile_shader(GL_FRAGMENT_SHADER, kFragmentShaderSrc);
    shader_program_ = link_program(vert, frag);
    gl_DeleteShader(vert);
    gl_DeleteShader(frag);

    // Fullscreen quad vertices: pos(x,y) + uv(s,t)
    // UV is flipped vertically (t: 1→0) because OpenGL textures are bottom-up
    // but our rendered image is top-down.
    float quad_vertices[] = {
        // pos       // uv
        -1.0f, -1.0f,  0.0f, 1.0f,  // bottom-left
         1.0f, -1.0f,  1.0f, 1.0f,  // bottom-right
        -1.0f,  1.0f,  0.0f, 0.0f,  // top-left
         1.0f,  1.0f,  1.0f, 0.0f,  // top-right
    };

    gl_GenBuffers(1, &vbo_);
    gl_BindBuffer(GL_ARRAY_BUFFER, vbo_);
    gl_BufferData(GL_ARRAY_BUFFER, sizeof(quad_vertices), quad_vertices, GL_STATIC_DRAW);
    gl_BindBuffer(GL_ARRAY_BUFFER, 0);

    // Initialize CUDA-GL interop for zero-copy GPU→texture transfer
    init_cuda_gl_interop();
}

void Viewer::init_imgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    // Scale UI for readability
    ImGui::GetStyle().ScaleAllSizes(1.2f);

    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 120");
}

void Viewer::cleanup() {
    if (window_) {
        cleanup_cuda_gl_interop();

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        if (shader_program_ && gl_DeleteProgram) gl_DeleteProgram(shader_program_);
        if (vbo_ && gl_DeleteBuffers) gl_DeleteBuffers(1, &vbo_);
        if (texture_id_) glDeleteTextures(1, &texture_id_);

        glfwDestroyWindow(window_);
        glfwTerminate();
        window_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Input handling
// ---------------------------------------------------------------------------

void Viewer::process_input() {
    // Don't process mouse if ImGui wants it
    if (ImGui::GetIO().WantCaptureMouse) {
        mouse_left_down_ = false;
        mouse_middle_down_ = false;
        mouse_right_down_ = false;
        return;
    }

    double mx, my;
    glfwGetCursorPos(window_, &mx, &my);

    bool left = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    bool middle = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    bool right = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    if (mouse_left_down_ && left) {
        float dx = static_cast<float>(mx - mouse_last_x_);
        float dy = static_cast<float>(my - mouse_last_y_);
        camera_.rotate(dx, dy);
    }

    if ((mouse_middle_down_ && middle) || (mouse_right_down_ && right)) {
        float dx = static_cast<float>(mx - mouse_last_x_);
        float dy = static_cast<float>(my - mouse_last_y_);
        camera_.pan(dx, dy);
    }

    mouse_left_down_ = left;
    mouse_middle_down_ = middle;
    mouse_right_down_ = right;
    mouse_last_x_ = mx;
    mouse_last_y_ = my;

    // Keyboard shortcuts
    if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window_, GLFW_TRUE);
    }
    if (glfwGetKey(window_, GLFW_KEY_1) == GLFW_PRESS) render_mode_ = RenderMode::kRGB;
    if (glfwGetKey(window_, GLFW_KEY_2) == GLFW_PRESS) render_mode_ = RenderMode::kDepth;
    if (glfwGetKey(window_, GLFW_KEY_3) == GLFW_PRESS) render_mode_ = RenderMode::kHeatmap;
}

// ---------------------------------------------------------------------------
// Colormaps
// ---------------------------------------------------------------------------

torch::Tensor Viewer::apply_turbo_colormap(const torch::Tensor& values) {
    // Simplified turbo-like colormap: blue → cyan → green → yellow → red
    auto v = values.unsqueeze(-1);  // [H,W,1]
    auto r = torch::clamp(1.5f - torch::abs(v * 4.0f - 3.0f), 0.0f, 1.0f);
    auto g = torch::clamp(1.5f - torch::abs(v * 4.0f - 2.0f), 0.0f, 1.0f);
    auto b = torch::clamp(1.5f - torch::abs(v * 4.0f - 1.0f), 0.0f, 1.0f);
    return torch::cat({r, g, b}, -1);  // [H,W,3]
}

torch::Tensor Viewer::apply_heat_colormap(const torch::Tensor& values) {
    // Heat colormap: blue → green → yellow → red
    auto v = values.unsqueeze(-1);  // [H,W,1]
    auto r = torch::clamp(v * 3.0f - 1.0f, 0.0f, 1.0f);
    auto g = torch::clamp(torch::min(v * 3.0f, 3.0f - v * 3.0f), 0.0f, 1.0f);
    auto b = torch::clamp(1.0f - v * 3.0f, 0.0f, 1.0f);
    return torch::cat({r, g, b}, -1);  // [H,W,3]
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

void Viewer::upload_texture(const torch::Tensor& image) {
    // image: [H, W, 4] float32 RGBA, assumed contiguous and on CPU
    int h = static_cast<int>(image.size(0));
    int w = static_cast<int>(image.size(1));

    glBindTexture(GL_TEXTURE_2D, texture_id_);

    // Re-allocate if size changed (compare against actual texture dimensions)
    if (w != texture_width_ || h != texture_height_) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT,
                     image.data_ptr<float>());
        texture_width_ = w;
        texture_height_ = h;
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT,
                        image.data_ptr<float>());
    }

    glBindTexture(GL_TEXTURE_2D, 0);
}

void Viewer::init_cuda_gl_interop() {
    size_t buf_size = static_cast<size_t>(framebuffer_width_) * framebuffer_height_ * 4 * sizeof(float);
    if (buf_size == 0) {
        interop_available_ = false;
        return;
    }

    gl_GenBuffers(1, &pbo_id_);
    if (pbo_id_ == 0) {
        spdlog::warn("CUDA-GL interop: failed to create PBO, using CPU fallback");
        interop_available_ = false;
        return;
    }

    gl_BindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id_);
    gl_BufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<ptrdiff_t>(buf_size), nullptr, GL_STREAM_DRAW);
    gl_BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaError_t err = cudaGraphicsGLRegisterBuffer(
        &cuda_pbo_resource_, pbo_id_, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        spdlog::warn("CUDA-GL interop: cudaGraphicsGLRegisterBuffer failed ({}), using CPU fallback",
                     cudaGetErrorString(err));
        gl_DeleteBuffers(1, &pbo_id_);
        pbo_id_ = 0;
        cuda_pbo_resource_ = nullptr;
        interop_available_ = false;
        return;
    }

    pbo_size_ = buf_size;
    interop_available_ = true;
    spdlog::info("CUDA-GL interop enabled (PBO {} bytes)", buf_size);
}

void Viewer::resize_pbo(int w, int h) {
    size_t new_size = static_cast<size_t>(w) * h * 4 * sizeof(float);
    if (new_size == pbo_size_) return;

    // Unregister old resource
    if (cuda_pbo_resource_) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource_);
        cuda_pbo_resource_ = nullptr;
    }

    // Resize the PBO
    gl_BindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id_);
    gl_BufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<ptrdiff_t>(new_size), nullptr, GL_STREAM_DRAW);
    gl_BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Re-register
    cudaError_t err = cudaGraphicsGLRegisterBuffer(
        &cuda_pbo_resource_, pbo_id_, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        spdlog::warn("CUDA-GL interop: resize re-register failed ({}), falling back to CPU",
                     cudaGetErrorString(err));
        interop_available_ = false;
        return;
    }

    pbo_size_ = new_size;
}

void Viewer::upload_texture_interop(const torch::Tensor& rgba_gpu) {
    int h = static_cast<int>(rgba_gpu.size(0));
    int w = static_cast<int>(rgba_gpu.size(1));
    size_t byte_count = static_cast<size_t>(w) * h * 4 * sizeof(float);

    // Resize PBO if needed
    if (byte_count != pbo_size_) {
        resize_pbo(w, h);
        if (!interop_available_) return;
    }

    // Map PBO into CUDA address space
    cudaError_t err = cudaGraphicsMapResources(1, &cuda_pbo_resource_, 0);
    if (err != cudaSuccess) {
        spdlog::warn("cudaGraphicsMapResources failed: {}", cudaGetErrorString(err));
        interop_available_ = false;
        return;
    }

    void* pbo_ptr = nullptr;
    size_t mapped_size = 0;
    err = cudaGraphicsResourceGetMappedPointer(&pbo_ptr, &mapped_size, cuda_pbo_resource_);
    if (err != cudaSuccess) {
        cudaGraphicsUnmapResources(1, &cuda_pbo_resource_, 0);
        spdlog::warn("cudaGraphicsResourceGetMappedPointer failed: {}", cudaGetErrorString(err));
        interop_available_ = false;
        return;
    }

    // GPU-to-GPU copy (no CPU sync!)
    cudaMemcpy(pbo_ptr, rgba_gpu.data_ptr<float>(), byte_count, cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1, &cuda_pbo_resource_, 0);

    // Upload from PBO to texture
    glBindTexture(GL_TEXTURE_2D, texture_id_);
    gl_BindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_id_);

    if (w != texture_width_ || h != texture_height_) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
        texture_width_ = w;
        texture_height_ = h;
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_FLOAT, nullptr);
    }

    gl_BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Viewer::cleanup_cuda_gl_interop() {
    if (cuda_pbo_resource_) {
        cudaGraphicsUnregisterResource(cuda_pbo_resource_);
        cuda_pbo_resource_ = nullptr;
    }
    if (pbo_id_ && gl_DeleteBuffers) {
        gl_DeleteBuffers(1, &pbo_id_);
        pbo_id_ = 0;
    }
    pbo_size_ = 0;
    interop_available_ = false;
}

void Viewer::check_dirty() {
    bool camera_moved = camera_.version() != last_camera_version_;
    bool settings_changed =
        render_settings_.active_sh_degree != last_render_settings_.active_sh_degree ||
        render_settings_.scale_modifier != last_render_settings_.scale_modifier ||
        render_settings_.background[0] != last_render_settings_.background[0] ||
        render_settings_.background[1] != last_render_settings_.background[1] ||
        render_settings_.background[2] != last_render_settings_.background[2];
    bool mode_changed = render_mode_ != last_render_mode_;
    bool size_changed = framebuffer_width_ != last_render_width_ ||
                        framebuffer_height_ != last_render_height_;

    if (camera_moved || settings_changed || mode_changed || size_changed) {
        needs_render_ = true;
        camera_moved_this_frame_ = camera_moved;
    } else if (last_was_interactive_) {
        // Camera just stopped — render one full-resolution frame
        needs_render_ = true;
        camera_moved_this_frame_ = false;
    } else {
        needs_render_ = false;
        camera_moved_this_frame_ = false;
    }
}

void Viewer::render_frame() {
    // Ensure we have a valid framebuffer
    if (framebuffer_width_ <= 0 || framebuffer_height_ <= 0) return;

    // Interactive resolution scaling: render at half resolution during camera drag
    int rw = framebuffer_width_;
    int rh = framebuffer_height_;
    if (camera_moved_this_frame_) {
        rw = std::max(rw / 2, 1);
        rh = std::max(rh / 2, 1);
    }
    render_width_ = rw;
    render_height_ = rh;

    // Build camera for current view
    auto camera = camera_.build_camera(rw, rh);

    // Render with CUDA rasterizer
    torch::NoGradGuard no_grad;
    auto render_out = render(model_, camera, render_settings_);

    // Select display tensor based on render mode
    torch::Tensor display;
    switch (render_mode_) {
        case RenderMode::kRGB:
            display = render_out.color;
            break;

        case RenderMode::kDepth: {
            // Use 1 - final_T as depth proxy (high where opaque, low where transparent)
            auto depth = 1.0f - render_out.final_T;
            display = apply_turbo_colormap(depth);
            break;
        }

        case RenderMode::kHeatmap: {
            auto contrib = render_out.n_contrib.to(torch::kFloat32);
            float max_val = contrib.max().item<float>();
            if (max_val > 0.0f) {
                contrib = contrib / max_val;
            }
            display = apply_heat_colormap(contrib);
            break;
        }
    }

    // Convert RGB [H,W,3] to RGBA [H,W,4].
    // RGBA avoids GL_RGB32F driver bugs that cause vertical seam artifacts.
    auto alpha = torch::ones({display.size(0), display.size(1), 1}, display.options());
    auto rgba = torch::cat({display, alpha}, /*dim=*/2).contiguous();

    if (interop_available_) {
        // Fast path: GPU-to-GPU transfer via PBO (no CPU sync)
        upload_texture_interop(rgba);
        if (!interop_available_) {
            // Interop failed mid-frame, fall back to CPU path
            auto cpu_image = rgba.to(torch::kCPU).contiguous();
            upload_texture(cpu_image);
        }
    } else {
        auto cpu_image = rgba.to(torch::kCPU).contiguous();
        upload_texture(cpu_image);
    }

    // First-frame diagnostics to verify rendering is working
    if (frame_count_ == 0) {
        auto cpu_diag = rgba.to(torch::kCPU);
        float pixel_min = cpu_diag.min().item<float>();
        float pixel_max = cpu_diag.max().item<float>();
        float pixel_mean = cpu_diag.mean().item<float>();
        spdlog::info("First frame: pixel range [{:.4f}, {:.4f}], mean={:.4f}, "
                     "resolution={}x{}, n_contrib max={}",
                     pixel_min, pixel_max, pixel_mean,
                     rw, rh,
                     render_out.n_contrib.max().item<int>());
    }
    ++frame_count_;

    // Update dirty tracking state
    last_camera_version_ = camera_.version();
    last_render_settings_ = render_settings_;
    last_render_mode_ = render_mode_;
    last_render_width_ = framebuffer_width_;
    last_render_height_ = framebuffer_height_;
    last_was_interactive_ = camera_moved_this_frame_;
}

void Viewer::draw_scene() {
    // Draw fullscreen quad with cached texture
    glClear(GL_COLOR_BUFFER_BIT);

    gl_UseProgram(shader_program_);

    // Set texture uniform
    GLint tex_loc = gl_GetUniformLocation(shader_program_, "u_texture");
    gl_Uniform1i(tex_loc, 0);
    gl_ActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_id_);

    // Bind VBO and set vertex attributes
    gl_BindBuffer(GL_ARRAY_BUFFER, vbo_);

    GLint pos_attr = gl_GetAttribLocation(shader_program_, "a_pos");
    GLint uv_attr = gl_GetAttribLocation(shader_program_, "a_uv");

    gl_EnableVertexAttribArray(pos_attr);
    gl_VertexAttribPointer(pos_attr, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

    gl_EnableVertexAttribArray(uv_attr);
    gl_VertexAttribPointer(uv_attr, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                           (void*)(2 * sizeof(float)));

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    gl_DisableVertexAttribArray(pos_attr);
    gl_DisableVertexAttribArray(uv_attr);
    gl_BindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    gl_UseProgram(0);
}

// ---------------------------------------------------------------------------
// ImGui overlay
// ---------------------------------------------------------------------------

void Viewer::draw_imgui() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(320, 0), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Viewer Controls")) {
        // Frame timing
        ImGui::Text("FPS: %.1f (%.2f ms)", fps_, frame_dt_ * 1000.0f);
        ImGui::Separator();

        // Model info
        ImGui::Text("Gaussians: %lld", static_cast<long long>(model_.num_gaussians()));
        ImGui::Text("Max SH degree: %d", model_.max_sh_degree());
        ImGui::Text("Resolution: %d x %d", framebuffer_width_, framebuffer_height_);
        if (render_width_ > 0 && render_height_ > 0 &&
            (render_width_ != framebuffer_width_ || render_height_ != framebuffer_height_)) {
            ImGui::Text("Render: %d x %d (interactive)", render_width_, render_height_);
        } else {
            ImGui::Text("Render: %d x %d", framebuffer_width_, framebuffer_height_);
        }
        ImGui::Text("Transfer: %s", interop_available_ ? "CUDA-GL interop" : "CPU fallback");

        // VRAM
        auto vram = vram_info_mb();
        if (vram.valid()) {
            ImGui::Text("VRAM: %.0f / %.0f MB", vram.used_mb(), vram.total_mb);
        }
        ImGui::Separator();

        // Render mode
        const char* mode_names[] = {"RGB", "Depth", "Heatmap"};
        int mode_idx = static_cast<int>(render_mode_);
        if (ImGui::Combo("Render Mode", &mode_idx, mode_names, 3)) {
            render_mode_ = static_cast<RenderMode>(mode_idx);
        }
        ImGui::Text("(Keys: 1=RGB, 2=Depth, 3=Heatmap)");

        // SH degree
        ImGui::SliderInt("SH Degree", &render_settings_.active_sh_degree,
                         0, model_.max_sh_degree());

        // Scale modifier
        ImGui::SliderFloat("Scale", &render_settings_.scale_modifier, 0.1f, 2.0f);

        // Background color
        ImGui::ColorEdit3("Background", render_settings_.background);

        ImGui::Separator();

        // Camera info
        if (ImGui::CollapsingHeader("Camera")) {
            ImGui::Text("Azimuth:   %.1f deg", camera_.azimuth());
            ImGui::Text("Elevation: %.1f deg", camera_.elevation());
            ImGui::Text("Radius:    %.2f", camera_.radius());
            ImGui::Text("FOV:       %.1f deg", camera_.fov_y());

            float fov = camera_.fov_y();
            if (ImGui::SliderFloat("FOV Y", &fov, 10.0f, 120.0f)) {
                camera_.set_fov_y(fov);
            }
        }
    }
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------

void Viewer::run() {
    init_window();
    init_gl_resources();
    init_imgui();

    spdlog::info("Viewer started — {} x {} — {} Gaussians",
                 framebuffer_width_, framebuffer_height_, model_.num_gaussians());
    spdlog::info("Controls: Left-drag=rotate, Middle/Right-drag=pan, Scroll=zoom, Esc=quit");

    last_frame_time_ = glfwGetTime();

    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();
        process_input();
        check_dirty();

        if (needs_render_) {
            render_frame();
            needs_render_ = false;
        }

        draw_scene();
        draw_imgui();

        glfwSwapBuffers(window_);

        // Update timing
        double now = glfwGetTime();
        frame_dt_ = static_cast<float>(now - last_frame_time_);
        fps_ = (frame_dt_ > 0.0f) ? (1.0f / frame_dt_) : 0.0f;
        last_frame_time_ = now;
    }

    spdlog::info("Viewer closed");
}

} // namespace cugs
