#pragma once

/// @file camera_controller.hpp
/// @brief Interactive orbit camera controller for the real-time viewer.
///
/// Provides an orbit camera that rotates around a target point with mouse
/// drag, pans with middle-click, and zooms with scroll wheel. Produces
/// CameraInfo structs compatible with the rasterizer (COLMAP convention:
/// X-right, Y-down, Z-forward).

#include "core/types.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <torch/torch.h>

#include <algorithm>
#include <cmath>

namespace cugs {

/// @brief Orbit camera controller for interactive viewing.
///
/// The camera orbits around a target point in spherical coordinates
/// (azimuth, elevation, radius). Produces world-to-camera rotation
/// and translation matrices in COLMAP convention for the rasterizer.
class CameraController {
public:
    /// @brief Initialize the camera to view a scene bounding box.
    /// @param bbox_min Minimum corner of the scene bounding box.
    /// @param bbox_max Maximum corner of the scene bounding box.
    void reset(const Eigen::Vector3f& bbox_min, const Eigen::Vector3f& bbox_max) {
        target_ = (bbox_min + bbox_max) * 0.5f;
        float extent = (bbox_max - bbox_min).norm();
        radius_ = std::max(extent * 1.5f, 0.1f);
        azimuth_ = 0.0f;
        elevation_ = 20.0f;  // slight downward angle
        fov_y_ = 50.0f;      // degrees
    }

    /// @brief Initialize from Gaussian positions tensor (CPU, [N, 3]).
    ///
    /// Computes a robust initial viewpoint by analyzing the point cloud
    /// distribution. Uses the median center (robust to outliers) and
    /// places the camera at a distance that frames the central 90% of
    /// Gaussians in view.
    void reset_from_positions(const torch::Tensor& positions_cpu) {
        int64_t n = positions_cpu.size(0);
        if (n == 0) {
            target_ = Eigen::Vector3f::Zero();
            radius_ = 5.0f;
            return;
        }

        // Use median as robust center (outlier-resistant)
        auto sorted_x = std::get<0>(positions_cpu.select(1, 0).sort(0));
        auto sorted_y = std::get<0>(positions_cpu.select(1, 1).sort(0));
        auto sorted_z = std::get<0>(positions_cpu.select(1, 2).sort(0));

        int64_t mid = n / 2;
        target_.x() = sorted_x[mid].item<float>();
        target_.y() = sorted_y[mid].item<float>();
        target_.z() = sorted_z[mid].item<float>();

        // Use 5th-95th percentile range for extent (ignore extreme outliers)
        int64_t p5 = n / 20;
        int64_t p95 = n - 1 - p5;
        float range_x = sorted_x[p95].item<float>() - sorted_x[p5].item<float>();
        float range_y = sorted_y[p95].item<float>() - sorted_y[p5].item<float>();
        float range_z = sorted_z[p95].item<float>() - sorted_z[p5].item<float>();
        float extent = std::sqrt(range_x * range_x + range_y * range_y + range_z * range_z);

        radius_ = std::max(extent * 1.2f, 0.1f);
        azimuth_ = 30.0f;    // angled view
        elevation_ = 20.0f;  // slight downward angle
        fov_y_ = 50.0f;
    }

    /// @brief Rotate the camera by mouse drag delta (in pixels).
    /// @param dx Horizontal pixel delta (positive = rotate right).
    /// @param dy Vertical pixel delta (positive = rotate down).
    void rotate(float dx, float dy) {
        azimuth_ -= dx * rotate_sensitivity_;
        elevation_ += dy * rotate_sensitivity_;
        // Clamp elevation to avoid gimbal lock
        elevation_ = std::clamp(elevation_, -89.0f, 89.0f);
    }

    /// @brief Pan the target point in the camera's local XY plane.
    /// @param dx Horizontal pixel delta.
    /// @param dy Vertical pixel delta.
    void pan(float dx, float dy) {
        auto [right, up, forward] = camera_axes();
        float scale = radius_ * pan_sensitivity_;
        // Right direction is camera X (screen right)
        // Up direction is world up projected to camera plane (screen up)
        target_ += right * (-dx * scale) + up * (dy * scale);
    }

    /// @brief Zoom by adjusting the orbit radius.
    /// @param delta Scroll delta (positive = zoom in).
    void zoom(float delta) {
        radius_ *= (1.0f - delta * zoom_sensitivity_);
        radius_ = std::max(radius_, 0.01f);
    }

    /// @brief Construct a CameraInfo for the current orbit state.
    ///
    /// Produces a world-to-camera transform in COLMAP convention:
    /// X-right, Y-down, Z-forward. The camera looks along +Z in
    /// camera space.
    ///
    /// @param width  Image width in pixels.
    /// @param height Image height in pixels.
    /// @return CameraInfo with world-to-camera extrinsics and pinhole intrinsics.
    CameraInfo build_camera(int width, int height) const {
        float az_rad = azimuth_ * kDegToRad;
        float el_rad = elevation_ * kDegToRad;

        // Camera position in world space (spherical -> Cartesian)
        float cos_el = std::cos(el_rad);
        Eigen::Vector3f cam_pos;
        cam_pos.x() = target_.x() + radius_ * cos_el * std::sin(az_rad);
        cam_pos.y() = target_.y() + radius_ * std::sin(el_rad);
        cam_pos.z() = target_.z() + radius_ * cos_el * std::cos(az_rad);

        // Build look-at vectors in world space
        Eigen::Vector3f forward = (target_ - cam_pos).normalized();
        Eigen::Vector3f world_up(0.0f, 1.0f, 0.0f);
        Eigen::Vector3f right = forward.cross(world_up).normalized();
        // Handle degenerate case when looking straight up/down
        if (right.squaredNorm() < 1e-6f) {
            world_up = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
            right = forward.cross(world_up).normalized();
        }
        Eigen::Vector3f up = right.cross(forward).normalized();

        // World-to-camera rotation in COLMAP convention:
        //   X_cam = right     (row 0)
        //   Y_cam = -up       (row 1, Y points down)
        //   Z_cam = forward   (row 2, Z points into the scene)
        Eigen::Matrix3f R;
        R.row(0) = right;
        R.row(1) = -up;
        R.row(2) = forward;

        // World-to-camera translation: t = -R * cam_pos
        Eigen::Vector3f t = -R * cam_pos;

        // Pinhole intrinsics from FOV
        float fy = static_cast<float>(height) / (2.0f * std::tan(fov_y_ * kDegToRad * 0.5f));
        float fx = fy;  // square pixels
        float cx = static_cast<float>(width) * 0.5f;
        float cy = static_cast<float>(height) * 0.5f;

        CameraInfo camera;
        camera.width = width;
        camera.height = height;
        camera.intrinsics = {fx, fy, cx, cy};
        camera.rotation = R;
        camera.translation = t;
        return camera;
    }

    // --- Accessors ---

    Eigen::Vector3f target() const { return target_; }
    float radius() const { return radius_; }
    float azimuth() const { return azimuth_; }
    float elevation() const { return elevation_; }
    float fov_y() const { return fov_y_; }

    void set_fov_y(float fov) { fov_y_ = std::clamp(fov, 10.0f, 120.0f); }
    void set_rotate_sensitivity(float s) { rotate_sensitivity_ = s; }
    void set_pan_sensitivity(float s) { pan_sensitivity_ = s; }
    void set_zoom_sensitivity(float s) { zoom_sensitivity_ = s; }

private:
    static constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;

    struct Axes {
        Eigen::Vector3f right, up, forward;
    };

    /// @brief Compute camera axes from current orbit state.
    Axes camera_axes() const {
        float az_rad = azimuth_ * kDegToRad;
        float el_rad = elevation_ * kDegToRad;

        float cos_el = std::cos(el_rad);
        Eigen::Vector3f cam_pos;
        cam_pos.x() = target_.x() + radius_ * cos_el * std::sin(az_rad);
        cam_pos.y() = target_.y() + radius_ * std::sin(el_rad);
        cam_pos.z() = target_.z() + radius_ * cos_el * std::cos(az_rad);

        Eigen::Vector3f forward = (target_ - cam_pos).normalized();
        Eigen::Vector3f world_up(0.0f, 1.0f, 0.0f);
        Eigen::Vector3f right = forward.cross(world_up).normalized();
        if (right.squaredNorm() < 1e-6f) {
            world_up = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
            right = forward.cross(world_up).normalized();
        }
        Eigen::Vector3f up = right.cross(forward).normalized();
        return {right, up, forward};
    }

    Eigen::Vector3f target_ = Eigen::Vector3f::Zero();
    float radius_ = 5.0f;
    float azimuth_ = 0.0f;      // degrees
    float elevation_ = 20.0f;   // degrees
    float fov_y_ = 50.0f;       // degrees

    float rotate_sensitivity_ = 0.3f;
    float pan_sensitivity_ = 0.002f;
    float zoom_sensitivity_ = 0.1f;
};

} // namespace cugs
