#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cstdint>
#include <string>
#include <vector>

namespace cugs {

// ---------------------------------------------------------------------------
// Camera model enum — matches COLMAP's model IDs
// ---------------------------------------------------------------------------

enum class CameraModel : int32_t {
    kSimplePinhole = 0,  // params: f, cx, cy
    kPinhole = 1,        // params: fx, fy, cx, cy
    kSimpleRadial = 2,   // params: f, cx, cy, k1
    kRadial = 3,         // params: f, cx, cy, k1, k2
    kOpenCV = 4,         // params: fx, fy, cx, cy, k1, k2, p1, p2
};

/// @brief Number of intrinsic parameters for a given camera model.
inline int camera_model_num_params(CameraModel model) {
    switch (model) {
        case CameraModel::kSimplePinhole: return 3;
        case CameraModel::kPinhole:       return 4;
        case CameraModel::kSimpleRadial:  return 4;
        case CameraModel::kRadial:        return 5;
        case CameraModel::kOpenCV:        return 8;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Normalised camera intrinsics — always (fx, fy, cx, cy)
// ---------------------------------------------------------------------------

struct CameraIntrinsics {
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
};

// ---------------------------------------------------------------------------
// Raw COLMAP parse results (double precision, matching binary format)
// ---------------------------------------------------------------------------

struct ColmapCamera {
    uint32_t camera_id = 0;
    CameraModel model = CameraModel::kPinhole;
    uint64_t width = 0;
    uint64_t height = 0;
    std::vector<double> params;  // model-dependent parameter vector
};

struct ColmapImage {
    uint32_t image_id = 0;
    double qvec[4] = {};   // quaternion wxyz (scalar-first), world-to-camera
    double tvec[3] = {};   // translation, world-to-camera
    uint32_t camera_id = 0;
    std::string name;      // image filename (e.g. "IMG_0001.JPG")
};

struct SparsePoint {
    uint64_t point_id = 0;
    Eigen::Vector3f position = Eigen::Vector3f::Zero();
    uint8_t color[3] = {};  // RGB, 0-255
    float error = 0.0f;     // reprojection error
};

// ---------------------------------------------------------------------------
// Merged camera info — the type the pipeline actually consumes
// ---------------------------------------------------------------------------

struct CameraInfo {
    uint32_t image_id = 0;
    uint32_t camera_id = 0;
    int width = 0;
    int height = 0;
    CameraIntrinsics intrinsics;

    /// World-to-camera rotation (from COLMAP quaternion wxyz).
    Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();

    /// World-to-camera translation.
    Eigen::Vector3f translation = Eigen::Vector3f::Zero();

    /// Image filename relative to dataset images/ directory.
    std::string image_name;

    /// Full path to image on disk (resolved by Dataset).
    std::string image_path;

    /// Camera center in world coordinates: C = -R^T * t
    Eigen::Vector3f camera_center() const {
        return -rotation.transpose() * translation;
    }

    /// Full 4x4 world-to-camera transform.
    Eigen::Matrix4f world_to_camera() const {
        Eigen::Matrix4f m = Eigen::Matrix4f::Identity();
        m.block<3, 3>(0, 0) = rotation;
        m.block<3, 1>(0, 3) = translation;
        return m;
    }
};

// ---------------------------------------------------------------------------
// Helper: quaternion (wxyz) to rotation matrix
// ---------------------------------------------------------------------------

/// @brief Convert a unit quaternion (w, x, y, z) to a 3x3 rotation matrix.
///        Follows COLMAP convention: world-to-camera rotation.
inline Eigen::Matrix3f qvec_to_rotation(double w, double x, double y, double z) {
    // Normalise for safety
    Eigen::Quaterniond q(w, x, y, z);
    q.normalize();
    return q.toRotationMatrix().cast<float>();
}

} // namespace cugs
