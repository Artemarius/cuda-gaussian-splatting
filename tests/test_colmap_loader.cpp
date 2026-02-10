#include <gtest/gtest.h>

#include "core/types.hpp"
#include "data/colmap_loader.hpp"

#include <Eigen/Core>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Binary write helpers — mirror the read helpers in colmap_loader.cpp
// ---------------------------------------------------------------------------

template <typename T>
void write_binary(std::ofstream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_null_terminated(std::ofstream& out, const std::string& str) {
    out.write(str.c_str(), static_cast<std::streamsize>(str.size() + 1));
}

// ---------------------------------------------------------------------------
// Synthetic COLMAP data generators
// ---------------------------------------------------------------------------

/// @brief Create a minimal cameras.bin with two cameras (SIMPLE_PINHOLE + PINHOLE).
void create_cameras_bin(const std::filesystem::path& path) {
    std::ofstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    // Number of cameras
    write_binary<uint64_t>(f, 2);

    // Camera 1: SIMPLE_PINHOLE (model 0), 3 params: f, cx, cy
    write_binary<uint32_t>(f, 1);       // camera_id
    write_binary<uint32_t>(f, 0);       // model = SIMPLE_PINHOLE
    write_binary<uint64_t>(f, 800);     // width
    write_binary<uint64_t>(f, 600);     // height
    write_binary<double>(f, 500.0);     // f
    write_binary<double>(f, 400.0);     // cx
    write_binary<double>(f, 300.0);     // cy

    // Camera 2: PINHOLE (model 1), 4 params: fx, fy, cx, cy
    write_binary<uint32_t>(f, 2);       // camera_id
    write_binary<uint32_t>(f, 1);       // model = PINHOLE
    write_binary<uint64_t>(f, 1920);    // width
    write_binary<uint64_t>(f, 1080);    // height
    write_binary<double>(f, 1000.0);    // fx
    write_binary<double>(f, 1000.0);    // fy
    write_binary<double>(f, 960.0);     // cx
    write_binary<double>(f, 540.0);     // cy
}

/// @brief Create a minimal images.bin with 3 images.
void create_images_bin(const std::filesystem::path& path) {
    std::ofstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    write_binary<uint64_t>(f, 3);  // num images

    // Image 1: identity rotation, zero translation
    write_binary<uint32_t>(f, 1);       // image_id
    write_binary<double>(f, 1.0);       // qw
    write_binary<double>(f, 0.0);       // qx
    write_binary<double>(f, 0.0);       // qy
    write_binary<double>(f, 0.0);       // qz
    write_binary<double>(f, 0.0);       // tx
    write_binary<double>(f, 0.0);       // ty
    write_binary<double>(f, 0.0);       // tz
    write_binary<uint32_t>(f, 1);       // camera_id
    write_null_terminated(f, "img_001.jpg");
    write_binary<uint64_t>(f, 0);       // num 2D points

    // Image 2: 90-degree rotation around Y, translation (1,2,3)
    // Quaternion for 90° around Y: (cos(45°), 0, sin(45°), 0) = (0.7071, 0, 0.7071, 0)
    write_binary<uint32_t>(f, 2);
    write_binary<double>(f, 0.7071067811865476);   // qw
    write_binary<double>(f, 0.0);                  // qx
    write_binary<double>(f, 0.7071067811865476);   // qy
    write_binary<double>(f, 0.0);                  // qz
    write_binary<double>(f, 1.0);                  // tx
    write_binary<double>(f, 2.0);                  // ty
    write_binary<double>(f, 3.0);                  // tz
    write_binary<uint32_t>(f, 1);                  // camera_id
    write_null_terminated(f, "img_002.jpg");
    // 2 dummy 2D points to verify skip logic
    write_binary<uint64_t>(f, 2);
    // point 0: x, y, point3d_id
    write_binary<double>(f, 100.0);
    write_binary<double>(f, 200.0);
    write_binary<uint64_t>(f, 42);
    // point 1
    write_binary<double>(f, 300.0);
    write_binary<double>(f, 400.0);
    write_binary<uint64_t>(f, 99);

    // Image 3: identity rotation, translation (5,0,0), uses camera 2
    write_binary<uint32_t>(f, 3);
    write_binary<double>(f, 1.0);  // qw
    write_binary<double>(f, 0.0);  // qx
    write_binary<double>(f, 0.0);  // qy
    write_binary<double>(f, 0.0);  // qz
    write_binary<double>(f, 5.0);  // tx
    write_binary<double>(f, 0.0);  // ty
    write_binary<double>(f, 0.0);  // tz
    write_binary<uint32_t>(f, 2);  // camera_id
    write_null_terminated(f, "img_003.jpg");
    write_binary<uint64_t>(f, 0);  // num 2D points
}

/// @brief Create a minimal points3D.bin with 4 points.
void create_points3d_bin(const std::filesystem::path& path) {
    std::ofstream f(path, std::ios::binary);
    ASSERT_TRUE(f.is_open());

    write_binary<uint64_t>(f, 4);  // num points

    // Point 1: origin, red
    write_binary<uint64_t>(f, 1);     // point_id
    write_binary<double>(f, 0.0);     // x
    write_binary<double>(f, 0.0);     // y
    write_binary<double>(f, 0.0);     // z
    write_binary<uint8_t>(f, 255);    // r
    write_binary<uint8_t>(f, 0);      // g
    write_binary<uint8_t>(f, 0);      // b
    write_binary<double>(f, 0.5);     // error
    write_binary<uint64_t>(f, 0);     // track_len

    // Point 2: (1,0,0), green, with track
    write_binary<uint64_t>(f, 2);
    write_binary<double>(f, 1.0);
    write_binary<double>(f, 0.0);
    write_binary<double>(f, 0.0);
    write_binary<uint8_t>(f, 0);
    write_binary<uint8_t>(f, 255);
    write_binary<uint8_t>(f, 0);
    write_binary<double>(f, 1.0);
    write_binary<uint64_t>(f, 2);     // track_len = 2
    write_binary<uint32_t>(f, 1);     // image_id
    write_binary<uint32_t>(f, 0);     // point2d_idx
    write_binary<uint32_t>(f, 2);     // image_id
    write_binary<uint32_t>(f, 1);     // point2d_idx

    // Point 3: (0,1,0), blue
    write_binary<uint64_t>(f, 3);
    write_binary<double>(f, 0.0);
    write_binary<double>(f, 1.0);
    write_binary<double>(f, 0.0);
    write_binary<uint8_t>(f, 0);
    write_binary<uint8_t>(f, 0);
    write_binary<uint8_t>(f, 255);
    write_binary<double>(f, 0.3);
    write_binary<uint64_t>(f, 0);

    // Point 4: (0,0,1), white
    write_binary<uint64_t>(f, 4);
    write_binary<double>(f, 0.0);
    write_binary<double>(f, 0.0);
    write_binary<double>(f, 1.0);
    write_binary<uint8_t>(f, 255);
    write_binary<uint8_t>(f, 255);
    write_binary<uint8_t>(f, 255);
    write_binary<double>(f, 0.1);
    write_binary<uint64_t>(f, 0);
}

/// @brief Create a complete synthetic COLMAP sparse directory and return its path.
std::filesystem::path create_synthetic_sparse(const std::filesystem::path& test_dir) {
    auto sparse_dir = test_dir / "sparse" / "0";
    std::filesystem::create_directories(sparse_dir);
    create_cameras_bin(sparse_dir / "cameras.bin");
    create_images_bin(sparse_dir / "images.bin");
    create_points3d_bin(sparse_dir / "points3D.bin");
    return sparse_dir;
}

// Temporary directory helper
class TempDir {
public:
    TempDir() {
        path_ = std::filesystem::temp_directory_path() / "cugs_test_colmap";
        std::filesystem::create_directories(path_);
    }
    ~TempDir() {
        std::filesystem::remove_all(path_);
    }
    const std::filesystem::path& path() const { return path_; }
private:
    std::filesystem::path path_;
};

} // anonymous namespace

// ===========================================================================
// Tests
// ===========================================================================

class ColmapLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        sparse_dir_ = create_synthetic_sparse(tmp_.path());
    }

    TempDir tmp_;
    std::filesystem::path sparse_dir_;
};

TEST_F(ColmapLoaderTest, ParseCamerasBin_Count) {
    auto cameras = cugs::parse_cameras_bin(sparse_dir_ / "cameras.bin");
    EXPECT_EQ(cameras.size(), 2u);
}

TEST_F(ColmapLoaderTest, ParseCamerasBin_SimplePinhole) {
    auto cameras = cugs::parse_cameras_bin(sparse_dir_ / "cameras.bin");
    ASSERT_GE(cameras.size(), 1u);

    const auto& cam = cameras[0];
    EXPECT_EQ(cam.camera_id, 1u);
    EXPECT_EQ(cam.model, cugs::CameraModel::kSimplePinhole);
    EXPECT_EQ(cam.width, 800u);
    EXPECT_EQ(cam.height, 600u);
    ASSERT_EQ(cam.params.size(), 3u);
    EXPECT_DOUBLE_EQ(cam.params[0], 500.0);  // f
    EXPECT_DOUBLE_EQ(cam.params[1], 400.0);  // cx
    EXPECT_DOUBLE_EQ(cam.params[2], 300.0);  // cy
}

TEST_F(ColmapLoaderTest, ParseCamerasBin_Pinhole) {
    auto cameras = cugs::parse_cameras_bin(sparse_dir_ / "cameras.bin");
    ASSERT_GE(cameras.size(), 2u);

    const auto& cam = cameras[1];
    EXPECT_EQ(cam.camera_id, 2u);
    EXPECT_EQ(cam.model, cugs::CameraModel::kPinhole);
    EXPECT_EQ(cam.width, 1920u);
    EXPECT_EQ(cam.height, 1080u);
    ASSERT_EQ(cam.params.size(), 4u);
    EXPECT_DOUBLE_EQ(cam.params[0], 1000.0);  // fx
    EXPECT_DOUBLE_EQ(cam.params[1], 1000.0);  // fy
    EXPECT_DOUBLE_EQ(cam.params[2], 960.0);   // cx
    EXPECT_DOUBLE_EQ(cam.params[3], 540.0);   // cy
}

TEST_F(ColmapLoaderTest, ParseImagesBin_Count) {
    auto images = cugs::parse_images_bin(sparse_dir_ / "images.bin");
    EXPECT_EQ(images.size(), 3u);
}

TEST_F(ColmapLoaderTest, ParseImagesBin_IdentityRotation) {
    auto images = cugs::parse_images_bin(sparse_dir_ / "images.bin");
    ASSERT_GE(images.size(), 1u);

    const auto& img = images[0];
    EXPECT_EQ(img.image_id, 1u);
    EXPECT_DOUBLE_EQ(img.qvec[0], 1.0);  // qw
    EXPECT_DOUBLE_EQ(img.qvec[1], 0.0);  // qx
    EXPECT_DOUBLE_EQ(img.qvec[2], 0.0);  // qy
    EXPECT_DOUBLE_EQ(img.qvec[3], 0.0);  // qz
    EXPECT_DOUBLE_EQ(img.tvec[0], 0.0);
    EXPECT_DOUBLE_EQ(img.tvec[1], 0.0);
    EXPECT_DOUBLE_EQ(img.tvec[2], 0.0);
    EXPECT_EQ(img.camera_id, 1u);
    EXPECT_EQ(img.name, "img_001.jpg");
}

TEST_F(ColmapLoaderTest, ParseImagesBin_RotatedImage) {
    auto images = cugs::parse_images_bin(sparse_dir_ / "images.bin");
    ASSERT_GE(images.size(), 2u);

    const auto& img = images[1];
    EXPECT_EQ(img.image_id, 2u);
    EXPECT_NEAR(img.qvec[0], 0.7071067811865476, 1e-10);
    EXPECT_NEAR(img.qvec[2], 0.7071067811865476, 1e-10);
    EXPECT_DOUBLE_EQ(img.tvec[0], 1.0);
    EXPECT_DOUBLE_EQ(img.tvec[1], 2.0);
    EXPECT_DOUBLE_EQ(img.tvec[2], 3.0);
    EXPECT_EQ(img.name, "img_002.jpg");
}

TEST_F(ColmapLoaderTest, ParseImagesBin_SkipsPoint2D) {
    // Image 2 has 2 dummy 2D points — the parser must skip them correctly
    // to read Image 3 properly
    auto images = cugs::parse_images_bin(sparse_dir_ / "images.bin");
    ASSERT_EQ(images.size(), 3u);

    const auto& img = images[2];
    EXPECT_EQ(img.image_id, 3u);
    EXPECT_EQ(img.camera_id, 2u);
    EXPECT_EQ(img.name, "img_003.jpg");
    EXPECT_DOUBLE_EQ(img.tvec[0], 5.0);
}

TEST_F(ColmapLoaderTest, ParsePoints3dBin_Count) {
    auto points = cugs::parse_points3d_bin(sparse_dir_ / "points3D.bin");
    EXPECT_EQ(points.size(), 4u);
}

TEST_F(ColmapLoaderTest, ParsePoints3dBin_PositionAndColor) {
    auto points = cugs::parse_points3d_bin(sparse_dir_ / "points3D.bin");
    ASSERT_GE(points.size(), 4u);

    // Point 1: origin, red
    EXPECT_EQ(points[0].point_id, 1u);
    EXPECT_FLOAT_EQ(points[0].position.x(), 0.0f);
    EXPECT_FLOAT_EQ(points[0].position.y(), 0.0f);
    EXPECT_FLOAT_EQ(points[0].position.z(), 0.0f);
    EXPECT_EQ(points[0].color[0], 255);
    EXPECT_EQ(points[0].color[1], 0);
    EXPECT_EQ(points[0].color[2], 0);
    EXPECT_FLOAT_EQ(points[0].error, 0.5f);

    // Point 2: (1,0,0), green — skip track entries must work
    EXPECT_EQ(points[1].point_id, 2u);
    EXPECT_FLOAT_EQ(points[1].position.x(), 1.0f);
    EXPECT_EQ(points[1].color[1], 255);

    // Point 4: (0,0,1), white
    EXPECT_EQ(points[3].point_id, 4u);
    EXPECT_FLOAT_EQ(points[3].position.z(), 1.0f);
    EXPECT_EQ(points[3].color[0], 255);
    EXPECT_EQ(points[3].color[1], 255);
    EXPECT_EQ(points[3].color[2], 255);
}

TEST_F(ColmapLoaderTest, ParseColmapSparse_All) {
    auto data = cugs::parse_colmap_sparse(sparse_dir_);
    EXPECT_EQ(data.cameras.size(), 2u);
    EXPECT_EQ(data.images.size(), 3u);
    EXPECT_EQ(data.points.size(), 4u);
}

TEST_F(ColmapLoaderTest, MergeCamerasImages_Count) {
    auto data = cugs::parse_colmap_sparse(sparse_dir_);
    auto merged = cugs::merge_cameras_images(data.cameras, data.images);
    EXPECT_EQ(merged.size(), 3u);
}

TEST_F(ColmapLoaderTest, MergeCamerasImages_Intrinsics) {
    auto data = cugs::parse_colmap_sparse(sparse_dir_);
    auto merged = cugs::merge_cameras_images(data.cameras, data.images);

    // Images 1 and 2 use camera 1 (SIMPLE_PINHOLE: f=500)
    // Find the one with image_id=1
    const cugs::CameraInfo* cam1 = nullptr;
    for (const auto& c : merged) {
        if (c.image_id == 1) { cam1 = &c; break; }
    }
    ASSERT_NE(cam1, nullptr);
    EXPECT_FLOAT_EQ(cam1->intrinsics.fx, 500.0f);
    EXPECT_FLOAT_EQ(cam1->intrinsics.fy, 500.0f);  // SIMPLE_PINHOLE: fx==fy
    EXPECT_FLOAT_EQ(cam1->intrinsics.cx, 400.0f);
    EXPECT_FLOAT_EQ(cam1->intrinsics.cy, 300.0f);
    EXPECT_EQ(cam1->width, 800);
    EXPECT_EQ(cam1->height, 600);
}

TEST_F(ColmapLoaderTest, MergeCamerasImages_IdentityRotation) {
    auto data = cugs::parse_colmap_sparse(sparse_dir_);
    auto merged = cugs::merge_cameras_images(data.cameras, data.images);

    const cugs::CameraInfo* cam1 = nullptr;
    for (const auto& c : merged) {
        if (c.image_id == 1) { cam1 = &c; break; }
    }
    ASSERT_NE(cam1, nullptr);

    // Identity rotation
    Eigen::Matrix3f eye = Eigen::Matrix3f::Identity();
    EXPECT_TRUE(cam1->rotation.isApprox(eye, 1e-5f));

    // Camera center at origin (C = -R^T * t = -I * 0 = 0)
    Eigen::Vector3f center = cam1->camera_center();
    EXPECT_NEAR(center.x(), 0.0f, 1e-5f);
    EXPECT_NEAR(center.y(), 0.0f, 1e-5f);
    EXPECT_NEAR(center.z(), 0.0f, 1e-5f);
}

TEST_F(ColmapLoaderTest, MergeCamerasImages_CameraCenter) {
    auto data = cugs::parse_colmap_sparse(sparse_dir_);
    auto merged = cugs::merge_cameras_images(data.cameras, data.images);

    // Image 3: identity rotation, t=(5,0,0) → center = -I*(5,0,0) = (-5,0,0)
    const cugs::CameraInfo* cam3 = nullptr;
    for (const auto& c : merged) {
        if (c.image_id == 3) { cam3 = &c; break; }
    }
    ASSERT_NE(cam3, nullptr);
    Eigen::Vector3f center = cam3->camera_center();
    EXPECT_NEAR(center.x(), -5.0f, 1e-5f);
    EXPECT_NEAR(center.y(), 0.0f, 1e-5f);
    EXPECT_NEAR(center.z(), 0.0f, 1e-5f);
}

TEST_F(ColmapLoaderTest, MergeCamerasImages_RotatedCameraCenter) {
    auto data = cugs::parse_colmap_sparse(sparse_dir_);
    auto merged = cugs::merge_cameras_images(data.cameras, data.images);

    // Image 2: 90° around Y, t=(1,2,3)
    // R = Ry(90°), so R^T = Ry(-90°)
    // C = -R^T * t
    const cugs::CameraInfo* cam2 = nullptr;
    for (const auto& c : merged) {
        if (c.image_id == 2) { cam2 = &c; break; }
    }
    ASSERT_NE(cam2, nullptr);

    Eigen::Vector3f center = cam2->camera_center();
    // Ry(90°): [[0,0,1],[0,1,0],[-1,0,0]]
    // R^T = Ry(-90°): [[0,0,-1],[0,1,0],[1,0,0]]
    // C = -R^T * (1,2,3) = -((0*1+0*2-1*3), (0*1+1*2+0*3), (1*1+0*2+0*3))
    //   = -((-3), (2), (1)) = (3, -2, -1)
    EXPECT_NEAR(center.x(), 3.0f, 1e-4f);
    EXPECT_NEAR(center.y(), -2.0f, 1e-4f);
    EXPECT_NEAR(center.z(), -1.0f, 1e-4f);
}

TEST_F(ColmapLoaderTest, ParseCamerasBin_FileNotFound) {
    EXPECT_THROW(
        cugs::parse_cameras_bin(sparse_dir_ / "nonexistent.bin"),
        std::runtime_error);
}

// Test camera models with distortion params
TEST(ColmapLoaderCameraModels, SimpleRadial) {
    TempDir tmp;
    auto path = tmp.path() / "cameras.bin";

    {
        std::ofstream f(path, std::ios::binary);
        write_binary<uint64_t>(f, 1);
        write_binary<uint32_t>(f, 1);       // camera_id
        write_binary<uint32_t>(f, 2);       // model = SIMPLE_RADIAL
        write_binary<uint64_t>(f, 640);
        write_binary<uint64_t>(f, 480);
        write_binary<double>(f, 320.0);     // f
        write_binary<double>(f, 320.0);     // cx
        write_binary<double>(f, 240.0);     // cy
        write_binary<double>(f, 0.01);      // k1
    }

    auto cameras = cugs::parse_cameras_bin(path);
    ASSERT_EQ(cameras.size(), 1u);
    EXPECT_EQ(cameras[0].model, cugs::CameraModel::kSimpleRadial);
    EXPECT_EQ(cameras[0].params.size(), 4u);
    EXPECT_DOUBLE_EQ(cameras[0].params[3], 0.01);  // k1
}

TEST(ColmapLoaderCameraModels, SimpleRadialIntrinsics) {
    // Verify that SIMPLE_RADIAL extracts fx==fy==f and ignores distortion
    std::vector<cugs::ColmapCamera> cameras(1);
    cameras[0].camera_id = 1;
    cameras[0].model = cugs::CameraModel::kSimpleRadial;
    cameras[0].width = 640;
    cameras[0].height = 480;
    cameras[0].params = {320.0, 320.0, 240.0, 0.01};

    std::vector<cugs::ColmapImage> images(1);
    images[0].image_id = 1;
    images[0].qvec[0] = 1.0;
    images[0].camera_id = 1;
    images[0].name = "test.jpg";

    auto merged = cugs::merge_cameras_images(cameras, images);
    ASSERT_EQ(merged.size(), 1u);
    EXPECT_FLOAT_EQ(merged[0].intrinsics.fx, 320.0f);
    EXPECT_FLOAT_EQ(merged[0].intrinsics.fy, 320.0f);
}
