#include "core/types.hpp"
#include "utils/ply_io.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

/// @brief Helper to read the full contents of a binary file.
static std::string read_file(const fs::path& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    EXPECT_TRUE(ifs.good());
    const auto size = ifs.tellg();
    ifs.seekg(0);
    std::string buf(static_cast<size_t>(size), '\0');
    ifs.read(buf.data(), size);
    return buf;
}

// ---------------------------------------------------------------------------
// write_points_ply
// ---------------------------------------------------------------------------

TEST(PlyIo, WritePointsPly_EmptyPoints) {
    const auto dir = fs::temp_directory_path() / "cugs_ply_test_empty";
    fs::create_directories(dir);
    const auto path = dir / "empty.ply";

    const std::vector<cugs::SparsePoint> points;
    EXPECT_TRUE(cugs::write_points_ply(path, points));
    EXPECT_TRUE(fs::exists(path));

    // File should have a header with 0 vertices and no binary data after it.
    const auto data = read_file(path);
    EXPECT_NE(data.find("element vertex 0"), std::string::npos);

    fs::remove_all(dir);
}

TEST(PlyIo, WritePointsPly_CorrectBinaryLayout) {
    const auto dir = fs::temp_directory_path() / "cugs_ply_test_points";
    fs::create_directories(dir);
    const auto path = dir / "points.ply";

    std::vector<cugs::SparsePoint> points(2);
    points[0].position = Eigen::Vector3f(1.0f, 2.0f, 3.0f);
    points[0].color[0] = 10;
    points[0].color[1] = 20;
    points[0].color[2] = 30;
    points[1].position = Eigen::Vector3f(4.0f, 5.0f, 6.0f);
    points[1].color[0] = 40;
    points[1].color[1] = 50;
    points[1].color[2] = 60;

    EXPECT_TRUE(cugs::write_points_ply(path, points));

    // Read back and verify binary payload
    const auto data = read_file(path);
    EXPECT_NE(data.find("element vertex 2"), std::string::npos);

    // Find end_header to locate the start of binary data
    const auto header_end = data.find("end_header\n");
    ASSERT_NE(header_end, std::string::npos);
    const auto payload_start = header_end + std::string("end_header\n").size();

    // Each vertex: 3 floats (12 bytes) + 3 bytes RGB = 15 bytes
    const size_t vertex_size = 3 * sizeof(float) + 3;
    ASSERT_EQ(data.size() - payload_start, 2 * vertex_size);

    // Verify first vertex
    const char* v0 = data.data() + payload_start;
    float x0, y0, z0;
    std::memcpy(&x0, v0 + 0, sizeof(float));
    std::memcpy(&y0, v0 + 4, sizeof(float));
    std::memcpy(&z0, v0 + 8, sizeof(float));
    EXPECT_FLOAT_EQ(x0, 1.0f);
    EXPECT_FLOAT_EQ(y0, 2.0f);
    EXPECT_FLOAT_EQ(z0, 3.0f);
    EXPECT_EQ(static_cast<uint8_t>(v0[12]), 10);
    EXPECT_EQ(static_cast<uint8_t>(v0[13]), 20);
    EXPECT_EQ(static_cast<uint8_t>(v0[14]), 30);

    // Verify second vertex
    const char* v1 = data.data() + payload_start + vertex_size;
    float x1, y1, z1;
    std::memcpy(&x1, v1 + 0, sizeof(float));
    std::memcpy(&y1, v1 + 4, sizeof(float));
    std::memcpy(&z1, v1 + 8, sizeof(float));
    EXPECT_FLOAT_EQ(x1, 4.0f);
    EXPECT_FLOAT_EQ(y1, 5.0f);
    EXPECT_FLOAT_EQ(z1, 6.0f);
    EXPECT_EQ(static_cast<uint8_t>(v1[12]), 40);
    EXPECT_EQ(static_cast<uint8_t>(v1[13]), 50);
    EXPECT_EQ(static_cast<uint8_t>(v1[14]), 60);

    fs::remove_all(dir);
}

TEST(PlyIo, WritePointsPly_HeaderIsValidAscii) {
    const auto dir = fs::temp_directory_path() / "cugs_ply_test_header";
    fs::create_directories(dir);
    const auto path = dir / "header.ply";

    std::vector<cugs::SparsePoint> points(1);
    points[0].position = Eigen::Vector3f::Zero();
    EXPECT_TRUE(cugs::write_points_ply(path, points));

    const auto data = read_file(path);
    // Check all required PLY header lines
    EXPECT_EQ(data.substr(0, 4), "ply\n");
    EXPECT_NE(data.find("format binary_little_endian 1.0"), std::string::npos);
    EXPECT_NE(data.find("property float x"), std::string::npos);
    EXPECT_NE(data.find("property float y"), std::string::npos);
    EXPECT_NE(data.find("property float z"), std::string::npos);
    EXPECT_NE(data.find("property uchar red"), std::string::npos);
    EXPECT_NE(data.find("property uchar green"), std::string::npos);
    EXPECT_NE(data.find("property uchar blue"), std::string::npos);
    EXPECT_NE(data.find("end_header"), std::string::npos);

    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// write_cameras_ply
// ---------------------------------------------------------------------------

TEST(PlyIo, WriteCamerasPly_CorrectPositionAndColor) {
    const auto dir = fs::temp_directory_path() / "cugs_ply_test_cameras";
    fs::create_directories(dir);
    const auto path = dir / "cameras.ply";

    // Create a camera at known position: camera_center = -R^T * t
    // With identity rotation, center = -t, so set t = (-5, -6, -7) → center = (5, 6, 7)
    std::vector<cugs::CameraInfo> cameras(1);
    cameras[0].rotation = Eigen::Matrix3f::Identity();
    cameras[0].translation = Eigen::Vector3f(-5.0f, -6.0f, -7.0f);

    EXPECT_TRUE(cugs::write_cameras_ply(path, cameras, 0, 255, 0));

    const auto data = read_file(path);
    EXPECT_NE(data.find("element vertex 1"), std::string::npos);

    const auto header_end = data.find("end_header\n");
    ASSERT_NE(header_end, std::string::npos);
    const auto payload_start = header_end + std::string("end_header\n").size();

    const char* v = data.data() + payload_start;
    float x, y, z;
    std::memcpy(&x, v + 0, sizeof(float));
    std::memcpy(&y, v + 4, sizeof(float));
    std::memcpy(&z, v + 8, sizeof(float));
    EXPECT_FLOAT_EQ(x, 5.0f);
    EXPECT_FLOAT_EQ(y, 6.0f);
    EXPECT_FLOAT_EQ(z, 7.0f);
    // Check green color
    EXPECT_EQ(static_cast<uint8_t>(v[12]), 0);
    EXPECT_EQ(static_cast<uint8_t>(v[13]), 255);
    EXPECT_EQ(static_cast<uint8_t>(v[14]), 0);

    fs::remove_all(dir);
}

TEST(PlyIo, WritePointsPly_InvalidPath) {
    const fs::path bad_path{"Z:/nonexistent_drive/no_such_dir/bad.ply"};
    const std::vector<cugs::SparsePoint> points;
    EXPECT_FALSE(cugs::write_points_ply(bad_path, points));
}
