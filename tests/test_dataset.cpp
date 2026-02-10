#include <gtest/gtest.h>

#include "core/types.hpp"
#include "data/dataset.hpp"
#include "data/image_io.hpp"

#include <Eigen/Core>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Binary write helpers (duplicated from test_colmap_loader.cpp for isolation)
// ---------------------------------------------------------------------------

template <typename T>
void write_binary(std::ofstream& out, T value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void write_null_terminated(std::ofstream& out, const std::string& str) {
    out.write(str.c_str(), static_cast<std::streamsize>(str.size() + 1));
}

// ---------------------------------------------------------------------------
// Create a synthetic dataset directory with COLMAP data and dummy images
// ---------------------------------------------------------------------------

/// @brief Create a minimal valid PPM image file (binary, RGB).
///        stb_image can read PPM files.
void create_dummy_ppm(const std::filesystem::path& path, int w, int h,
                      uint8_t r, uint8_t g, uint8_t b) {
    std::ofstream f(path, std::ios::binary);
    // PPM header: "P6\n<width> <height>\n255\n"
    std::string header = "P6\n" + std::to_string(w) + " " + std::to_string(h) + "\n255\n";
    f.write(header.c_str(), static_cast<std::streamsize>(header.size()));
    for (int i = 0; i < w * h; ++i) {
        f.put(static_cast<char>(r));
        f.put(static_cast<char>(g));
        f.put(static_cast<char>(b));
    }
}

/// @brief Create a complete synthetic dataset with 16 images.
///        Images are named img_00.ppm through img_15.ppm (sorted).
///        All use camera 1 (SIMPLE_PINHOLE), identity rotation, increasing x-translation.
void create_synthetic_dataset(const std::filesystem::path& base_path) {
    // Create directories
    auto sparse_dir = base_path / "sparse" / "0";
    auto images_dir = base_path / "images";
    std::filesystem::create_directories(sparse_dir);
    std::filesystem::create_directories(images_dir);

    constexpr int kNumImages = 16;

    // cameras.bin: one SIMPLE_PINHOLE camera
    {
        std::ofstream f(sparse_dir / "cameras.bin", std::ios::binary);
        write_binary<uint64_t>(f, 1);
        write_binary<uint32_t>(f, 1);       // camera_id
        write_binary<uint32_t>(f, 0);       // SIMPLE_PINHOLE
        write_binary<uint64_t>(f, 64);      // width
        write_binary<uint64_t>(f, 48);      // height
        write_binary<double>(f, 50.0);      // f
        write_binary<double>(f, 32.0);      // cx
        write_binary<double>(f, 24.0);      // cy
    }

    // images.bin: 16 images with increasing x-translation
    {
        std::ofstream f(sparse_dir / "images.bin", std::ios::binary);
        write_binary<uint64_t>(f, kNumImages);
        for (int i = 0; i < kNumImages; ++i) {
            char name[16];
            snprintf(name, sizeof(name), "img_%02d.ppm", i);

            write_binary<uint32_t>(f, static_cast<uint32_t>(i + 1));
            // Identity quaternion
            write_binary<double>(f, 1.0);
            write_binary<double>(f, 0.0);
            write_binary<double>(f, 0.0);
            write_binary<double>(f, 0.0);
            // Translation: x increases
            write_binary<double>(f, static_cast<double>(i));
            write_binary<double>(f, 0.0);
            write_binary<double>(f, 0.0);
            write_binary<uint32_t>(f, 1);  // camera_id
            write_null_terminated(f, std::string(name));
            write_binary<uint64_t>(f, 0);  // no 2D points
        }
    }

    // points3D.bin: 8 points forming a unit cube
    {
        std::ofstream f(sparse_dir / "points3D.bin", std::ios::binary);
        write_binary<uint64_t>(f, 8);
        double coords[8][3] = {
            {0,0,0}, {1,0,0}, {0,1,0}, {1,1,0},
            {0,0,1}, {1,0,1}, {0,1,1}, {1,1,1}
        };
        for (int i = 0; i < 8; ++i) {
            write_binary<uint64_t>(f, static_cast<uint64_t>(i + 1));
            write_binary<double>(f, coords[i][0]);
            write_binary<double>(f, coords[i][1]);
            write_binary<double>(f, coords[i][2]);
            write_binary<uint8_t>(f, 128);  // r
            write_binary<uint8_t>(f, 128);  // g
            write_binary<uint8_t>(f, 128);  // b
            write_binary<double>(f, 0.5);   // error
            write_binary<uint64_t>(f, 0);   // no track
        }
    }

    // Create dummy image files (small PPM images)
    for (int i = 0; i < kNumImages; ++i) {
        char name[16];
        snprintf(name, sizeof(name), "img_%02d.ppm", i);
        uint8_t shade = static_cast<uint8_t>(i * 16);
        create_dummy_ppm(images_dir / name, 64, 48, shade, shade, shade);
    }
}

class TempDir {
public:
    TempDir() {
        path_ = std::filesystem::temp_directory_path() / "cugs_test_dataset";
        // Clean any previous run
        std::filesystem::remove_all(path_);
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

class DatasetTest : public ::testing::Test {
protected:
    void SetUp() override {
        create_synthetic_dataset(tmp_.path());
    }

    TempDir tmp_;
};

TEST_F(DatasetTest, Constructor_LoadsCorrectly) {
    cugs::Dataset ds(tmp_.path());
    // 16 images, test_every_n=8: indices 0,8 are test → 2 test, 14 train
    EXPECT_EQ(ds.num_train(), 14u);
    EXPECT_EQ(ds.num_test(), 2u);
    EXPECT_EQ(ds.num_points(), 8u);
}

TEST_F(DatasetTest, TrainTestSplit_EveryEighth) {
    cugs::Dataset ds(tmp_.path(), 1, 8);
    // Sorted images: img_00..img_15, indices 0,8 → test
    EXPECT_EQ(ds.num_test(), 2u);
    EXPECT_EQ(ds.num_train(), 14u);

    // Verify test images are the expected ones
    const auto& test_cams = ds.test_cameras();
    EXPECT_EQ(test_cams[0].image_name, "img_00.ppm");
    EXPECT_EQ(test_cams[1].image_name, "img_08.ppm");
}

TEST_F(DatasetTest, TrainTestSplit_EveryFourth) {
    cugs::Dataset ds(tmp_.path(), 1, 4);
    // indices 0,4,8,12 → test (4 test, 12 train)
    EXPECT_EQ(ds.num_test(), 4u);
    EXPECT_EQ(ds.num_train(), 12u);
}

TEST_F(DatasetTest, TrainTestSplit_NoTest) {
    cugs::Dataset ds(tmp_.path(), 1, 0);
    // All images to train, none to test
    EXPECT_EQ(ds.num_train(), 16u);
    EXPECT_EQ(ds.num_test(), 0u);
}

TEST_F(DatasetTest, SceneBounds_Reasonable) {
    cugs::Dataset ds(tmp_.path());
    const auto& bounds = ds.scene_bounds();

    // Points are in [0,1]^3, cameras have centers at (-i, 0, 0) for i=0..15
    // So min_bound.x should be around -15, max_bound.x around 1
    EXPECT_LE(bounds.min_bound.x(), 0.0f);
    EXPECT_GE(bounds.max_bound.x(), 1.0f);

    // Y: points in [0,1], cameras at y=0
    EXPECT_LE(bounds.min_bound.y(), 0.0f);
    EXPECT_GE(bounds.max_bound.y(), 1.0f);

    // Extent should be positive
    EXPECT_GT(bounds.extent, 0.0f);
}

TEST_F(DatasetTest, SceneBounds_Center) {
    cugs::Dataset ds(tmp_.path());
    const auto& bounds = ds.scene_bounds();

    // Center should be roughly midway
    Eigen::Vector3f expected_center = (bounds.min_bound + bounds.max_bound) * 0.5f;
    EXPECT_NEAR(bounds.center.x(), expected_center.x(), 1e-5f);
    EXPECT_NEAR(bounds.center.y(), expected_center.y(), 1e-5f);
    EXPECT_NEAR(bounds.center.z(), expected_center.z(), 1e-5f);
}

TEST_F(DatasetTest, LazyImageLoading_Train) {
    cugs::Dataset ds(tmp_.path());

    auto img = ds.load_train_image(0);
    EXPECT_TRUE(img.valid());
    EXPECT_EQ(img.width, 64);
    EXPECT_EQ(img.height, 48);
    EXPECT_EQ(img.channels, 3);
    EXPECT_EQ(img.data.size(), static_cast<size_t>(64 * 48 * 3));
}

TEST_F(DatasetTest, LazyImageLoading_Test) {
    cugs::Dataset ds(tmp_.path());

    auto img = ds.load_test_image(0);
    EXPECT_TRUE(img.valid());
    EXPECT_EQ(img.width, 64);
    EXPECT_EQ(img.height, 48);
}

TEST_F(DatasetTest, LazyImageLoading_OutOfBounds) {
    cugs::Dataset ds(tmp_.path());
    EXPECT_THROW(ds.load_train_image(999), std::runtime_error);
    EXPECT_THROW(ds.load_test_image(999), std::runtime_error);
}

TEST_F(DatasetTest, ResolutionScale_HalvesIntrinsics) {
    cugs::Dataset ds(tmp_.path(), 2);

    const auto& cam = ds.train_cameras()[0];
    // Original: 64x48, f=50, cx=32, cy=24
    // At scale 2: 32x24, f=25, cx=16, cy=12
    EXPECT_EQ(cam.width, 32);
    EXPECT_EQ(cam.height, 24);
    EXPECT_FLOAT_EQ(cam.intrinsics.fx, 25.0f);
    EXPECT_FLOAT_EQ(cam.intrinsics.fy, 25.0f);
    EXPECT_FLOAT_EQ(cam.intrinsics.cx, 16.0f);
    EXPECT_FLOAT_EQ(cam.intrinsics.cy, 12.0f);
}

TEST_F(DatasetTest, CameraInfoFields) {
    cugs::Dataset ds(tmp_.path());

    const auto& cam = ds.train_cameras()[0];
    // Verify basic fields are populated
    EXPECT_GT(cam.image_id, 0u);
    EXPECT_GT(cam.camera_id, 0u);
    EXPECT_FALSE(cam.image_name.empty());
    EXPECT_FALSE(cam.image_path.empty());
    EXPECT_EQ(cam.width, 64);
    EXPECT_EQ(cam.height, 48);
}

TEST_F(DatasetTest, SparsePoints_CorrectValues) {
    cugs::Dataset ds(tmp_.path());
    const auto& pts = ds.sparse_points();

    EXPECT_EQ(pts.size(), 8u);
    // All points have color (128,128,128)
    for (const auto& pt : pts) {
        EXPECT_EQ(pt.color[0], 128);
        EXPECT_EQ(pt.color[1], 128);
        EXPECT_EQ(pt.color[2], 128);
    }
}

TEST_F(DatasetTest, PrintSummary_DoesNotThrow) {
    cugs::Dataset ds(tmp_.path());
    EXPECT_NO_THROW(ds.print_summary());
}

TEST_F(DatasetTest, NonexistentPath_Throws) {
    EXPECT_THROW(
        cugs::Dataset(tmp_.path() / "nonexistent"),
        std::runtime_error);
}

// ---------------------------------------------------------------------------
// Image I/O tests (standalone, no dataset needed)
// ---------------------------------------------------------------------------

TEST(ImageIO, LoadPPM) {
    TempDir tmp;
    auto path = tmp.path() / "test.ppm";
    create_dummy_ppm(path, 4, 3, 255, 0, 128);

    auto img = cugs::load_image(path);
    EXPECT_EQ(img.width, 4);
    EXPECT_EQ(img.height, 3);
    EXPECT_EQ(img.channels, 3);
    EXPECT_EQ(img.data.size(), static_cast<size_t>(4 * 3 * 3));

    // First pixel should be (1.0, 0.0, ~0.502)
    EXPECT_NEAR(img.data[0], 1.0f, 1e-3f);   // R
    EXPECT_NEAR(img.data[1], 0.0f, 1e-3f);   // G
    EXPECT_NEAR(img.data[2], 128.0f / 255.0f, 1e-3f);  // B
}

TEST(ImageIO, ResizeImage) {
    cugs::Image src;
    src.width = 4;
    src.height = 4;
    src.channels = 3;
    src.data.resize(4 * 4 * 3, 0.5f);  // uniform gray

    auto dst = cugs::resize_image(src, 2, 2);
    EXPECT_EQ(dst.width, 2);
    EXPECT_EQ(dst.height, 2);
    EXPECT_EQ(dst.channels, 3);
    // Uniform input → uniform output
    for (float v : dst.data) {
        EXPECT_NEAR(v, 0.5f, 1e-3f);
    }
}

TEST(ImageIO, LoadImageResized_NoScale) {
    TempDir tmp;
    auto path = tmp.path() / "test.ppm";
    create_dummy_ppm(path, 8, 6, 100, 100, 100);

    auto img = cugs::load_image_resized(path, 1);
    EXPECT_EQ(img.width, 8);
    EXPECT_EQ(img.height, 6);
}

TEST(ImageIO, LoadImageResized_HalfScale) {
    TempDir tmp;
    auto path = tmp.path() / "test.ppm";
    create_dummy_ppm(path, 8, 6, 100, 100, 100);

    auto img = cugs::load_image_resized(path, 2);
    EXPECT_EQ(img.width, 4);
    EXPECT_EQ(img.height, 3);
}

TEST(ImageIO, LoadImage_FileNotFound) {
    EXPECT_THROW(
        cugs::load_image("nonexistent_file.png"),
        std::runtime_error);
}
