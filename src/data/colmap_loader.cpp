#include "data/colmap_loader.hpp"

#include <spdlog/spdlog.h>

#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace cugs {

// ---------------------------------------------------------------------------
// Binary read helpers
// ---------------------------------------------------------------------------

/// @brief Read a single POD value from a binary stream (little-endian assumed).
template <typename T>
static T read_binary(std::ifstream& stream) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream) {
        throw std::runtime_error("Unexpected end of COLMAP binary file");
    }
    return value;
}

/// @brief Read a null-terminated string from a binary stream.
static std::string read_null_terminated_string(std::ifstream& stream) {
    std::string result;
    char c = 0;
    while (stream.get(c) && c != '\0') {
        result.push_back(c);
    }
    if (!stream) {
        throw std::runtime_error("Unexpected end of file reading null-terminated string");
    }
    return result;
}

// ---------------------------------------------------------------------------
// cameras.bin
// ---------------------------------------------------------------------------

std::vector<ColmapCamera> parse_cameras_bin(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open " + path.string());
    }

    const auto num_cameras = read_binary<uint64_t>(file);
    spdlog::debug("Parsing cameras.bin: {} cameras", num_cameras);

    std::vector<ColmapCamera> cameras;
    cameras.reserve(static_cast<size_t>(num_cameras));

    for (uint64_t i = 0; i < num_cameras; ++i) {
        ColmapCamera cam;
        cam.camera_id = read_binary<uint32_t>(file);
        auto model_id = read_binary<uint32_t>(file);
        cam.model = static_cast<CameraModel>(model_id);
        cam.width = read_binary<uint64_t>(file);
        cam.height = read_binary<uint64_t>(file);

        int num_params = camera_model_num_params(cam.model);
        cam.params.resize(static_cast<size_t>(num_params));
        for (int p = 0; p < num_params; ++p) {
            cam.params[static_cast<size_t>(p)] = read_binary<double>(file);
        }

        cameras.push_back(std::move(cam));
    }

    return cameras;
}

// ---------------------------------------------------------------------------
// images.bin
// ---------------------------------------------------------------------------

std::vector<ColmapImage> parse_images_bin(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open " + path.string());
    }

    const auto num_images = read_binary<uint64_t>(file);
    spdlog::debug("Parsing images.bin: {} images", num_images);

    std::vector<ColmapImage> images;
    images.reserve(static_cast<size_t>(num_images));

    for (uint64_t i = 0; i < num_images; ++i) {
        ColmapImage img;
        img.image_id = read_binary<uint32_t>(file);

        // Quaternion: wxyz (scalar-first)
        img.qvec[0] = read_binary<double>(file);
        img.qvec[1] = read_binary<double>(file);
        img.qvec[2] = read_binary<double>(file);
        img.qvec[3] = read_binary<double>(file);

        // Translation
        img.tvec[0] = read_binary<double>(file);
        img.tvec[1] = read_binary<double>(file);
        img.tvec[2] = read_binary<double>(file);

        img.camera_id = read_binary<uint32_t>(file);
        img.name = read_null_terminated_string(file);

        // Skip 2D point observations (we don't need them for training)
        const auto num_points2d = read_binary<uint64_t>(file);
        // Each 2D point: double x, double y, uint64_t point3d_id = 24 bytes
        constexpr size_t kBytes_per_point2d = 2 * sizeof(double) + sizeof(uint64_t);
        file.seekg(static_cast<std::streamoff>(num_points2d * kBytes_per_point2d),
                   std::ios::cur);

        if (!file) {
            throw std::runtime_error("Error reading images.bin at image " +
                                     std::to_string(i));
        }

        images.push_back(std::move(img));
    }

    return images;
}

// ---------------------------------------------------------------------------
// points3D.bin
// ---------------------------------------------------------------------------

std::vector<SparsePoint> parse_points3d_bin(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open " + path.string());
    }

    const auto num_points = read_binary<uint64_t>(file);
    spdlog::debug("Parsing points3D.bin: {} points", num_points);

    std::vector<SparsePoint> points;
    points.reserve(static_cast<size_t>(num_points));

    for (uint64_t i = 0; i < num_points; ++i) {
        SparsePoint pt;
        pt.point_id = read_binary<uint64_t>(file);

        double x = read_binary<double>(file);
        double y = read_binary<double>(file);
        double z = read_binary<double>(file);
        pt.position = Eigen::Vector3f(
            static_cast<float>(x),
            static_cast<float>(y),
            static_cast<float>(z));

        pt.color[0] = read_binary<uint8_t>(file);
        pt.color[1] = read_binary<uint8_t>(file);
        pt.color[2] = read_binary<uint8_t>(file);

        pt.error = static_cast<float>(read_binary<double>(file));

        // Skip track (we don't need per-point visibility for training)
        const auto track_len = read_binary<uint64_t>(file);
        // Each track element: uint32_t image_id + uint32_t point2d_idx = 8 bytes
        constexpr size_t kBytes_per_track = sizeof(uint32_t) + sizeof(uint32_t);
        file.seekg(static_cast<std::streamoff>(track_len * kBytes_per_track),
                   std::ios::cur);

        if (!file) {
            throw std::runtime_error("Error reading points3D.bin at point " +
                                     std::to_string(i));
        }

        points.push_back(std::move(pt));
    }

    return points;
}

// ---------------------------------------------------------------------------
// parse_colmap_sparse: convenience wrapper
// ---------------------------------------------------------------------------

ColmapData parse_colmap_sparse(const std::filesystem::path& sparse_dir) {
    ColmapData data;
    data.cameras = parse_cameras_bin(sparse_dir / "cameras.bin");
    data.images = parse_images_bin(sparse_dir / "images.bin");
    data.points = parse_points3d_bin(sparse_dir / "points3D.bin");

    spdlog::info("Loaded COLMAP sparse: {} cameras, {} images, {} points",
                 data.cameras.size(), data.images.size(), data.points.size());

    return data;
}

// ---------------------------------------------------------------------------
// merge_cameras_images
// ---------------------------------------------------------------------------

/// @brief Extract normalised (fx, fy, cx, cy) from a ColmapCamera.
static CameraIntrinsics extract_intrinsics(const ColmapCamera& cam) {
    CameraIntrinsics intr;
    switch (cam.model) {
        case CameraModel::kSimplePinhole:
            // params: f, cx, cy
            intr.fx = static_cast<float>(cam.params[0]);
            intr.fy = static_cast<float>(cam.params[0]);
            intr.cx = static_cast<float>(cam.params[1]);
            intr.cy = static_cast<float>(cam.params[2]);
            break;
        case CameraModel::kPinhole:
            // params: fx, fy, cx, cy
            intr.fx = static_cast<float>(cam.params[0]);
            intr.fy = static_cast<float>(cam.params[1]);
            intr.cx = static_cast<float>(cam.params[2]);
            intr.cy = static_cast<float>(cam.params[3]);
            break;
        case CameraModel::kSimpleRadial:
            // params: f, cx, cy, k1 — distortion ignored
            intr.fx = static_cast<float>(cam.params[0]);
            intr.fy = static_cast<float>(cam.params[0]);
            intr.cx = static_cast<float>(cam.params[1]);
            intr.cy = static_cast<float>(cam.params[2]);
            break;
        case CameraModel::kRadial:
            // params: f, cx, cy, k1, k2 — distortion ignored
            intr.fx = static_cast<float>(cam.params[0]);
            intr.fy = static_cast<float>(cam.params[0]);
            intr.cx = static_cast<float>(cam.params[1]);
            intr.cy = static_cast<float>(cam.params[2]);
            break;
        case CameraModel::kOpenCV:
            // params: fx, fy, cx, cy, k1, k2, p1, p2 — distortion ignored
            intr.fx = static_cast<float>(cam.params[0]);
            intr.fy = static_cast<float>(cam.params[1]);
            intr.cx = static_cast<float>(cam.params[2]);
            intr.cy = static_cast<float>(cam.params[3]);
            break;
    }
    return intr;
}

std::vector<CameraInfo> merge_cameras_images(
    const std::vector<ColmapCamera>& cameras,
    const std::vector<ColmapImage>& images) {

    // Build camera lookup by ID
    std::unordered_map<uint32_t, const ColmapCamera*> cam_map;
    for (const auto& cam : cameras) {
        cam_map[cam.camera_id] = &cam;
    }

    std::vector<CameraInfo> result;
    result.reserve(images.size());

    for (const auto& img : images) {
        auto it = cam_map.find(img.camera_id);
        if (it == cam_map.end()) {
            throw std::runtime_error(
                "Image '" + img.name + "' references camera ID " +
                std::to_string(img.camera_id) + " which was not found");
        }
        const ColmapCamera& cam = *it->second;

        CameraInfo info;
        info.image_id = img.image_id;
        info.camera_id = img.camera_id;
        info.width = static_cast<int>(cam.width);
        info.height = static_cast<int>(cam.height);
        info.intrinsics = extract_intrinsics(cam);
        info.rotation = qvec_to_rotation(img.qvec[0], img.qvec[1],
                                         img.qvec[2], img.qvec[3]);
        info.translation = Eigen::Vector3f(
            static_cast<float>(img.tvec[0]),
            static_cast<float>(img.tvec[1]),
            static_cast<float>(img.tvec[2]));
        info.image_name = img.name;

        result.push_back(std::move(info));
    }

    return result;
}

} // namespace cugs
