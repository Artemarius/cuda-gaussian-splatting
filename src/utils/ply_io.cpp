#include "utils/ply_io.hpp"

#include <spdlog/spdlog.h>

#include <fstream>

namespace cugs {

bool write_points_ply(const std::filesystem::path& path,
                      std::span<const SparsePoint> points) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        spdlog::error("Failed to open PLY file for writing: {}", path.string());
        return false;
    }

    // ASCII header
    ofs << "ply\n"
        << "format binary_little_endian 1.0\n"
        << "element vertex " << points.size() << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property uchar red\n"
        << "property uchar green\n"
        << "property uchar blue\n"
        << "end_header\n";

    // Binary vertex data
    for (const auto& pt : points) {
        const float x = pt.position.x();
        const float y = pt.position.y();
        const float z = pt.position.z();
        ofs.write(reinterpret_cast<const char*>(&x), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&y), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&z), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&pt.color[0]), 1);
        ofs.write(reinterpret_cast<const char*>(&pt.color[1]), 1);
        ofs.write(reinterpret_cast<const char*>(&pt.color[2]), 1);
    }

    spdlog::info("Wrote {} points to PLY: {}", points.size(), path.string());
    return true;
}

bool write_cameras_ply(const std::filesystem::path& path,
                       std::span<const CameraInfo> cameras,
                       uint8_t r, uint8_t g, uint8_t b) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        spdlog::error("Failed to open PLY file for writing: {}", path.string());
        return false;
    }

    // ASCII header
    ofs << "ply\n"
        << "format binary_little_endian 1.0\n"
        << "element vertex " << cameras.size() << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property uchar red\n"
        << "property uchar green\n"
        << "property uchar blue\n"
        << "end_header\n";

    // Binary vertex data — one point per camera center
    for (const auto& cam : cameras) {
        const Eigen::Vector3f c = cam.camera_center();
        const float x = c.x();
        const float y = c.y();
        const float z = c.z();
        ofs.write(reinterpret_cast<const char*>(&x), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&y), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&z), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&r), 1);
        ofs.write(reinterpret_cast<const char*>(&g), 1);
        ofs.write(reinterpret_cast<const char*>(&b), 1);
    }

    spdlog::info("Wrote {} camera centers to PLY: {}", cameras.size(),
                 path.string());
    return true;
}

} // namespace cugs
