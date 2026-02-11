#include "utils/ply_io.hpp"
#include "core/gaussian.hpp"

#include <spdlog/spdlog.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

// ---------------------------------------------------------------------------
// Gaussian model PLY writer
// ---------------------------------------------------------------------------

bool write_gaussian_ply(const std::filesystem::path& path,
                        const GaussianModel& model) {
    if (!model.is_valid()) {
        spdlog::error("Cannot write invalid GaussianModel to PLY");
        return false;
    }

    // Move tensors to CPU, contiguous
    auto positions = model.positions.cpu().contiguous().to(torch::kFloat32);
    auto sh_coeffs = model.sh_coeffs.cpu().contiguous().to(torch::kFloat32);
    auto opacities = model.opacities.cpu().contiguous().to(torch::kFloat32);
    auto scales    = model.scales.cpu().contiguous().to(torch::kFloat32);
    auto rotations = model.rotations.cpu().contiguous().to(torch::kFloat32);

    const int64_t n = positions.size(0);
    const int num_coeffs = static_cast<int>(sh_coeffs.size(2));
    // DC coefficients (3) + rest coefficients (3 * (num_coeffs - 1))
    const int num_rest = 3 * (num_coeffs - 1);

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        spdlog::error("Failed to open PLY file for writing: {}", path.string());
        return false;
    }

    // -- Write ASCII header (reference implementation format) --
    ofs << "ply\n"
        << "format binary_little_endian 1.0\n"
        << "element vertex " << n << "\n"
        << "property float x\n"
        << "property float y\n"
        << "property float z\n"
        << "property float nx\n"
        << "property float ny\n"
        << "property float nz\n";

    // DC SH coefficients (one per color channel)
    for (int i = 0; i < 3; ++i)
        ofs << "property float f_dc_" << i << "\n";

    // Higher-order SH coefficients (interleaved: all channels for each coeff)
    for (int i = 0; i < num_rest; ++i)
        ofs << "property float f_rest_" << i << "\n";

    ofs << "property float opacity\n"
        << "property float scale_0\n"
        << "property float scale_1\n"
        << "property float scale_2\n"
        << "property float rot_0\n"
        << "property float rot_1\n"
        << "property float rot_2\n"
        << "property float rot_3\n"
        << "end_header\n";

    // -- Write binary data --
    auto pos_acc = positions.accessor<float, 2>();
    auto sh_acc  = sh_coeffs.accessor<float, 3>();  // [N, 3, C]
    auto opa_acc = opacities.accessor<float, 2>();
    auto scl_acc = scales.accessor<float, 2>();
    auto rot_acc = rotations.accessor<float, 2>();

    const float zero = 0.0f;

    for (int64_t i = 0; i < n; ++i) {
        // Position (x, y, z)
        ofs.write(reinterpret_cast<const char*>(&pos_acc[i][0]), 3 * sizeof(float));

        // Normals (nx, ny, nz) — always zero for compatibility
        ofs.write(reinterpret_cast<const char*>(&zero), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&zero), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&zero), sizeof(float));

        // DC SH coefficients: f_dc_0 (R), f_dc_1 (G), f_dc_2 (B)
        for (int ch = 0; ch < 3; ++ch) {
            ofs.write(reinterpret_cast<const char*>(&sh_acc[i][ch][0]), sizeof(float));
        }

        // Higher-order SH coefficients (reference format: interleaved)
        // Layout: for coeff k in [1..num_coeffs-1], write ch0, ch1, ch2
        for (int k = 1; k < num_coeffs; ++k) {
            for (int ch = 0; ch < 3; ++ch) {
                ofs.write(reinterpret_cast<const char*>(&sh_acc[i][ch][k]), sizeof(float));
            }
        }

        // Opacity
        ofs.write(reinterpret_cast<const char*>(&opa_acc[i][0]), sizeof(float));

        // Scale (log-space)
        ofs.write(reinterpret_cast<const char*>(&scl_acc[i][0]), 3 * sizeof(float));

        // Rotation (quaternion wxyz)
        ofs.write(reinterpret_cast<const char*>(&rot_acc[i][0]), 4 * sizeof(float));
    }

    spdlog::info("Wrote {} Gaussians to PLY: {} (SH coeffs: {})",
                 n, path.string(), num_coeffs);
    return true;
}

// ---------------------------------------------------------------------------
// Gaussian model PLY reader
// ---------------------------------------------------------------------------

namespace {

/// @brief Parse a PLY header and return the property names in order.
struct PlyHeader {
    int64_t vertex_count = 0;
    std::vector<std::string> property_names;
    std::streampos data_offset = 0;
};

PlyHeader parse_ply_header(std::ifstream& ifs) {
    PlyHeader header;
    std::string line;

    // First line must be "ply"
    std::getline(ifs, line);
    if (line.find("ply") == std::string::npos)
        throw std::runtime_error("Not a PLY file");

    // Second line: format
    std::getline(ifs, line);
    if (line.find("binary_little_endian") == std::string::npos)
        throw std::runtime_error("Only binary_little_endian PLY is supported");

    while (std::getline(ifs, line)) {
        // Trim carriage return if present (Windows line endings)
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (line == "end_header") {
            header.data_offset = ifs.tellg();
            break;
        }

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "element") {
            std::string elem_type;
            int64_t count;
            iss >> elem_type >> count;
            if (elem_type == "vertex") {
                header.vertex_count = count;
            }
        } else if (token == "property") {
            std::string dtype, name;
            iss >> dtype >> name;
            header.property_names.push_back(name);
        }
    }

    return header;
}

} // anonymous namespace

GaussianModel read_gaussian_ply(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        throw std::runtime_error("Failed to open PLY file: " + path.string());

    auto header = parse_ply_header(ifs);
    const int64_t n = header.vertex_count;

    // Build property name -> index map
    std::unordered_map<std::string, int> prop_index;
    for (int i = 0; i < static_cast<int>(header.property_names.size()); ++i) {
        prop_index[header.property_names[i]] = i;
    }

    const int num_props = static_cast<int>(header.property_names.size());

    // Count SH rest coefficients
    int num_rest = 0;
    while (prop_index.count("f_rest_" + std::to_string(num_rest))) {
        ++num_rest;
    }
    // Total SH coefficients per channel: 1 (DC) + num_rest/3 (higher order)
    const int num_sh_coeffs = 1 + num_rest / 3;

    spdlog::info("Reading PLY: {} vertices, {} properties, {} SH coeffs/channel",
                 n, num_props, num_sh_coeffs);

    // Read all vertex data as flat float buffer (all properties are float)
    std::vector<float> buffer(n * num_props);
    ifs.seekg(header.data_offset);
    ifs.read(reinterpret_cast<char*>(buffer.data()),
             n * num_props * sizeof(float));
    if (!ifs)
        throw std::runtime_error("Failed to read PLY binary data");

    // Helper to get property value
    auto get_prop = [&](int64_t vertex_idx, const std::string& name) -> float {
        auto it = prop_index.find(name);
        if (it == prop_index.end())
            throw std::runtime_error("Missing PLY property: " + name);
        return buffer[vertex_idx * num_props + it->second];
    };

    // Allocate tensors
    GaussianModel model;
    model.positions = torch::zeros({n, 3}, torch::kFloat32);
    model.sh_coeffs = torch::zeros({n, 3, num_sh_coeffs}, torch::kFloat32);
    model.opacities = torch::zeros({n, 1}, torch::kFloat32);
    model.scales    = torch::zeros({n, 3}, torch::kFloat32);
    model.rotations = torch::zeros({n, 4}, torch::kFloat32);

    auto pos_acc = model.positions.accessor<float, 2>();
    auto sh_acc  = model.sh_coeffs.accessor<float, 3>();
    auto opa_acc = model.opacities.accessor<float, 2>();
    auto scl_acc = model.scales.accessor<float, 2>();
    auto rot_acc = model.rotations.accessor<float, 2>();

    for (int64_t i = 0; i < n; ++i) {
        // Position
        pos_acc[i][0] = get_prop(i, "x");
        pos_acc[i][1] = get_prop(i, "y");
        pos_acc[i][2] = get_prop(i, "z");

        // DC SH coefficients
        for (int ch = 0; ch < 3; ++ch) {
            sh_acc[i][ch][0] = get_prop(i, "f_dc_" + std::to_string(ch));
        }

        // Higher-order SH coefficients (interleaved in PLY)
        for (int k = 1; k < num_sh_coeffs; ++k) {
            for (int ch = 0; ch < 3; ++ch) {
                int rest_idx = (k - 1) * 3 + ch;
                sh_acc[i][ch][k] = get_prop(i, "f_rest_" + std::to_string(rest_idx));
            }
        }

        // Opacity
        opa_acc[i][0] = get_prop(i, "opacity");

        // Scale
        scl_acc[i][0] = get_prop(i, "scale_0");
        scl_acc[i][1] = get_prop(i, "scale_1");
        scl_acc[i][2] = get_prop(i, "scale_2");

        // Rotation
        rot_acc[i][0] = get_prop(i, "rot_0");
        rot_acc[i][1] = get_prop(i, "rot_1");
        rot_acc[i][2] = get_prop(i, "rot_2");
        rot_acc[i][3] = get_prop(i, "rot_3");
    }

    spdlog::info("Loaded {} Gaussians from PLY: {}", n, path.string());
    return model;
}

} // namespace cugs
