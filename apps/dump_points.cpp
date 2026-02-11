/// @file dump_points.cpp
/// @brief Load a COLMAP dataset and dump sparse points + camera centers to PLY
///        for visual verification in MeshLab or CloudCompare.
///
/// Usage:
///   dump_points <dataset_path> [output_dir]
///
/// Outputs:
///   <output_dir>/sparse_points.ply   — colored sparse 3D points
///   <output_dir>/camera_centers.ply  — camera positions (red = train, blue = test)

#include "data/dataset.hpp"
#include "utils/ply_io.hpp"

#include <spdlog/spdlog.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        spdlog::error("Usage: dump_points <dataset_path> [output_dir]");
        return EXIT_FAILURE;
    }

    const std::filesystem::path dataset_path{argv[1]};
    const std::filesystem::path output_dir =
        (argc >= 3) ? std::filesystem::path{argv[2]} : dataset_path;

    // Load the COLMAP dataset
    spdlog::info("Loading dataset from: {}", dataset_path.string());
    cugs::Dataset dataset(dataset_path);
    dataset.print_summary();

    // Ensure output directory exists
    std::filesystem::create_directories(output_dir);

    // Dump sparse points
    const auto points_path = output_dir / "sparse_points.ply";
    if (!cugs::write_points_ply(points_path, dataset.sparse_points())) {
        return EXIT_FAILURE;
    }

    // Dump train camera centers (red)
    const auto train_cam_path = output_dir / "cameras_train.ply";
    if (!cugs::write_cameras_ply(train_cam_path, dataset.train_cameras(),
                                 255, 0, 0)) {
        return EXIT_FAILURE;
    }

    // Dump test camera centers (blue)
    if (dataset.num_test() > 0) {
        const auto test_cam_path = output_dir / "cameras_test.ply";
        if (!cugs::write_cameras_ply(test_cam_path, dataset.test_cameras(),
                                     0, 0, 255)) {
            return EXIT_FAILURE;
        }
    }

    spdlog::info("Done. Open the PLY files in MeshLab or CloudCompare.");
    return EXIT_SUCCESS;
}
