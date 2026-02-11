#include "core/gaussian.hpp"
#include "utils/ply_io.hpp"

namespace cugs {

bool GaussianModel::save_ply(const std::filesystem::path& path) const {
    return write_gaussian_ply(path, *this);
}

GaussianModel GaussianModel::load_ply(const std::filesystem::path& path) {
    return read_gaussian_ply(path);
}

} // namespace cugs
