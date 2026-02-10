#include "utils/cuda_info.hpp"

#include <spdlog/spdlog.h>
#include <cstdlib>

int main() {
    spdlog::info("cuda-gaussian-splatting — hello_cuda");
    spdlog::info("------------------------------------");

    bool ok = cugs::print_gpu_info_and_test();
    if (!ok) {
        spdlog::error("GPU verification failed.");
        return EXIT_FAILURE;
    }

    spdlog::info("All good. Ready to splat.");
    return EXIT_SUCCESS;
}
