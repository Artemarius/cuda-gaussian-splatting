cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(cuda-gaussian-splatting LANGUAGES CXX CUDA)

# -----------------------------------------------------------------------------
# Standards
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 3060 (compute capability 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# -----------------------------------------------------------------------------
# Compiler flags (MSVC)
# -----------------------------------------------------------------------------
if(MSVC)
    # Apply MSVC-specific flags only to C/CXX, not to CUDA (nvcc chokes on them).
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    )
    # CUDA needs /utf-8 for fmt/spdlog headers — pass it via -Xcompiler.
    # NVCC's own preprocessor doesn't honour /utf-8, so fmt's compile-time
    # check fails.  FMT_UNICODE=0 disables the check; the host compiler
    # still gets /utf-8 via -Xcompiler, so actual string handling is fine.
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
    add_compile_definitions(
        $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>
    )
    # libtorch ships with /MD — match it across the whole project.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()

# PyTorch ignores CMAKE_CUDA_ARCHITECTURES and uses TORCH_CUDA_ARCH_LIST instead.
set(TORCH_CUDA_ARCH_LIST "8.6")

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

# libtorch — set CMAKE_PREFIX_PATH to external/libtorch on the command line or
# in CMakePresets.json if it isn't found automatically.
find_package(Torch REQUIRED)

find_package(Eigen3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

# stb — header-only, pulled via vcpkg
find_path(STB_INCLUDE_DIR stb_image.h PATH_SUFFIXES stb)

# -----------------------------------------------------------------------------
# Internal library: cugs_utils
# -----------------------------------------------------------------------------
add_library(cugs_utils
    src/utils/cuda_info.cu
)

target_include_directories(cugs_utils PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${STB_INCLUDE_DIR}
)

target_link_libraries(cugs_utils PUBLIC
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
    spdlog::spdlog
    nlohmann_json::nlohmann_json
)

# Enable separable compilation only when needed (cross-TU __device__ calls).
# set_target_properties(cugs_utils PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Propagate CUDA include directories so .cpp files can include cuda_runtime.h
target_include_directories(cugs_utils PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# -----------------------------------------------------------------------------
# Put all executables in the build root so they can find DLLs.
# For multi-config generators (VS), use per-config output dirs.
# -----------------------------------------------------------------------------
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Copy libtorch DLLs into the build directory so executables can run.
if(WIN32 AND EXISTS "${TORCH_INSTALL_PREFIX}/lib")
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    file(GLOB TORCH_BIN_DLLS "${TORCH_INSTALL_PREFIX}/bin/*.dll")
    list(APPEND TORCH_DLLS ${TORCH_BIN_DLLS})
    file(COPY ${TORCH_DLLS} DESTINATION "${CMAKE_BINARY_DIR}")
    # Multi-config generators put exe in <build>/<Config>/ — copy there too.
    foreach(CFG Debug Release RelWithDebInfo MinSizeRel)
        file(COPY ${TORCH_DLLS} DESTINATION "${CMAKE_BINARY_DIR}/${CFG}")
    endforeach()
endif()

# -----------------------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------------------
add_subdirectory(apps)

enable_testing()
add_subdirectory(tests)
