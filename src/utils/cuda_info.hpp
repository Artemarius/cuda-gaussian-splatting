#pragma once

namespace cugs {

/// @brief Query the GPU, log its properties, and launch a trivial kernel to
///        verify the CUDA toolchain is working end to end.
/// @return true if a CUDA-capable GPU was found and the test kernel succeeded.
bool print_gpu_info_and_test();

} // namespace cugs
