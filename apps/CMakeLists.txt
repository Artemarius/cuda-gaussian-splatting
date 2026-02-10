add_executable(hello_cuda hello_cuda.cpp)
target_link_libraries(hello_cuda PRIVATE cugs_utils)
