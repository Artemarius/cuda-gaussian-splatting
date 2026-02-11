add_executable(hello_cuda hello_cuda.cpp)
target_link_libraries(hello_cuda PRIVATE cugs_utils)

add_executable(dump_points dump_points.cpp)
target_link_libraries(dump_points PRIVATE cugs_data)
