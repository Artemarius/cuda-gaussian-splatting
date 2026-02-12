add_executable(hello_cuda hello_cuda.cpp)
target_link_libraries(hello_cuda PRIVATE cugs_utils)

add_executable(dump_points dump_points.cpp)
target_link_libraries(dump_points PRIVATE cugs_data)

add_executable(train train_main.cpp)
target_link_libraries(train PRIVATE cugs_training)

add_executable(eval eval_main.cpp)
target_link_libraries(eval PRIVATE cugs_training)
