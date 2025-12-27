cmake_minimum_required(VERSION 3.10)
project(ToyAI)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(src/core)

# Core library sources
set(CORE_SOURCES
    src/core/Matrix.cpp
    src/core/Embedding.cpp
    src/core/Softmax.cpp
    src/core/LinearProjection.cpp
    src/core/Attention.cpp
    src/core/Loss.cpp
    src/core/Optimizer.cpp
)

# Create core library
add_library(core_lib STATIC ${CORE_SOURCES})

# Example 1: Forward only
add_executable(example1 examples/example1_forward_only/main.cpp)
target_link_libraries(example1 core_lib)

# Example 2: Single training step
add_executable(example2 examples/example2_single_step/main.cpp)
target_link_libraries(example2 core_lib)

# Example 3: Full backprop
add_executable(example3 examples/example3_full_backprop/main.cpp)
target_link_libraries(example3 core_lib)

# Example 4: Multiple patterns
add_executable(example4 examples/example4_multiple_patterns/main.cpp)
target_link_libraries(example4 core_lib)

# Example 5: Feed-forward
add_executable(example5 examples/example5_feedforward/main.cpp)
target_link_libraries(example5 core_lib)

# Example 6: Complete transformer
add_executable(example6 examples/example6_complete/main.cpp)
target_link_libraries(example6 core_lib)

# Example 7: Character recognition
add_executable(example7 examples/example7_character_recognition/main.cpp)
target_link_libraries(example7 core_lib)

# Set output directories
set_target_properties(example1 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/example1_forward_only"
)
set_target_properties(example2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/example2_single_step"
)
set_target_properties(example3 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/example3_full_backprop"
)
set_target_properties(example4 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/example4_multiple_patterns"
)
set_target_properties(example5 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/example5_feedforward"
)
set_target_properties(example6 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/example6_complete"
)
set_target_properties(example7 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/example7_character_recognition"
)

