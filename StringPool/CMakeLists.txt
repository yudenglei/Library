cmake_minimum_required(VERSION 3.16)
project(string_pool VERSION 1.0.0 LANGUAGES CXX)

# ============================================================
# C++17 standard
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# Compiler warnings & optimizations
# ============================================================
if(MSVC)
    add_compile_options(/W4 /WX /O2 /EHsc /utf-8)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -O2)
    if(APPLE)
        add_compile_options(-stdlib=libc++)
    endif()
endif()

# ============================================================
# Header-only interface library (no compilation needed)
# ============================================================
add_library(string_pool INTERFACE)
target_include_directories(string_pool INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(string_pool INTERFACE cxx_std_17)

# Thread support
find_package(Threads REQUIRED)
target_link_libraries(string_pool INTERFACE Threads::Threads)

# ============================================================
# Tests
# ============================================================
enable_testing()

add_executable(test_correctness tests/test_correctness.cpp)
target_link_libraries(test_correctness PRIVATE string_pool)
add_test(NAME test_correctness COMMAND test_correctness)

add_executable(test_memory tests/test_memory.cpp)
target_link_libraries(test_memory PRIVATE string_pool)
add_test(NAME test_memory COMMAND test_memory)

# ============================================================
# Benchmark
# ============================================================
add_executable(bench_main benchmark/bench_main.cpp)
target_link_libraries(bench_main PRIVATE string_pool)

# ============================================================
# Install
# ============================================================
install(FILES include/string_pool.hpp DESTINATION include)
