cmake_minimum_required(VERSION 3.18.0)

set(PROJECT_NAME tinycoder_unit_tests)

project(${PROJECT_NAME} VERSION 0.5.0 DESCRIPTION "TinyCoder Inference Unit Tests")

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG main
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(BUILD_GTEST ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# tinycoder_core static library is built in the root CMakeLists.txt
# and is available as a target in this subdirectory scope.

# ---- Google Test executable ----
add_executable(tinycoder_test
    Main.cpp
    ModelTest.cpp
    DequantizeTest.cpp
    ReferenceCompareTest.cpp
    ThreadPoolReentrancyTest.cpp
    GPUCpuCompareTest.cpp
    Int8PackParityTest.cpp
    MmqParityTest.cpp
    SharedTestEnv.cpp
)

target_link_libraries(tinycoder_test
    PRIVATE
    gtest
    tinycoder_core
)

# ---- Toolchain workaround (CUDA + GCC) ----
# nvcc's device-link step injects `-L/usr/lib/gcc/x86_64-linux-gnu/<N>` ahead of
# the host C++ runtime, so the driver's implicit `-lstdc++` resolves to the
# gcc-<N> libstdc++ (a build that predates CXXABI_1.3.15).  googletest's
# `main` references `__cxa_call_terminate`, which only the newer SYSTEM
# libstdc++ defines, so the test fails to link with:
#   undefined reference to `__cxa_call_terminate'
# Appending the system libstdc++ by full path (after the other libs) restores
# the symbol without changing the CPU-only build.  Guarded by an EXISTS check
# so a non-standard layout is simply left to the default toolchain.
if(ENABLE_CUDA AND NOT APPLE AND EXISTS "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    target_link_libraries(tinycoder_test PRIVATE /usr/lib/x86_64-linux-gnu/libstdc++.so.6)
endif()

# ---- Standalone logits probe (diagnostic; not a gtest) ----
# Links the same way as tinycoder_test so CUDA separable compilation,
# np, and tinycoder_core are all wired correctly for tools/tiny_logits_probe.cpp.
add_executable(tiny_logits_probe
    ../tools/tiny_logits_probe.cpp
)

target_link_libraries(tiny_logits_probe
    PRIVATE
    tinycoder_core
)

target_include_directories(tinycoder_test
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Testing
)

# Set C++ standard for the test executable
set_target_properties(tinycoder_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Register the test with CTest
add_test(NAME tinycoder_test COMMAND tinycoder_test)
