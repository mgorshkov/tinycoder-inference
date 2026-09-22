# llama-bench-parity warm generation benchmark (2026-08-29).
# Reproduces llama-bench's tg protocol: fixed-token-count prompt, greedy
# decode, decode-loop-only timing, warmup + N warm reps (mean ± stdev).
# See BenchMain.cpp for the protocol rationale.
#
# Built only when BUILD_TESTS is ON (it compiles against the core static lib
# and the gtest-free harness needs no extra deps beyond tinycoder_core).

add_executable(tinycoder_bench BenchMain.cpp)

target_link_libraries(tinycoder_bench
    PRIVATE
    tinycoder_core
)

set_target_properties(tinycoder_bench PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Split-KV decode-attention parity harness (2026-10-01): greedy token-stream
# dump for TINYCODER_ATTN_SPLIT on/off diffing.  Links tinycoder_core the same
# way as tinycoder_bench so CUDA separable compilation / device linking works.
add_executable(tinycoder_attn_parity ${CMAKE_CURRENT_SOURCE_DIR}/../tools/attn_split_parity.cpp)
target_link_libraries(tinycoder_attn_parity PRIVATE tinycoder_core)
set_target_properties(tinycoder_attn_parity PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
