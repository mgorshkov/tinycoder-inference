/*
⚡ TinyCoder AI

Copyright (c) 2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/**
 * TinyCoder Model Unit Test (Google Test)
 *
 * Tests model loading, tokenization, forward pass, KV cache management,
 * and sample question answering.
 *
 * GPU acceleration is used automatically when the library is compiled
 * with ENABLE_CUDA=ON and a CUDA-capable GPU is available.
 *
 * Usage:
 *   TINYCODER_MODEL_PATH=<path_to_model.gguf> ./tinycoder_test
 */

#include "SharedTestEnv.hpp"
#include "ThreadPool.hpp"
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <unistd.h>

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize the thread pool with the LOGICAL-CPU count
    // (ThreadPool::recommendedThreadCount: $TINYCODER_THREADS override, else
    // sysconf(_SC_NPROCESSORS_ONLN) on Linux). The measured optimum tracks the
    // logical-CPU count rather than the physical-core count; oversubscribing
    // past it regresses sharply — see plans/generation_optimizations.md
    // Appendix B.
    tinycoder::ThreadPool::instance().initialize(
            tinycoder::ThreadPool::recommendedThreadCount());

    // Read model path from environment variable.
    const char *envPath = std::getenv("TINYCODER_MODEL_PATH");
    if (envPath != nullptr && envPath[0] != '\0') {
        SharedTestEnv::modelPath = envPath;
    }

    // Command-line argument overrides environment variable
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--model-path" && i + 1 < argc) {
            SharedTestEnv::modelPath = argv[++i];
        }
    }
    // If no model path was provided (neither TINYCODER_MODEL_PATH nor
    // --model-path), fall back to the first candidate that actually exists on
    // disk.  This keeps the self-contained suites (EXL3 codec/safetensors/
    // loader) running even when no model is available: an empty path just makes
    // the model-dependent fixtures skip themselves instead of failing the whole
    // run.  The order prefers the small GGUF references, then any EXL3
    // checkpoint directory.
    if (SharedTestEnv::modelPath.empty()) {
        namespace fs = std::filesystem;
        const char *candidates[] = {
                "/data/models/qwen/qwen2.5/qwen2.5-coder-1.5b-instruct-q2_k.gguf",
                "/data/models/qwen/qwen2.5-coder-1.5b-instruct-q2_k.gguf",
                "/data/models/qwen/qwen2.5/qwen2.5-coder-7b-instruct-iq3_xxs-imat.gguf",
                "/data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw",
        };
        for (const char *c: candidates) {
            if (fs::exists(c)) {
                SharedTestEnv::modelPath = c;
                break;
            }
        }
        if (SharedTestEnv::modelPath.empty()) {
            std::cerr << "[tinycoder_test] No TINYCODER_MODEL_PATH and no default "
                         "model found; model-dependent tests will be skipped."
                      << std::endl;
        }
    }

    // Register the shared test environment (loads model once before all tests)
    ::testing::AddGlobalTestEnvironment(new SharedTestEnv());

    return RUN_ALL_TESTS();
}
