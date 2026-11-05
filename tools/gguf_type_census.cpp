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

// gguf_type_census.cpp
//
// Print a per-tensor-type census (name -> GGML type) for a GGUF file using only
// GGUFLoader::loadMetadata() (header parsing; NO tensor payload reads, NO
// model construction).  Runs in well under a second even on multi-GB models.
//
// Build:
//   g++ -O2 -std=c++17 tools/gguf_type_census.cpp -I include \
//       build/libtinycoder_core.a -lpthread -o /tmp/gguf_type_census
//
// Run:
//   /tmp/gguf_type_census /data/models/qwen/Qwen3.6-27B-Q6_K.gguf [--names]
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "GGUFLoader.hpp"

using namespace tinycoder;

// Type-name table. IMPORTANT: the IDs MUST match the canonical GGML enum in
// ggml/include/ggml.h (see GGML_TYPE_* in include/GGUFLoader.hpp for the subset
// this loader implements). Earlier drafts of this table mislabeled 19-24 (e.g.
// 22 as "IQ2_M" — a type that does not exist in GGML; 22 is IQ2_S), which led
// to wrong type names in model censuses.
static const char *typeName(uint32_t t) {
    switch (t) {
    case GGML_TYPE_F32: return "F32";
    case GGML_TYPE_F16: return "F16";
    case GGML_TYPE_Q4_0: return "Q4_0";
    case GGML_TYPE_Q4_1: return "Q4_1";
    case GGML_TYPE_Q5_0: return "Q5_0";
    case GGML_TYPE_Q5_1: return "Q5_1";
    case GGML_TYPE_Q8_0: return "Q8_0";
    case GGML_TYPE_Q8_1: return "Q8_1";
    case GGML_TYPE_Q2_K: return "Q2_K";
    case GGML_TYPE_Q3_K: return "Q3_K";
    case GGML_TYPE_Q4_K: return "Q4_K";
    case GGML_TYPE_Q5_K: return "Q5_K";
    case GGML_TYPE_Q6_K: return "Q6_K";
    case GGML_TYPE_Q8_K: return "Q8_K";
    case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
    case GGML_TYPE_IQ2_XS: return "IQ2_XS";
    case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
    case GGML_TYPE_IQ1_S: return "IQ1_S";
    case GGML_TYPE_IQ4_NL: return "IQ4_NL";
    case GGML_TYPE_IQ3_S: return "IQ3_S";
    case GGML_TYPE_IQ2_S: return "IQ2_S";
    case GGML_TYPE_IQ4_XS: return "IQ4_XS";
    case GGML_TYPE_I8: return "I8";
    case 27: return "I64";
    case 28: return "F64";
    case 29: return "IQ1_M";
    case 30: return "BF16";
    case 34: return "TQ1_0";
    case 35: return "TQ2_0";
    case 36: return "Q4_K_M (fork)";
    case 37: return "Q4_K_L (fork)";
    case 38: return "Q4_K_XL (fork)";
    case 39: return "MXFP4";
    case 40: return "NVFP4";
    case 41: return "Q1_0";
    case 42: return "Q2_0";
    default: return "?";
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s model.gguf [--names]\n", argv[0]);
        return 2;
    }
    const bool showNames = (argc > 2 && std::string(argv[2]) == "--names");

    GGUFLoader loader;
    if (!loader.loadMetadata(argv[1])) {
        std::fprintf(stderr, "loadMetadata failed: %s\n", argv[1]);
        return 1;
    }
    const ModelConfig &cfg = loader.config();
    std::printf("model: %s | arch=%s | layers=%u hidden=%u\n",
                cfg.modelName.c_str(), cfg.architecture.c_str(),
                cfg.numLayers, cfg.hiddenSize);

    const auto &recs = loader.tensorRecords();
    std::map<uint32_t, size_t> countByType;
    for (const auto &r: recs) countByType[r.type]++;

    std::printf("\n-- tensor type census (%zu tensors) --\n", recs.size());
    for (const auto &[t, n]: countByType) {
        std::printf("  %2u %-10s : %zu\n", t, typeName(t), n);
    }

    if (showNames) {
        std::printf("\n-- first 8 layers (name: type) --\n");
        size_t shown = 0;
        for (const auto &r: recs) {
            if (r.name.find("blk.") == 0) {
                const size_t dot = r.name.find('.', 4);
                if (dot != std::string::npos) {
                    const int lidx = std::atoi(r.name.substr(4, dot - 4).c_str());
                    if (lidx > 8) continue;
                }
            }
            std::printf("  %-40s %2u %s\n", r.name.c_str(), r.type,
                        typeName(r.type));
            if (++shown > 200) { std::printf("  ...\n"); break; }
        }
    }
    return 0;
}
