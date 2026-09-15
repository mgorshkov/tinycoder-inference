// tiny_embed_dump.cpp
//
// Dump TinyCoder's quantized token-embedding dequant for the first N tokens of
// a model, in the same format as llama_ref_probe's `[embed]` block, so the two
// can be diffed directly.  Build:
//   g++ -O2 -std=c++17 tools/tiny_embed_dump.cpp -I include -Isrc/cpp/core \
//       build/libtinycoder_core.a -lpthread -o /tmp/tiny_embed_dump
// Run:
//   /tmp/tiny_embed_dump <model.gguf> [numTokens]
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "GGUFLoader.hpp"
#include "GGMLDequantize.hpp"

using namespace tinycoder;

static uint64_t fnv1a64(const float *v, int64_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (int64_t i = 0; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &v[i], sizeof(bits));
        h ^= bits;
        h *= 1099511628211ULL;
    }
    return h;
}

int main(int argc, char **argv) {
    const char *modelPath = argc >= 2 ? argv[1] : nullptr;
    if (!modelPath) {
        fprintf(stderr, "usage: %s <model.gguf> [numTokens]\n", argv[0]);
        return 1;
    }
    int64_t nTok = argc >= 3 ? atoll(argv[2]) : 8;

    GGUFLoader loader;
    if (!loader.load(modelPath)) {
        fprintf(stderr, "load failed\n");
        return 2;
    }
    const GGUFLoader::TensorInfo *info = loader.getTensorInfo("token_embd.weight");
    const uint8_t *td = loader.getTensor("token_embd.weight");
    if (!info || !td) {
        fprintf(stderr, "token_embd.weight not found\n");
        return 2;
    }
    // GGUF ne = [hiddenSize, vocabSize]; data is rowwise: row = token.
    // GGUF shape[0] (fastest) = hiddenSize.
    const uint32_t type = info->type;
    // Use the loader's per-row layout directly: blocksPerRow for hidden columns.
    // Hidden = col count.
    int64_t cols = info->shape.empty() ? 0 : info->shape[0];
    int64_t rows = info->shape.size() > 1 ? info->shape[1] : 0;
    if (cols <= 0 || rows <= 0) {
        fprintf(stderr, "bad shape\n");
        return 2;
    }
    const int64_t ts = ggmlTypeSize(type);
    const int64_t bs = ggmlBlockSize(type);
    const int64_t blocksPerRow = (cols + bs - 1) / bs;
    const int64_t bytesPerRow = blocksPerRow * ts;

    printf("  [embed] TinyCoder token-embedding dequant (type=%u ne=[%lld,%lld]):\n",
           (unsigned) type, (long long) cols, (long long) rows);
    for (int64_t t = 0; t < nTok && t < rows; ++t) {
        const uint8_t *rowData = td + t * bytesPerRow;
        std::vector<float> dq = GGMLDequantize::dequantize(type, rowData, (uint64_t) cols);
        double s = 0.0;
        for (int64_t j = 0; j < cols; ++j) s += (double) dq[j] * dq[j];
        printf("  tok[%lld] norm=%10.6f rms=%.6f first8=[", (long long) t, std::sqrt(s),
               std::sqrt(s / (double) cols));
        for (int j = 0; j < 8; ++j) {
            printf("% .7f ", dq[j]);
        }
        printf("] fnv=%016llx\n", (unsigned long long) fnv1a64(dq.data(), cols));
    }
    return 0;
}
