// tiny_weights_dump_qwen2.cpp
//
// Qwen2-dense weight dequant fingerprint dump — TinyCoder's counterpart to
// `llama_ref_probe <model> --weights-only`.  Loads the GGUF via GGUFLoader
// (mmap), dequantizes selected Qwen2 tensors with GGMLDequantize, and prints
// the SAME norm/fnv/first8/per-block-fnv format so the outputs can be diffed
// against llama.cpp's ggml reference dequant.
//
// Build:
//   g++ -O2 -std=c++17 tools/tiny_weights_dump_qwen2.cpp \
//       -I include -Isrc/cpp/core \
//       build/libtinycoder_core.a -lpthread -o /tmp/tiny_weights_dump_qwen2
//
// Run:
//   /tmp/tiny_weights_dump_qwen2 /data/models/qwen/Qwen2.5-Coder-7B-Instruct-IQ2_S.gguf

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
    const char *modelPath = std::getenv("TINYCODER_MODEL_PATH");
    if (argc >= 2 && argv[1][0] != '-') modelPath = argv[1];
    if (!modelPath) {
        fprintf(stderr, "usage: %s <model.gguf> | TINYCODER_MODEL_PATH=<path>\n", argv[0]);
        return 1;
    }

    GGUFLoader loader;
    if (!loader.load(modelPath)) {
        fprintf(stderr, "GGUFLoader::load failed for %s\n", modelPath);
        return 2;
    }

    struct Spec {
        const char *name;
        int64_t rows;   // GGUF shape[1]
        int64_t cols;   // GGUF shape[0]
    };
    // 1.5B qwen2.5-coder-iq3_xxs-imat: hidden=1536, 12 Q heads, 2 KV heads,
    // intermediate=8960, vocab=151936.
    static const Spec specs[] = {
            {"blk.0.attn_q.weight",     1536,   1536},
            {"blk.0.attn_k.weight",      256,   1536},
            {"blk.0.attn_v.weight",      256,   1536},
            {"blk.0.attn_output.weight", 1536,  1536},
            {"blk.0.ffn_gate.weight",    8960,  1536},
            {"blk.0.ffn_up.weight",      8960,  1536},
            {"blk.0.ffn_down.weight",    1536,  8960},
            {"token_embd.weight",      151936, 1536},
            {"output.weight",          151936, 1536},
    };

    printf("  [weights] raw weight-block fingerprints (TinyCoder dequant):\n");
    for (const auto &spec : specs) {
        const GGUFLoader::TensorInfo *info = loader.getTensorInfo(spec.name);
        const uint8_t *tdata = loader.getTensor(spec.name);
        if (!info || !tdata) {
            printf("    [%s] <tensor not found>\n", spec.name);
            continue;
        }
        // GGUF stores rowwise; the GGUF shape is [cols, rows] and tensor data
        // is rows of rowBytes = ceil(cols/blockSize)*typeSize.
        const uint32_t type = info->type;
        const int64_t cols = spec.cols;
        const int64_t ts = ggmlTypeSize(type);
        const int64_t bs = ggmlBlockSize(type);
        const int64_t blocksPerRow = (cols + bs - 1) / bs;
        const int64_t bytesPerRow = blocksPerRow * ts;

        printf("    [%s] ne=[%lld,%lld] type=%u blocksPerRow=%lld rowwise\n",
               spec.name, (long long) cols, (long long) spec.rows,
               (unsigned) type, (long long) blocksPerRow);

        const int64_t rowsToDump[2] = {0, 10};
        for (int64_t ri = 0; ri < 2; ++ri) {
            const int64_t row = rowsToDump[ri];
            if (row >= spec.rows) {
                printf("      row %lld: OOB\n", (long long) row);
                continue;
            }
            const uint8_t *rowData = tdata + (size_t) row * bytesPerRow;
            std::vector<float> dq = GGMLDequantize::dequantize(type, rowData, (uint64_t) cols);
            if ((int64_t) dq.size() != cols) {
                printf("      row %lld: dequantize failed (size %zu)\n",
                       (long long) row, dq.size());
                continue;
            }
            double s = 0.0;
            for (int64_t j = 0; j < cols; ++j) s += (double) dq[j] * dq[j];
            printf("      row %lld: norm=%12.6f fnv=%016llx first8=[% .6f % .6f % .6f % .6f % .6f % .6f % .6f % .6f]\n",
                   (long long) row, std::sqrt(s),
                   (unsigned long long) fnv1a64(dq.data(), cols),
                   dq[0], dq[1], dq[2], dq[3], dq[4], dq[5], dq[6], dq[7]);
            printf("        blockFnv: ");
            std::vector<int64_t> blocks;
            for (int64_t b = 0; b < blocksPerRow && b < 4; ++b) blocks.push_back(b);
            if (blocksPerRow > 4) blocks.push_back(blocksPerRow - 1);
            for (int64_t b : blocks) {
                printf("b%lld=%016llx ", (long long) b,
                       (unsigned long long) fnv1a64(dq.data() + b * bs, bs));
            }
            printf("\n");
        }

        // Exhaustive full-tensor rolling FNV (same scheme as llama_ref_probe's
        // ALL hash) so a single bad row/block anywhere is detected.
        {
            uint64_t allRowFnv = 1469598103934665603ULL;
            double totalSq = 0.0;
            for (int64_t row = 0; row < spec.rows; ++row) {
                const uint8_t *rowData = tdata + (size_t) row * bytesPerRow;
                std::vector<float> dq = GGMLDequantize::dequantize(type, rowData, (uint64_t) cols);
                if ((int64_t) dq.size() != cols) break;
                for (int64_t j = 0; j < cols; ++j) {
                    uint32_t bits;
                    std::memcpy(&bits, &dq[j], sizeof(bits));
                    allRowFnv ^= bits;
                    allRowFnv *= 1099511628211ULL;
                    totalSq += (double) dq[j] * dq[j];
                }
            }
            printf("      ALL: rowsTotalFnv=%016llx totalNorm=%12.6f\n",
                   (unsigned long long) allRowFnv, std::sqrt(totalSq));
        }
    }
    return 0;
}
