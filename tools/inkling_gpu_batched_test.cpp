// Batched-prefill GEMV gate for the Inkling hot block types.
//
// inklingGemvBatched exists so that a prefill batch of m tokens decodes every
// weight block ONCE and multiplies it by all m activations.  This tool proves
// three things on the real IQ2_XXS checkpoint:
//
//   parity - for each hot type (NVFP4, Q2_0, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S), a
//            batched call at m=16 is compared per output element against 16
//            single-token inklingGemv calls on the same weights and the same 16
//            activations: max|d| / max|ref| must be < 1e-2.  The two paths share
//            one decoder and one accumulation order, so the ratio should be
//            exactly zero; anything else means the batch is mixing tokens or
//            dropping a row.
//   tail   - rows=253 (a partial 4-row warp group) at m=3 (a partial M bucket),
//            so a wrong tail guard or a wrong `m` clamp cannot pass.
//   speed  - m=16 on an 8-expert slice of the ffn tensors: one batched call
//            against 16 single-token calls, plus the implied weight bandwidth.
//
// Usage: inkling_gpu_batched_test [<model.gguf>] [--scale]
//
// --scale prints the m=1,2,4,8,16 amortisation curve for NVFP4/Q2_0/IQ2_XXS
// instead of the gate.  It is how the current limit was identified: the
// marginal cost per token is activation traffic re-read once per warp, not the
// decoder.
//
// The speed comparison is reported, not gated: the gate is parity, the speed is
// the point of the change.

#include "GGUFLoader.hpp"
#include "gpu/InklingGPU.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

using namespace tinycoder;

namespace {

    constexpr uint32_t kM = 16;
    constexpr float kGate = 1e-2f;
    constexpr uint32_t kParityRows = 256;
    constexpr uint32_t kExpertSlice = 8;

    struct HotType {
        uint32_t type;
        const char *name;
        const char *tensor;
    };

    // One real tensor per hot type, taken from this checkpoint's inventory:
    // the routed experts' gate/up carry IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S, the
    // down projections carry NVFP4 / Q2_0.
    constexpr uint32_t kHotCount = 6;
    const HotType kHot[kHotCount] = {
            {40, "NVFP4", "blk.5.ffn_down_exps.weight"},
            {42, "Q2_0", "blk.0.ffn_down_exps.weight"},
            {19, "IQ1_S", "blk.0.ffn_gate_up_exps.weight"},
            {16, "IQ2_XXS", "blk.5.ffn_gate_up_exps.weight"},
            {17, "IQ2_XS", "blk.10.ffn_gate_up_exps.weight"},
            {22, "IQ2_S", "blk.16.ffn_gate_up_exps.weight"},
    };

    uint32_t blockElems(uint32_t type) {
        if (type == 40 || type == 42) return 64u;
        if (type == 0 || type == 30 || type == 8) return 32u;
        return 256u;
    }

    uint32_t blockBytesFor(uint32_t type) {
        switch (type) {
            case 0: return 128;  // F32, 32 weights
            case 8: return 34;   // Q8_0
            case 12: return 144; // Q4_K
            case 13: return 176; // Q5_K
            case 14: return 210; // Q6_K
            case 16: return 66;  // IQ2_XXS
            case 17: return 74;  // IQ2_XS
            case 18: return 98;  // IQ3_XXS
            case 19: return 50;  // IQ1_S
            case 22: return 82;  // IQ2_S
            case 23: return 136; // IQ4_XS
            case 30: return 64;  // BF16, 32 weights
            case 40: return 36;  // NVFP4
            case 42: return 18;  // Q2_0
            default: return 0;
        }
    }

    double nowMs() {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double, std::milli>(
                       clock::now().time_since_epoch())
                .count();
    }

    struct Buf {
        void *p = nullptr;
        explicit Buf(size_t bytes) {
            if (cudaMalloc(&p, bytes ? bytes : 1) != cudaSuccess) p = nullptr;
        }
        ~Buf() {
            if (p) cudaFree(p);
        }
        Buf(const Buf &) = delete;
        Buf &operator=(const Buf &) = delete;
        template <typename T>
        T *as() const {
            return static_cast<T *>(p);
        }
        explicit operator bool() const { return p != nullptr; }
    };

    struct Tensor {
        const uint8_t *host = nullptr;
        uint32_t type = 0;
        uint32_t cols = 0;
        uint32_t rowsPerExpert = 0;
        uint32_t experts = 1;
        uint64_t rowBytes = 0;
    };

    bool loadTensor(GGUFLoader &loader, const HotType &hot, Tensor &t) {
        const TensorInfo *info = loader.getTensorInfo(hot.tensor);
        if (info == nullptr) {
            std::fprintf(stderr, "missing tensor %s\n", hot.tensor);
            return false;
        }
        if (info->type != hot.type) {
            std::fprintf(stderr, "%s: type %u, expected %u for %s\n", hot.tensor,
                         info->type, hot.type, hot.name);
            return false;
        }
        t.host = loader.getTensor(hot.tensor);
        if (t.host == nullptr) {
            std::fprintf(stderr, "%s: no data\n", hot.tensor);
            return false;
        }
        t.type = info->type;
        t.cols = static_cast<uint32_t>(info->shape[0]);
        t.rowsPerExpert = static_cast<uint32_t>(info->shape[1]);
        t.experts = info->shape.size() > 2 ? static_cast<uint32_t>(info->shape[2]) : 1u;
        const uint32_t bb = blockBytesFor(t.type);
        const uint32_t bs = blockElems(t.type);
        if (bb == 0 || bs == 0) {
            std::fprintf(stderr, "%s: unsupported type %u\n", hot.tensor, t.type);
            return false;
        }
        t.rowBytes = static_cast<uint64_t>((t.cols + bs - 1) / bs) * bb;
        return true;
    }

    /// Deterministic activations: every one of the `m` token rows is different,
    /// so a kernel that reuses token 0's activations (or writes the wrong
    /// token's row) cannot pass.
    void fillActivations(std::vector<float> &x, uint32_t m, uint32_t cols) {
        x.resize(static_cast<size_t>(m) * cols);
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = 0.35f * std::sin(0.017f * static_cast<float>(i)) +
                   0.11f * std::cos(0.0031f * static_cast<float>(i));
    }

    /// Parity: batched at m against m single-token calls, same device weights.
    bool parityCase(GGUFLoader &loader, const HotType &hot, uint32_t rows,
                    uint32_t m, bool print) {
        Tensor t;
        if (!loadTensor(loader, hot, t)) return false;
        if (rows > t.rowsPerExpert) rows = t.rowsPerExpert;

        std::vector<float> x;
        fillActivations(x, m, t.cols);
        const size_t weightBytes = static_cast<size_t>(rows) * t.rowBytes;
        Buf dw(weightBytes);
        Buf dx(x.size() * sizeof(float));
        Buf dref(static_cast<size_t>(m) * rows * sizeof(float));
        Buf dout(static_cast<size_t>(m) * rows * sizeof(float));
        if (!dw || !dx || !dref || !dout) {
            std::fprintf(stderr, "%s: cudaMalloc failed\n", hot.tensor);
            return false;
        }
        if (cudaMemcpy(dw.p, t.host, weightBytes, cudaMemcpyHostToDevice) !=
            cudaSuccess) {
            std::fprintf(stderr, "%s: weight upload failed\n", hot.tensor);
            return false;
        }
        cudaMemcpy(dx.p, x.data(), x.size() * sizeof(float),
                   cudaMemcpyHostToDevice);
        // Canary: the batched output buffer is pre-filled with something that is
        // not a plausible result, so a row the kernel fails to write shows up as
        // a parity error instead of as zero-vs-zero.
        cudaMemset(dout.p, 0x7f, static_cast<size_t>(m) * rows * sizeof(float));

        for (uint32_t s = 0; s < m; ++s) {
            if (!gpu::inklingGemv(t.type, dw.p, dx.as<float>() + static_cast<size_t>(s) * t.cols,
                                  dref.as<float>() + static_cast<size_t>(s) * rows,
                                  rows, t.cols, nullptr)) {
                std::fprintf(stderr, "%s: single-token GEMV refused (type %u)\n",
                             hot.tensor, t.type);
                return false;
            }
        }
        if (!gpu::inklingGemvBatched(t.type, dw.p, dx.as<float>(), dout.as<float>(),
                                     rows, t.cols, m, nullptr)) {
            std::fprintf(stderr, "%s: batched GEMV refused (type %u, m=%u)\n",
                         hot.tensor, t.type, m);
            return false;
        }
        std::vector<float> ref(static_cast<size_t>(m) * rows);
        std::vector<float> got(static_cast<size_t>(m) * rows);
        cudaMemcpy(ref.data(), dref.p, ref.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        const cudaError_t ce = cudaMemcpy(got.data(), dout.p,
                                          got.size() * sizeof(float),
                                          cudaMemcpyDeviceToHost);
        if (ce != cudaSuccess) {
            std::fprintf(stderr, "%s: result copy failed: %s\n", hot.tensor,
                         cudaGetErrorString(ce));
            return false;
        }

        double maxAbs = 0.0, maxRef = 0.0;
        uint32_t worstS = 0, worstR = 0;
        for (uint32_t s = 0; s < m; ++s) {
            for (uint32_t r = 0; r < rows; ++r) {
                const size_t i = static_cast<size_t>(s) * rows + r;
                const double d = std::fabs(double(got[i]) - ref[i]);
                if (d > maxAbs) {
                    maxAbs = d;
                    worstS = s;
                    worstR = r;
                }
                maxRef = std::max(maxRef, std::fabs(double(ref[i])));
            }
        }
        const double ratio = maxAbs / std::max(maxRef, 1e-30);
        const bool ok = ratio < kGate;
        if (print) {
            std::printf("%-8s t=%2u rows=%-5u m=%-2u %-26s max|d|=%.3e "
                        "max|ref|=%.3e ratio=%.3e %s",
                        hot.name, t.type, rows, m, hot.tensor, maxAbs, maxRef,
                        ratio, ok ? "PASS" : "FAIL");
            if (!ok)
                std::printf("  worst s=%u r=%u ref=%.6f got=%.6f", worstS, worstR,
                            ref[static_cast<size_t>(worstS) * rows + worstR],
                            got[static_cast<size_t>(worstS) * rows + worstR]);
            std::printf("\n");
        }
        return ok;
    }

    struct Timing {
        double batchedMs = 0.0;
        double batchedMin = 0.0;
        double singleMs = 0.0;      // 16 x inklingGemv (each with its own sync)
        double singleMin = 0.0;
        double singleAsyncMs = 0.0; // 16 launches + one sync
        double singleAsyncMin = 0.0;
        size_t weightBytes = 0;
        uint64_t flops = 0;
    };

    /// Time one type at m=16 on an 8-expert slice of its ffn tensor.
    bool timeType(GGUFLoader &loader, const HotType &hot, uint32_t m,
                  uint32_t reps, Timing &out) {
        Tensor t;
        if (!loadTensor(loader, hot, t)) return false;
        const uint64_t rows = static_cast<uint64_t>(kExpertSlice) * t.rowsPerExpert;
        const size_t weightBytes = static_cast<size_t>(rows * t.rowBytes);
        std::vector<float> x;
        fillActivations(x, m, t.cols);
        Buf dw(weightBytes);
        Buf dx(x.size() * sizeof(float));
        Buf dout(static_cast<size_t>(m) * rows * sizeof(float));
        Buf dref(static_cast<size_t>(m) * rows * sizeof(float));
        if (!dw || !dx || !dout || !dref) return false;
        if (cudaMemcpy(dw.p, t.host, weightBytes, cudaMemcpyHostToDevice) !=
            cudaSuccess) {
            std::fprintf(stderr, "%s: slice upload failed (%zu MiB)\n", hot.tensor,
                         weightBytes >> 20);
            return false;
        }
        cudaMemcpy(dx.p, x.data(), x.size() * sizeof(float),
                   cudaMemcpyHostToDevice);

        const auto batchedOnce = [&]() {
            return gpu::inklingGemvBatched(t.type, dw.p, dx.as<float>(),
                                           dout.as<float>(),
                                           static_cast<uint32_t>(rows), t.cols, m,
                                           nullptr);
        };
        const auto singleOnce = [&]() {
            for (uint32_t s = 0; s < m; ++s)
                if (!gpu::inklingGemv(
                            t.type, dw.p,
                            dx.as<float>() + static_cast<size_t>(s) * t.cols,
                            dref.as<float>() + static_cast<size_t>(s) * rows,
                            static_cast<uint32_t>(rows), t.cols, nullptr))
                    return false;
            return true;
        };
        const auto singleAsyncOnce = [&]() {
            for (uint32_t s = 0; s < m; ++s)
                if (!gpu::inklingGemvAsync(
                            t.type, dw.p,
                            dx.as<float>() + static_cast<size_t>(s) * t.cols,
                            dref.as<float>() + static_cast<size_t>(s) * rows,
                            static_cast<uint32_t>(rows), t.cols, nullptr))
                    return false;
            return cudaDeviceSynchronize() == cudaSuccess;
        };

        // Warm-up: the first call allocates the module's context state.
        if (!batchedOnce() || !singleOnce() || !singleAsyncOnce()) {
            std::fprintf(stderr, "%s: warm-up call failed\n", hot.tensor);
            return false;
        }
        double bestB = 1e30, bestS = 1e30, bestA = 1e30, sumB = 0.0, sumS = 0.0,
               sumA = 0.0;
        for (uint32_t i = 0; i < reps; ++i) {
            const double b0 = nowMs();
            if (!batchedOnce()) return false;
            const double b1 = nowMs();
            const double s0 = nowMs();
            if (!singleOnce()) return false;
            const double s1 = nowMs();
            const double a0 = nowMs();
            if (!singleAsyncOnce()) return false;
            const double a1 = nowMs();
            bestB = std::min(bestB, b1 - b0);
            bestS = std::min(bestS, s1 - s0);
            bestA = std::min(bestA, a1 - a0);
            sumB += b1 - b0;
            sumS += s1 - s0;
            sumA += a1 - a0;
        }
        out.batchedMs = sumB / reps;
        out.batchedMin = bestB;
        out.singleMs = sumS / reps;
        out.singleMin = bestS;
        out.singleAsyncMs = sumA / reps;
        out.singleAsyncMin = bestA;
        out.weightBytes = weightBytes;
        out.flops = 2ull * rows * t.cols * m;
        std::printf("%-8s t=%2u %-26s rows=%-6llu cols=%-5u %7.2f MiB  "
                    "batched %8.3f ms (min %8.3f)  single16 %8.3f ms (min %8.3f)"
                    "  speedup %6.2fx\n",
                    hot.name, t.type, hot.tensor,
                    static_cast<unsigned long long>(rows), t.cols,
                    double(weightBytes) / 1048576.0, out.batchedMs,
                    out.batchedMin, out.singleMs, out.singleMin,
                    out.singleMs / std::max(out.batchedMs, 1e-9));
        return true;
    }

}// namespace

int main(int argc, char **argv) {
    const char *modelPath =
            argc > 1 && argv[1][0] != '-'
                    ? argv[1]
                    : "/data/models/inkling/inkling-2-mini/"
                      "Inkling-2-Mini-UD-IQ2_XXS.gguf";
    const uint32_t reps = 8;

    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
        std::fprintf(stderr, "no CUDA device\n");
        return 2;
    }
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    std::printf("[batched-gemv] %s, sm_%d%d, m=%u, gate ratio<%.0e\n", prop.name,
                prop.major, prop.minor, kM, kGate);

    GGUFLoader loader;
    std::printf("[batched-gemv] loading %s\n", modelPath);
    if (!loader.load(modelPath)) {
        std::fprintf(stderr, "load failed: %s\n", modelPath);
        return 2;
    }

    // ---- parity: m=16 batched vs 16 single-token calls --------------------
    std::printf("[batched-gemv] --- parity: inklingGemvBatched(m=16) vs 16 x "
                "inklingGemv, real weights, %u rows ---\n", kParityRows);
    int fails = 0;
    for (const HotType &hot: kHot)
        fails += parityCase(loader, hot, kParityRows, kM, true) ? 0 : 1;

    // ---- tail: partial 4-row group x partial M bucket ---------------------
    // One type is enough: the tail logic is type-independent (it lives in the
    // kernel prologue, not in the decoder), so this checks the shape, not the
    // codec.
    std::printf("[batched-gemv] --- tail: rows=253 (partial 4-row group), m=3 "
                "(partial M bucket) ---\n");
    fails += parityCase(loader, kHot[3], 253, 3, true) ? 0 : 1;
    fails += parityCase(loader, kHot[0], 253, 3, true) ? 0 : 1;
    // The two smallest compiled M arms, so no bucket is left unexercised.
    fails += parityCase(loader, kHot[3], 64, 1, true) ? 0 : 1;
    fails += parityCase(loader, kHot[3], 64, 2, true) ? 0 : 1;

    if (argc > 2 && std::strcmp(argv[2], "--scale") == 0) {
        // Diagnostic mode: how batched time scales with m.
        for (const HotType &hot: kHot) {
            if (hot.type != 40 && hot.type != 16 && hot.type != 42) continue;
            for (uint32_t mm: {1u, 2u, 4u, 8u, 16u}) {
                Timing tm;
                if (timeType(loader, hot, mm, 5, tm))
                    std::printf("    scale %-6s m=%-2u batched %.4f ms  flops %.3f G  GB/s %.1f\n",
                                hot.name, mm, tm.batchedMs,
                                double(tm.flops) / 1e9,
                                double(tm.weightBytes) / (tm.batchedMin * 1e6));
            }
        }
        return 0;
    }

    // ---- speed: m=16 on an 8-expert slice --------------------------------
    std::printf("[batched-gemv] --- speed: m=16, 8-expert ffn slice, %u reps ---\n",
                reps);
    std::vector<Timing> times;
    times.reserve(kHotCount);
    for (const HotType &hot: kHot) {
        Timing tm;
        if (!timeType(loader, hot, kM, reps, tm)) {
            std::printf("%-8s t=%2u timing FAILED\n", hot.name, hot.type);
            fails += 1;
            times.push_back(tm);
            continue;
        }
        times.push_back(tm);
    }

    std::printf("[batched-gemv] --- summary ---\n");
    std::printf("%-8s %8s %8s %10s %10s %9s %9s %9s %9s\n", "type", "M=",
                "experts", "batched_ms", "single16", "speedup", "async16",
                "spd_async", "batched_GB/s");
    for (uint32_t i = 0; i < kHotCount; ++i) {
        const Timing &tm = times[i];
        if (tm.batchedMs <= 0.0) continue;
        const double gbs = double(tm.weightBytes) /
                           (std::max(tm.batchedMin, 1e-9) * 1e6);
        std::printf("%-8s %8u %8u %10.3f %10.3f %8.2fx %9.3f %8.2fx %9.1f\n",
                    kHot[i].name, kM, kExpertSlice, tm.batchedMs, tm.singleMs,
                    tm.singleMs / std::max(tm.batchedMs, 1e-9), tm.singleAsyncMs,
                    tm.singleAsyncMs / std::max(tm.batchedMs, 1e-9), gbs);
    }

    const bool pass = fails == 0;
    std::printf("%s\n", pass ? "INKLING BATCHED GEMV TEST PASSED"
                             : "INKLING BATCHED GEMV TEST FAILED");
    return pass ? 0 : 1;
}
