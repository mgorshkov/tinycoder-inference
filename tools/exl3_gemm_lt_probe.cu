/*
 * exl3_gemm_lt_probe -- is the EXL3 prefill GEMM limited by tile/wave
 * quantisation, and does cublasLt have a better kernel for these shapes?
 *
 * The engine's prefill GEMM is issued through cublasGemmEx, which picks
 * turing_s1688gemm_fp16_256x128_... for every EXL3 projection.  At the 27B's
 * shapes that tile gives (n/256) * (seq/128) blocks: 160 blocks for
 * n=10240, seq=512 on a 108-SM card, i.e. 1.48 waves.  This probe times the
 * same GEMM through cublasLt across all heuristic candidates (each with its
 * tile/stages/split-k printed) so the choice can be made on measurement.
 *
 * Build: nvcc -O3 -arch=sm_75 -std=c++17 tools/exl3_gemm_lt_probe.cu \
 *            -lcublas -lcublasLt -o /tmp/gemm_lt_probe
 * Run:   /tmp/gemm_lt_probe
 */
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#define CK(e)                                                                        \
    do {                                                                             \
        cudaError_t r_ = (e);                                                        \
        if (r_ != cudaSuccess) {                                                     \
            std::printf("ERR %s -> %s\n", #e, cudaGetErrorString(r_));                \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

static float clockNow() {
    FILE *f = popen("nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits",
                    "r");
    if (!f) return 0.0f;
    float m = 0.0f;
    if (std::fscanf(f, "%f", &m) != 1) m = 0.0f;
    pclose(f);
    return m / 1000.0f;
}

static float timeMs(int it, const std::function<void()> &fn) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    fn();
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < it; ++i) fn();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / (float) it;
}

int main() {
    cublasHandle_t h;
    cublasCreate(&h);
    cublasLtHandle_t lh;
    if (cublasLtCreate(&lh) != CUBLAS_STATUS_SUCCESS) {
        std::printf("cublasLtCreate failed\n");
        return 1;
    }
    const int shapes[][2] = {{17408, 5120}, {5120, 17408}, {5120, 10240},
                             {5120, 12288},  {6144, 5120},  {5120, 1024}};
    const int seq = 512;
    const size_t maxW = (size_t) 17408 * 17408;
    __half *w = nullptr, *x = nullptr;
    float *y = nullptr, *yRef = nullptr;
    void *ws = nullptr;
    const size_t wsBytes = 64u << 20;
    CK(cudaMalloc(&w, maxW * 2));
    CK(cudaMalloc(&x, (size_t) 17408 * seq * 2));
    CK(cudaMalloc(&y, (size_t) 17408 * seq * 4));
    CK(cudaMalloc(&yRef, (size_t) 17408 * seq * 4));
    CK(cudaMalloc(&ws, wsBytes));
    CK(cudaMemset(w, 0x3c, maxW * 2));
    CK(cudaMemset(x, 0x3c, (size_t) 17408 * seq * 2));

    for (int si = 0; si < 6; ++si) {
        const int k = shapes[si][0], n = shapes[si][1];
        const int m = n, nn = seq, kk = k;
        const double flop = 2.0 * k * n * (double) seq;
        const int it = std::max(3, (int) (3.0e11 / flop));
        float alpha = 1.0f, beta = 0.0f;
        // Reference through cublasGemmEx (what ships).
        CK(cudaMemset(yRef, 0, (size_t) n * seq * 4));
        const float msRef = timeMs(it, [&] {
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, m, nn, kk, &alpha, w,
                         CUDA_R_16F, n, x, CUDA_R_16F, k, &beta, yRef, CUDA_R_32F,
                         n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        });
        std::printf("\n=== k=%d n=%d seq=%d  (%.1f GFLOP)\n", k, n, seq,
                    flop / 1e9);
        std::printf("cublasGemmEx               %8.3f ms  %6.2f TFLOP/s\n", msRef,
                    flop / (msRef * 1e-3) / 1e12);

        cublasLtMatmulDesc_t opDesc = nullptr;
        cublasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr;
        cublasLtMatmulPreference_t pref = nullptr;
        if (cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
                    CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&Ad, CUDA_R_16F, m, kk, n) !=
                    CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&Bd, CUDA_R_16F, kk, nn, k) !=
                    CUBLAS_STATUS_SUCCESS ||
            cublasLtMatrixLayoutCreate(&Cd, CUDA_R_32F, m, nn, n) !=
                    CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
            std::printf("cublasLt descriptor create failed\n");
            return 1;
        }
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN,
                                       sizeof(opN));
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN,
                                       sizeof(opN));
        cublasLtMatmulPreferenceSetAttribute(
                pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &wsBytes,
                sizeof(wsBytes));
        const int want = 24;
        cublasLtMatmulHeuristicResult_t heur[24];
        int got = 0;
        cublasLtMatmulAlgoGetHeuristic(lh, opDesc, Ad, Bd, Cd, Cd, pref, want, heur,
                                       &got);
        std::printf("cublasLt returned %d candidates\n", got);
        double best = 1e9;
        int bestIdx = -1;
        for (int i = 0; i < got; ++i) {
            int tile = -1, stages = -1, splitK = -1;
            cublasLtMatmulAlgoConfigGetAttribute(
                    &heur[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile),
                    nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(
                    &heur[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages,
                    sizeof(stages), nullptr);
            cublasLtMatmulAlgoConfigGetAttribute(
                    &heur[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK,
                    sizeof(splitK), nullptr);
            CK(cudaMemset(y, 0, (size_t) n * seq * 4));
            const float ms = timeMs(it, [&] {
                cublasLtMatmul(lh, opDesc, &alpha, w, Ad, x, Bd, &beta, y, Cd, y, Cd,
                               &heur[i].algo, ws, wsBytes, nullptr);
            });
            if (ms < best) {
                best = ms;
                bestIdx = i;
            }
            std::printf("  lt[%2d] tile=%4d stages=%d splitK=%d  %8.3f ms  %6.2f "
                        "TFLOP/s%s\n",
                        i, tile, stages, splitK, ms, flop / (ms * 1e-3) / 1e12,
                        ms < msRef * 0.98 ? "  <-- beat cublasGemmEx" : "");
        }
        std::printf("best lt[%d] %.3f ms vs GemmEx %.3f ms  (%.2fx)\n", bestIdx, best,
                    msRef, msRef / best);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatmulDescDestroy(opDesc);
        cublasLtMatrixLayoutDestroy(Ad);
        cublasLtMatrixLayoutDestroy(Bd);
        cublasLtMatrixLayoutDestroy(Cd);
    }
    std::printf("\nSM clock at end: %.2f GHz\n", clockNow());
    return 0;
}
