/*
 * exl3_gemm_acc_probe -- is fp16 accumulate actually 2x on this card?
 *
 * Times the EXL3 prefill GEMM exactly as GPUComputeForward::gemmExl3Batch
 * issues it (A = the materialised weight as an (n x k) column-major operand
 * with lda = n, B = the fp16 activation, C in fp32 or fp16) at the 27B's real
 * per-projection shapes, for both accumulate types:
 *
 *   mode 0: C fp32, compute fp32   (what ships today)
 *   mode 1: C fp16, compute fp16   (the 2x-rate tensor path)
 *
 * It also prints the achieved TFLOP/s and reads the SM clock, so the result can
 * be checked against the card's published tensor peak instead of trusted.
 *
 * Build (no CMake target needed):
 *   nvcc -O3 -arch=sm_75 tools/exl3_gemm_acc_probe.cu -lcublas -o /tmp/gemm_acc_probe
 * Run:
 *   /tmp/gemm_acc_probe [k=n-shape ...]
 */
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CK(expr)                                                                     \
    do {                                                                             \
        cudaError_t e_ = (expr);                                                     \
        if (e_ != cudaSuccess) {                                                     \
            std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #expr,      \
                         cudaGetErrorString(e_));                                    \
            std::exit(1);                                                            \
        }                                                                            \
    } while (0)

static float clockGHz() {
    FILE *f = popen(
            "nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits 2>/dev/null",
            "r");
    if (!f) return 0.0f;
    float mhz = 0.0f;
    if (std::fscanf(f, "%f", &mhz) != 1) mhz = 0.0f;
    pclose(f);
    return mhz / 1000.0f;
}

// One timed configuration: returns tok/s equivalent for seq=512 (i.e. how many
// 512-token prefill passes of this projection fit in one second) and TFLOPS.
static void run(const char *label, int k, int n, int seq, int mode, int iters,
                cublasHandle_t h, cudaStream_t s, const __half *w, const __half *x,
                float *yf, __half *yh) {
    float alpha = 1.0f, beta = 0.0f;
    const cublasOperation_t opA = CUBLAS_OP_N;
    const int m = n, nn = seq, kk = k;
    const cudaDataType_t acc = mode == 0 ? CUDA_R_32F : CUDA_R_16F;
    void *c = mode == 0 ? static_cast<void *>(yf) : static_cast<void *>(yh);
    const cudaDataType_t ctype = mode == 0 ? CUDA_R_32F : CUDA_R_16F;

    // warm-up
    cublasStatus_t st = cublasGemmEx(h, opA, CUBLAS_OP_N, m, nn, kk, &alpha, w,
                                     CUDA_R_16F, n, x, CUDA_R_16F, k, &beta, c,
                                     ctype, n, acc, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::printf("%-28s k=%5d n=%5d mode=%d UNSUPPORTED (status %d)\n", label, k, n,
                    mode, (int) st);
        return;
    }
    CK(cudaStreamSynchronize(s));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, s));
    for (int i = 0; i < iters; ++i)
        cublasGemmEx(h, opA, CUBLAS_OP_N, m, nn, kk, &alpha, w, CUDA_R_16F, n, x,
                     CUDA_R_16F, k, &beta, c, ctype, n, acc,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    const double flop = 2.0 * k * n * (double) seq;
    const double tflops = flop * iters / (ms * 1e-3) / 1e12;
    std::printf("%-28s k=%5d n=%5d seq=%d %s: %8.3f ms  %6.2f TFLOP/s  (%.1f GHz)\n",
                label, k, n, seq, mode == 0 ? "fp32-acc" : "fp16-acc", ms / iters,
                tflops, clockGHz());
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

int main(int argc, char **argv) {
    (void) argc;
    (void) argv;
    cublasHandle_t h;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return 1;
    cudaStream_t s = nullptr;
    cublasSetStream(h, s);
    const int seq = 512;
    // The 27B's per-layer linear geometries (k x n), k = contraction length.
    const int shapes[][2] = {
            {17408, 5120},  // ffn_down
            {5120, 17408},  // ffn_gate / ffn_up
            {5120, 10240},  // in_proj_qkv (recurrent layers)
            {5120, 12288},  // attn_q (full-attention layers)
            {6144, 5120},   // attn_output / out_proj
            {5120, 1024},   // attn_k / attn_v
    };
    const int nShapes = (int) (sizeof(shapes) / sizeof(shapes[0]));
    const size_t maxW = (size_t) 17408 * 17408;
    __half *w = nullptr, *x = nullptr, *yh = nullptr;
    float *yf = nullptr;
    CK(cudaMalloc(&w, maxW * sizeof(__half)));
    CK(cudaMalloc(&x, (size_t) 17408 * seq * sizeof(__half)));
    CK(cudaMalloc(&yf, (size_t) 17408 * seq * sizeof(float)));
    CK(cudaMalloc(&yh, (size_t) 17408 * seq * sizeof(__half)));
    // Fill with plausible magnitudes so the fp16 accumulator behaves like the
    // real RAW weights (O(1e3)); this probe times the GEMM, not the values.
    CK(cudaMemset(w, 0x3c, maxW * sizeof(__half)));
    CK(cudaMemset(x, 0x3c, (size_t) 17408 * seq * sizeof(__half)));

    for (int i = 0; i < nShapes; ++i) {
        const int k = shapes[i][0], n = shapes[i][1];
        const double iters = std::max(2.0, 2.0e11 / (2.0 * k * n * (double) seq));
        for (int mode = 0; mode < 2; ++mode)
            run("exl3 27B", k, n, seq, mode, (int) iters, h, s, w, x, yf, yh);
    }
    // Aggregate: the whole model's per-token GEMM (sum k*n = 24.75e9) at each
    // rate, expressed as a pp512-equivalent upper bound (GEMM only).
    std::printf("\nper-token GEMM params = 24.75e9; pp512 GEMM-only ceiling:\n");
    std::printf("  input-precision fp16, fp32 accumulate: see measured TFLOP/s above\n");
    std::printf("  implied pp512 if the GEMM ran alone at X TFLOP/s = "
                "X*1e12/(2*24.75e9)\n");
    CK(cudaFree(w));
    CK(cudaFree(x));
    CK(cudaFree(yf));
    CK(cudaFree(yh));
    cublasDestroy(h);
    return 0;
}
