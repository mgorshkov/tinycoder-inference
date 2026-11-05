// int8_mma_peak.cu — what is the real s8 IMMA rate of this SM?
//
// The whole prefill campaign depends on one number: the fp32-accumulate int8
// tensor roof, i.e. how many MAC/clk/SM `mma.sync.aligned.m8n8k16.s8.s8.s32`
// retires.  AGENTS.md quotes 53.8 TFLOPS *for fp16 HMMA at the 1.55 GHz
// nominal clock*; that is not directly comparable with a 1965 MHz measurement
// and it says nothing about IMMA.  This probe measures the IMMA roof with
// register-only operands (no shared memory, no epilogue), so the result is an
// upper bound no kernel in this tree can exceed.
//
// Build/run:
//   nvcc -arch=sm_75 -O3 -o /tmp/int8_mma_peak tools/int8_mma_peak.cu && /tmp/int8_mma_peak
#include <cstdio>
#include <cuda_runtime.h>

namespace {

    __global__ void immaPeak(float *out, int iters) {
        // 8 independent accumulators -> the MMA pipe never waits on a result.
        int a0 = threadIdx.x, a1 = threadIdx.x + 1;
        int c[8][2] = {};
#pragma unroll 1
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                asm volatile(
                        "mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                        "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                        : "+r"(c[i][0]), "+r"(c[i][1])
                        : "r"(a0), "r"(a1));
            }
        }
        int s = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i) s += c[i][0] + c[i][1];
        if (s == 0x7fffffff) out[threadIdx.x] = static_cast<float>(s);
    }

}// namespace

int main() {
    float *out = nullptr;
    cudaMalloc(&out, 1024 * sizeof(float));
    const int blocks = 68 * 4;// enough to fill every SM at high occupancy
    const int iters = 200000;
    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    // warm up and hold the boost clock
    for (int r = 0; r < 200; ++r) immaPeak<<<3 * sm, 128>>>(out, iters / 10);
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    float best = 1e9f;
    for (int r = 0; r < 5; ++r) {
        cudaEventRecord(a);
        immaPeak<<<blocks, 128>>>(out, iters);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    // per block: 4 warps x 8 accumulators x iters IMMA, each 8*8*16 MACs.
    const double macs = static_cast<double>(blocks) * 4 * 8 * iters * (8.0 * 8.0 * 16.0);
    const double tflops = 2.0 * macs / (static_cast<double>(best) * 1e-3) / 1e12;
    std::printf("SMs=%d  best=%.3f ms  IMMA roof = %.2f TFLOPS "
                "(%.0f MAC/clk/SM at 1965 MHz)\n",
                sm, best, tflops,
                macs / (static_cast<double>(best) * 1e-3) / (sm * 1.965e9));
    cudaFree(out);
    return 0;
}
