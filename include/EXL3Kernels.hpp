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

/*
 * EXL3 trellis decode + reconstruct + GEMV CUDA kernels (runs on Turing /
 * Ampere / Ada, incl. RTX 2080 Ti / 3090 / 4090 / 5090).
 *
 * This is the device-side port of the CPU reference codec in EXL3Codec.{hpp,cpp}.
 * It lets an EXL3 trellis weight stay COMPRESSED in VRAM (2-4 bits/weight via the
 * packed code) and be materialized on demand into a reusable F32 scratch, which
 * frees VRAM versus the correctness-first path that expands every linear to F16
 * at load time (see EXL3Loader).
 *
 * The pipeline is a literal device transcription of LinearEXL3.get_weight_tensor
 * and therefore differs from the CPU path only by ordinary F32 rounding:
 *   scratch = decode(trellis)                      (codebook + tensor-core perm)
 *   scratch = leftH128(scratch) * suh              (128-wide Hadamard over k)
 *   scratch = rightH128(scratch) * svh             (128-wide Hadamard over n)
 *   out[n]  = sum_k x[k] * scratch[k][n]           (row-major GEMV)
 *
 * Turing-oriented: no cp.async and no quantized tensor cores are used -- the
 * kernels are plain F32 smem/register code (they are tuned for the 68 SMs of
 * the RTX 2080 Ti, and reconstruct is a one-off per layer, so the bandwidth of
 * the GEMV dominates).  The same kernels run on Ampere/Ada and, via the
 * embedded forward-compatible PTX, on newer GPUs.  The trellis stays resident
 * so repeated GEMVs only touch packed 2-4 bit weights when reconstructing lazily.
 *
 * The API is intentionally self-contained (no Model coupling): a host caller
 * uploads one linear's trellis + sign vectors, optionally materializes the full
 * [k][n] weight for parity checks, and runs GEMVs.  Wiring it into the Model GPU
 * offload path is a follow-up; this translation unit is the verified kernel
 * layer.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#ifdef USE_CUDA

#include <cuda_runtime.h>

namespace tinycoder::exl3gpu {

    /// @brief One EXL3 linear's host-side description.
    ///
    /// The trellis is the raw safetensors tensor [k/16, n/16, 16*K] uint16; the
    /// sign vectors may be supplied either already expanded to float (±1, length
    /// k / n) or as the packed ±1 bitfields (`suhPacked` / `svhPacked`, k/16 or
    /// n/16 uint16 words).  The packed form is what EXL3 stores; expanding it on
    /// device avoids uploading a full float vector.
    struct Exl3HostLinear {
        const uint16_t *packed = nullptr;    // trellis codes [k/16 * n/16 * tileU16]
        uint32_t k = 0;                      // in_features  (must be % 128 == 0)
        uint32_t n = 0;                      // out_features (must be % 128 == 0)
        uint32_t bitrate10 = 0;              // K * 10: 20 = 2.0 bpw, 25 = 2.5 bpw ...
        int32_t codebook = 0;                // 0 = 3INST, 1 = MCG, 2 = MUL1
        const float *suh = nullptr;          // per-row (k) signs, or null
        const float *svh = nullptr;          // per-col (n) signs, or null
        const uint16_t *suhPacked = nullptr; // packed per-row signs (k/16 words)
        const uint16_t *svhPacked = nullptr; // packed per-col signs (n/16 words)
    };

    /// @brief True when a CUDA device is present and the module can run.
    bool available();

    /// @brief One uploaded EXL3 linear + its device buffers.
    struct Exl3DeviceLinear {
        void *packed = nullptr;   // device trellis (uint16*)
        float *suh = nullptr;     // device signs (float*), null when identity
        float *svh = nullptr;     // device signs (float*), null when identity
        float *scratch = nullptr; // device reconstructed weight [k][n] F32
        uint32_t k = 0;
        uint32_t n = 0;
        uint32_t tileU16 = 0;     // uint16 words per 16x16 tile
        uint32_t bitrate10 = 0;
        int32_t codebook = 0;
        bool frac = false;        // half-integer bitrate (mul1 only)
        bool reconstructed = false;
    };

    /// @brief Upload one linear's trellis (+signs) to the device.  Returns an
    /// opaque handle (>0) or -1 on failure (err set).  The handle owns device
    /// memory until releaseLinear().
    int64_t uploadLinear(const Exl3HostLinear &lin, std::string &err);

    /// @brief Release every device buffer owned by `handle`.
    void releaseLinear(int64_t handle);

    /// @brief Materialize the full weight for `handle` into `outHost`
    /// (row-major [k][n], k*n floats) via the decode -> leftH*suh -> rightH*svh
    /// pipeline.  Used for parity checks against EXL3Codec::reconstructMatrix.
    bool reconstruct(int64_t handle, float *outHost, std::string &err);

    /// @brief y = W @ x for the uploaded linear: W is [n][k] in the engine's
    /// row-major convention (W[row][col] = reconstructed[k=col][n=row]), x has k
    /// elements and out has n.  Reconstructs lazily on the first call.
    bool gemv(int64_t handle, const float *xHost, float *outHost, std::string &err);

    // =======================================================================
    // Fused (native) GEMV path
    //
    // The reconstruct()-then-F32-GEMV route above materializes a whole linear
    // ([k][n] F32) and re-reads it every token: 97 GB/token for Qwen3.5-27B
    // (5.3 tok/s on a 2080 Ti).  The fused path keeps the packed trellis
    // resident and decodes it in the GEMV, using the orthogonal factorisation
    //
    //     W[k,n] = Hk (suh ⊙ What) Hn (svh ⊙ ·)
    //     y = svh ⊙ Hn · (What^T · (Hk (suh ⊙ x)))
    //
    // so the Hadamard transforms are applied to the ACTIVATION (O(k) and
    // O(n)) instead of to the O(k*n) weight.  Per token this reads only the
    // packed trellis (~5.4 GB at 1.6 bpw) — the same traffic the EXL3 reference
    // native EXL3 kernels stream.
    // =======================================================================
    struct Exl3FusedLinear {
        const void *trellis = nullptr;// device uint16 trellis
        const float *suh = nullptr;   // device signs, length k (null = identity)
        const float *svh = nullptr;   // device signs, length n (null = identity)
        uint32_t k = 0;               // in features  (multiple of 128)
        uint32_t n = 0;               // out features (multiple of 128)
        uint32_t K = 0;               // integer bitrate (1..8)
        int32_t codebook = 0;         // 0 = 3INST, 1 = MCG, 2 = mul1
        bool frac = false;            // half-integer bitrate (mul1 only)
        uint32_t tileU16 = 0;         // uint16 words per 16x16 tile
    };

    /// @brief Upload one linear's packed trellis + float sign vectors and
    /// register it for the fused GEMV.  `out` receives the device pointers (the
    /// module owns them until releaseFusedLinear).
    bool uploadFusedLinear(const Exl3HostLinear &lin, Exl3FusedLinear &out,
                           std::string &err);

    /// @brief Unregister + free the device buffers owned by `lin`.
    void releaseFusedLinear(Exl3FusedLinear &lin);

    /// @brief Register an already-device-resident linear (caller owns memory).
    void registerFusedLinear(const Exl3FusedLinear &lin);
    void unregisterFusedLinear(const void *trellisDev);
    void clearFusedLinears();

    /// @brief y = W @ x for a registered fused linear; x/y are DEVICE pointers.
    /// Launches on `stream` (no host synchronisation, so it is CUDA-graph safe).
    bool gemvFused(const void *trellisDev, const float *xDev, float *yDev,
                   cudaStream_t stream, std::string &err);

    /// @brief Free all module-owned device memory (scratch, cached handles).
    void shutdown();

}// namespace tinycoder::exl3gpu

#endif// USE_CUDA
