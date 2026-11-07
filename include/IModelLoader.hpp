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

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ModelConfig.hpp"

namespace tinycoder {

    // -----------------------------------------------------------------------
    // GGML tensor types (shared between every loader and Model).
    //
    // These live here (rather than in GGUFLoader.hpp) so format-agnostic code
    // — the IModelLoader interface, the EXL3 loader and its dequant kernels —
    // can reference them without pulling in the GGUF reader. GGUFLoader.hpp
    // includes this header, so existing users that did `#include
    // "GGUFLoader.hpp"` and referenced GGML_TYPE_* keep working unchanged.
    // -----------------------------------------------------------------------
    enum GGMLType : uint32_t {
        GGML_TYPE_F32 = 0,
        GGML_TYPE_F16 = 1,
        GGML_TYPE_Q4_0 = 2,
        GGML_TYPE_Q4_1 = 3,
        GGML_TYPE_Q5_0 = 6,
        GGML_TYPE_Q5_1 = 7,
        GGML_TYPE_Q8_0 = 8,
        GGML_TYPE_Q8_1 = 9,
        GGML_TYPE_Q2_K = 10,
        GGML_TYPE_Q3_K = 11,
        GGML_TYPE_Q4_K = 12,
        GGML_TYPE_Q5_K = 13,
        GGML_TYPE_Q6_K = 14,
        GGML_TYPE_Q8_K = 15,
        GGML_TYPE_IQ2_XXS = 16,
        GGML_TYPE_IQ2_XS = 17,
        GGML_TYPE_IQ3_XXS = 18,
        GGML_TYPE_IQ1_S = 19,
        GGML_TYPE_IQ4_NL = 20,
        GGML_TYPE_IQ3_S = 21,
        GGML_TYPE_IQ2_S = 22,
        GGML_TYPE_IQ4_XS = 23,
        GGML_TYPE_I8 = 24,
        GGML_TYPE_I16 = 25,
        GGML_TYPE_I32 = 26,
        // bfloat16 (2 bytes/element). Used by the Ternary-Bonsai-2-27B export for
        // the ssm_alpha / ssm_beta projection matrices (converted to F32 at load).
        GGML_TYPE_BF16 = 30,
        // Prism-private 2-bit type (Ternary-Bonsai PQ2_0 export): identical 2-bit
        // codec to Q2_0 but with one fp16 scale per 128 weights (34-byte block).
        GGML_TYPE_PQ2_0 = 142,
        // Native EXL3 trellis linear (not a GGML block format): used only as the
        // GPU DeviceMatrix tag so launchQGemv dispatches the fused EXL3 kernel.
        // The host QuantizedMatrix keeps its materialized F16 data for the CPU
        // path and carries the packed trellis in `QuantizedMatrix::exl3`.
        GGML_TYPE_EXL3 = 200,
        GGML_TYPE_COUNT = 201,
    };

    /// @brief Get the byte size of one block for a GGML quantization type.
    /// For non-quantized types (F32, etc.), returns 1.
    /// For quantized types (Q4_K, Q5_K, etc.), returns the block size (e.g., 256).
    uint32_t ggmlBlockSize(uint32_t type);

    /// @brief Get the byte size of one block's data for a GGML quantization type.
    /// For non-quantized types (F32, etc.), returns the element size (e.g., 4).
    /// For quantized types (Q4_K, Q5_K, etc.), returns the block byte size (e.g.,
    /// 144).
    uint32_t ggmlTypeSize(uint32_t type);

    /// @brief Prism "Hadamard fold" metadata (prism.hadamard.* keys) for the
    /// Ternary-Bonsai-2-27B export. When present, the listed linear weights are
    /// stored pre-folded with a per-input-dimension rotation, so the activation
    /// fed to each such matmul must undergo the same transform (sign flip +
    /// blockwise normalized Hadamard rotation). The token embedding is stored
    /// "latent" and its looked-up rows must be restored with the inverse
    /// transform. See include/PrismHadamard.hpp.
    struct PrismMetadata {
        bool present = false;
        uint32_t version = 0;
        uint32_t blockSize = 0;                 // Hadamard block size (1024)
        bool gdnVGrouped = false;               // ssm_out value-head tiled→grouped perm
        std::vector<std::string> foldedWeights; // weights needing forward fold
        std::vector<std::string> inverseWeights;// latent tables needing inverse
        // Per-input-dimension sign vectors (width → ±1 values), concatenated in
        // prism.hadamard.sign_values and sliced by prism.hadamard.sign_widths.
        std::vector<uint32_t> signWidths;
        std::vector<int8_t> signValues;// concatenated per-width ±1 values
    };

    /// @brief Host-side description of one native EXL3 trellis linear.
    ///
    /// The loader owns the trellis/sign buffers (they live for the model's
    /// lifetime); this POD is what the GPU adapter reads when it uploads the
    /// compressed weights instead of the materialized F16 copy.
    struct Exl3LinearInfo {
        const uint16_t *trellis = nullptr;// [k/16][n/16][16*K] uint16
        uint64_t trellisBytes = 0;
        const float *suh = nullptr;// length k (input-side signs), null = identity
        const float *svh = nullptr;// length n (output-side signs), null = identity
        uint32_t k = 0;            // in features  (multiple of 128)
        uint32_t n = 0;            // out features (multiple of 128)
        uint32_t K = 0;            // integer bitrate (1..8)
        int32_t codebook = 0;      // 0 = 3INST, 1 = MCG, 2 = mul1
        bool frac = false;         // half-integer bitrate (mul1 only)
        uint32_t tileU16 = 0;      // uint16 words per 16x16 tile
    };

    /// @brief Format-neutral tensor record. The GGUF loader fills offset/size/
    /// shape/type straight from the GGUF header; the EXL3 loader synthesizes a
    /// record for each materialized (dequantized) tensor so the rest of the
    /// engine sees the same interface it always has.
    struct TensorInfo {
        std::string name;
        uint64_t offset = 0;         // Offset into the loader-owned buffer
        uint64_t size = 0;           // Bytes (0 when not precomputed)
        std::vector<uint32_t> shape;// Dimensions (numpy-style, reversed for GGUF)
        uint32_t type = 0;          // GGML tensor type enum
    };

    /// @brief Abstract model file reader.
    ///
    /// Model loads weights exclusively through this interface, so a new
    /// container format only has to materialize GGUF-named tensors (in a GGUF
    /// type) and fill a ModelConfig. See GGUFLoader (native GGUF v3) and
    /// EXL3Loader (EXL3 safetensors → dequantized F16/F32 tensors).
    ///
    /// Lifetime contract (identical to the previous GGUFLoader-only flow): the
    /// loader owns the buffers returned by getTensor() and must outlive every
    /// QuantizedMatrix that aliases them (Model::weightLoader_).
    class IModelLoader {
    public:
        virtual ~IModelLoader() = default;

        /// @brief Load a model file and make its tensor data available.
        /// @return true if loading succeeded.
        virtual bool load(const std::string &path) = 0;

        /// @brief Read only the metadata/config (no tensor payloads).
        /// @return true if metadata was read successfully.
        virtual bool loadMetadata(const std::string &path) = 0;

        /// @brief Model configuration parsed from the file.
        virtual const ModelConfig &config() const = 0;

        /// @brief Prism Hadamard-fold metadata (typically empty/present==false).
        virtual const PrismMetadata &prism() const = 0;

        /// @brief Raw tensor bytes by name (nullptr when absent).
        virtual const uint8_t *getTensor(const std::string &name) const = 0;

        /// @brief Tensor descriptor by name (nullptr when absent). The returned
        /// pointer is backed by a per-loader thread-local and is only valid
        /// until the next getTensorInfo() call on the same thread.
        virtual const TensorInfo *getTensorInfo(const std::string &name) const = 0;

        /// @brief Read-only view of every enumerated tensor record.
        virtual const std::vector<TensorInfo> &tensorRecords() const = 0;

        /// @brief True for native GGUF containers (used only for diagnostics and
        /// to decide whether a GGUF-embedded tokenizer can be read in place).
        virtual bool isGguf() const { return false; }
    };

}// namespace tinycoder
