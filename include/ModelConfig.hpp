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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tinycoder {

    /// @brief Supported model architecture identifiers (from GGUF
    /// `general.architecture`).
    static constexpr const char *ARCH_QWEN2 = "qwen2";
    static constexpr const char *ARCH_GEMMA4 = "gemma4";
    static constexpr const char *ARCH_QWEN35MOE = "qwen35moe";
    static constexpr const char *ARCH_QWEN35 = "qwen35";
    // Inkling-2-Mini looped MoE.  Two on-disk layouts exist for the same
    // tensors: the llama.cpp mainline `inkling2_loop` layout (24 physical
    // blocks, loop.start/end/count, per-layer loop_msg_query) and the Unsloth
    // expanded layout (`inkling2`, block_count = 48 logical cache slots, only
    // 24 physical block tensors, loop.core_start/core_end, global
    // loop_res_query / loop_kv_res_query rows).  Both are accepted.
    static constexpr const char *ARCH_INKLING2 = "inkling2";
    static constexpr const char *ARCH_INKLING2_LOOP = "inkling2_loop";

    /// @brief Check if an architecture string is supported.
    inline bool isSupportedArchitecture(const std::string &arch) {
        return arch == ARCH_QWEN2 ||
               arch == ARCH_GEMMA4 ||
               arch == ARCH_QWEN35MOE ||
               arch == ARCH_QWEN35 ||
               arch == ARCH_INKLING2 ||
               arch == ARCH_INKLING2_LOOP;
    }

    /// @brief Check if a model name is in the supported list.
    inline bool isSupportedModel(const std::string &name) {
        // Qwen2 models
        if (name == "Qwen2.5 Coder 0.5B" ||
            name == "Qwen2.5 Coder 1.5B" ||
            name == "Qwen2.5 Coder 1.5B Instruct" ||
            name == "Qwen2.5 Coder 1.5B Instruct GGUF" ||
            name == "Qwen2.5 Coder 7B Instruct")
            return true;
        // Gemma4 models
        if (name == "Gemma4 Coding Merged Fp16" ||
            name == "Gemma-4 26B-A4B IT (smart Q4_0, QAT-lossless)")
            return true;
        // Qwen35MoE models
        if (name == "Safetensors" ||
            name == "Qwen3.6 35B A3B Claude 4.7 Opus Reasoning Distilled" ||
            name == "Qwen3.6-35B-A3B")
            return true;
        // Qwen35 (dense) models — Qwen3.8 series and the Ornith-1.5 dense
        // GDN hybrid (Ornith-1.5-9B, GGUF general.architecture = "qwen35").
        if (name == "Qwen3.8-27B" ||
            name == "Qwen3.8-27B-UD" ||
            name == "Ornith 1.5 9B")
            return true;
        // Inkling-2-Mini (looped MoE)
        if (name == "Inkling-2-Mini" ||
            name == "Inkling 2 Mini" ||
            name.find("Inkling") != std::string::npos)
            return true;
        return false;
    }

    struct ModelConfig {
        std::string modelPath;    ///< Path to the GGUF file
        std::string tokenizerPath;///< Path to the tokenizer (optional, can be
                                  ///< embedded in GGUF)
        std::string architecture; ///< Model architecture (e.g., "qwen2", "gemma4", "qwen35moe")
        std::string modelName;    ///< Model name (e.g., "Qwen2.5 Coder 1.5B")
        std::string chatTemplate; ///< Jinja chat template from GGUF metadata (tokenizer.chat_template)

        // Architecture parameters (from model metadata)
        uint32_t vocabSize = 151936;     // Qwen2.5-Coder tokenizer vocab size
        uint32_t hiddenSize = 1536;      // 0.5B: 1024, 1.5B: 1536, 7B: 3584
        uint32_t intermediateSize = 8960;// 0.5B: 4096, 1.5B: 8960, 7B: 18944
        uint32_t numLayers = 28;         // 0.5B: 24, 1.5B: 28, 7B: 28
        uint32_t numAttentionHeads = 12; // 0.5B: 16, 1.5B: 12, 7B: 28
        uint32_t numKVHeads = 2;         // 0.5B: 4, 1.5B: 2, 7B: 4 (GQA)
        /// Context window.  0 = auto: Model::load grants 2048 (today's
        /// default) unless memory cannot hold it, and reports the
        /// memory-derived maximum.  Set it with Model::setContextLength
        /// before load, or it arrives through tc_engine_load's max_seq_len.
        uint32_t maxSeqLen = 0;
        float ropeTheta = 1000000.0f;
        uint32_t headDim = 128;// hiddenSize / numAttentionHeads

        // Gemma4-specific parameters
        float finalLogitSoftcapping = 0.0f;  // 0 = disabled
        uint32_t expertCount = 0;            // 0 = dense (non-MoE)
        uint32_t expertUsedCount = 0;        // 0 = dense (non-MoE)
        uint32_t expertFeedForwardLength = 0;// per-expert FFN size
        uint32_t ropeDimensionCount = 0;     // 0 = use headDim

        // Qwen35MoE-specific parameters
        uint32_t fullAttentionInterval = 0;// 0 = all layers use attention
        uint32_t ssmInnerSize = 0;
        uint32_t ssmStateSize = 0;
        uint32_t ssmConvKernel = 0;
        uint32_t ssmGroupCount = 0;
        uint32_t ssmTimeStepRank = 0;
        uint32_t expertSharedFeedForwardLength = 0;
        uint32_t nextnPredictLayers = 0;

        // Qwen35 (dense) specific parameters
        // (ropeDimensionCount is declared above in the Gemma4 group and reused
        //  here as n_rot — number of rotated dims per head, 0 = use headDim)
        uint32_t attentionKeyLength = 0;                 // 0 = derived from headDim (full-attn heads)
        uint32_t attentionValueLength = 0;               // 0 = derived from headDim
        uint32_t ropeDimensionSections[4] = {0, 0, 0, 0};// MRoPE sections (t,h,w,e)

        // ---- Inkling-2-Mini (looped MoE) ----
        // numLayers is always the number of PHYSICAL blocks (24).  The loop
        // re-runs blocks [loopStart, loopEnd) loopCount times; every pass owns
        // its own KV cache and short-convolution state, so the cache has
        // numKvSlots rows = physical + (loopCount-1)*(loopEnd-loopStart).
        uint32_t numPhysicalLayers = 0;  // 0 = same as numLayers
        uint32_t numKvSlots = 0;         // 0 = same as numLayers
        uint32_t loopStart = 0;
        uint32_t loopEnd = 0;
        uint32_t loopCount = 0;
        uint32_t loopGateRank = 0;
        uint32_t attnResBlockSize = 0;   // depth-attention block size (6)
        uint32_t relativeLength = 0;     // learned relative-bias features (16)
        uint32_t relativeExtent = 0;     // max relative distance (1024)
        uint32_t shortconvLCache = 0;    // causal depthwise conv taps (4)
        uint32_t slidingWindow = 0;      // SWA window (1024)
        uint32_t vocabSizeUnpadded = 0;  // LM head rows actually used (200058)
        std::vector<uint8_t> slidingWindowPattern;// per physical layer; empty = all full
        float qkClip = 0.0f;             // per-head L2 clip threshold (22.627)
        float relativeClip = 0.0f;       // relative-bias feature clip (8.0)
        float logScaleFloor = 0.0f;      // log-scaled positions floor (8192)
        float logScaleAlpha = 0.0f;      // log-scaled positions alpha (0.1)
        float situBeta = 0.0f;           // SITU gate beta (4.0)
        float situLinearBeta = 0.0f;     // SITU up beta (25.0)
        float expertWeightsScale = 0.0f; // router weight scale after norm (8.0)
        bool inklingExpandedLayout = false;// Unsloth 48-slot layout


        // ---- Gated-delta-net q/k -> value-head pairing order ----
        // The gated delta net has nKHeads (ssmGroupCount) key/query heads and
        // nVHeads (ssmTimeStepRank) value heads, nVHeads a multiple of nKHeads;
        // every value head reads one key head.  Two on-disk layouts disagree
        // about WHICH:
        //   grouped (HF / exllamav3 / Prism): value head hv reads key head
        //     hv / (nVHeads / nKHeads)   -- `repeat_interleave`.
        //   tiled (llama.cpp GGUF converter — `_reorder_v_heads` in
        //     conversion/qwen.py): the converter permutes every V-side tensor
        //     (v, z, alpha, beta, A_log, dt_bias, out_proj rows) into tiled
        //     order so a plain broadcast suffices, and value head hv then reads
        //     key head hv % nKHeads.
        // Picking the wrong one pairs 30 of 32 heads with the wrong value head
        // on Ornith-1.5-9B (measured: layer-0 branch output 23 % low, logits 2x
        // flat, generation becomes token salad inside <think>).
        bool gdnVHeadsTiled = false;

        // ---- Prism Hadamard-fold (Ternary-Bonsai-2-27B PQ2_0) ----
        // When prismFold is set, the model's linear weights were stored folded
        // with a per-input-dimension normalized Walsh-Hadamard rotation (plus an
        // optional sign flip), and the activation fed to every folded matmul
        // must be transformed the same way; the token embedding table is stored
        // latent and restored with the inverse transform. See
        // include/PrismHadamard.hpp and GGUFLoader::PrismMetadata.
        bool prismFold = false;       // prism.hadamard.version present
        uint32_t prismBlockSize = 0;  // Hadamard block size (1024)
        bool prismGdnVGrouped = false;// ssm_out needs the tiled→grouped perm

        // Runtime
        // CPU worker threads for this model's own pool; 0 = auto (inherit the
        // already-initialized process pool, else all logical CPUs). Set by the
        // loader/frontend (tc_engine_load) and by bench --threads.
        uint32_t nThreads = 0;

        /// @brief Estimate total memory needed to load this model.
        /// @param contextTokens Token window to charge the K/V cache for; 0
        ///        means "auto", the 2048 default.  This must be the window the
        ///        engine will grant, not `maxSeqLen`: that field is the
        ///        checkpoint's *advertised* limit (1,048,576 tokens for
        ///        Inkling-2-Mini), and charging the cache at that size reported
        ///        a 197 GB estimate for a model loaded with 2048.
        /// @return Estimated memory in bytes
        uint64_t estimatedMemoryBytes(uint32_t contextTokens = 2048) const {
            // Quantized weights: use a rough estimate based on hidden size and layers
            // For a typical transformer: ~2 * hiddenSize * intermediateSize * numLayers * bytes_per_weight
            // This is a rough estimate; actual size depends on quantization type
            uint64_t weightBytes = 700ULL * 1024 * 1024;// ~700 MB default

            // Token embeddings
            uint64_t numEmbedElements = static_cast<uint64_t>(vocabSize) * hiddenSize;
            uint64_t numEmbedBlocks = (numEmbedElements + 255) / 256;
            uint64_t embedBytes = numEmbedBlocks * 170;

            // KV cache (F32) for the granted window, capped by what the
            // checkpoint can serve.
            uint64_t seqLen = contextTokens ? contextTokens : 2048u;
            if (maxSeqLen != 0) seqLen = std::min<uint64_t>(seqLen, maxSeqLen);
            uint64_t kvCacheBytes = static_cast<uint64_t>(numLayers) * seqLen *
                                    numKVHeads * headDim * sizeof(float) * 2;

            // Norms (F32)
            uint64_t normBytes =
                    static_cast<uint64_t>(numLayers + 1) * hiddenSize * sizeof(float);

            // Activations (F32, temporary buffers during forward pass)
            uint64_t activationBytes =
                    static_cast<uint64_t>(hiddenSize) * sizeof(float) * 20;

            return weightBytes + embedBytes + kvCacheBytes + normBytes +
                   activationBytes;
        }
    };

    /// @brief Inference parameters for text generation.
    struct InferenceParams {
        int32_t maxTokens = 2048;  // Max tokens to generate
        float temperature = 0.7f;  // Sampling temperature
        float topP = 0.9f;         // Nucleus sampling threshold
        float topK = 40.0f;        // Top-K sampling
        float repeatPenalty = 1.1f;// Repetition penalty
        int32_t repeatLastN = 64;  // Number of tokens to consider for penalty
        float presencePenalty = 0.0f;
        float frequencyPenalty = 0.0f;
        uint32_t seed = 0;// 0 = random
    };

}// namespace tinycoder
