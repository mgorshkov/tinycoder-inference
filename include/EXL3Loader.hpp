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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "IModelLoader.hpp"
#include "SafetensorsReader.hpp"

namespace tinycoder {

    /// @brief EXL3 (EXL3) checkpoint loader.
    ///
    /// EXL3 checkpoints are Hugging Face directories: a `config.json`, one or
    /// more `*.safetensors` shards, and (usually) a `tokenizer.json`. The linear
    /// weights are stored as 3-D "trellis" tensors (see EXL3Codec) plus folded
    /// Hadamard sign vectors; the rest (norms, biases, embeddings) are plain
    /// F16/F32/BF16.
    ///
    /// This loader is a CORRECTNESS-FIRST bridge: it dequantizes every EXL3
    /// weight (and copies every plain tensor) into GGUF-named, F16/F32 buffers,
    /// so `Model::loadWeights` and the whole forward pass run unchanged. Native
    /// quantized EXL3 GEMV kernels (GPU) consume the raw trellis directly and
    /// are a later optimization; this path is the reference the kernels are
    /// checked against.
    ///
    /// Name mapping: HF names (`model.layers.N.self_attn.q_proj`,
    /// `model.embed_tokens.weight`, `lm_head.weight`, `model.norm.weight`, ...)
    /// are translated to the GGUF names the engine already knows
    /// (`blk.N.attn_q.weight`, `token_embd.weight`, `output.weight`,
    /// `output_norm.weight`, ...).
    class EXL3Loader : public IModelLoader {
    public:
        EXL3Loader();
        ~EXL3Loader() override;
        EXL3Loader(const EXL3Loader &) = delete;
        EXL3Loader &operator=(const EXL3Loader &) = delete;

        /// @brief Cheap path sniff: true for a `*.safetensors` file or a directory
        /// that looks like an EXL3/HF checkpoint (has *.safetensors and no .gguf).
        /// Used by Model::load to pick the loader without reading any bytes.
        static bool isExl3Path(const std::string &path);

        bool load(const std::string &path) override;
        bool loadMetadata(const std::string &path) override;

        const ModelConfig &config() const override { return config_; }
        const PrismMetadata &prism() const override { return prism_; }
        const uint8_t *getTensor(const std::string &name) const override;
        const TensorInfo *getTensorInfo(const std::string &name) const override;
        const std::vector<TensorInfo> &tensorRecords() const override { return records_; }
        bool isGguf() const override { return false; }

        /// @brief Engine architecture id mapped from config.json ("qwen2", ...).
        const std::string &detectedArchitecture() const { return detectedArch_; }
        /// @brief Raw HF `model_type` from config.json.
        const std::string &modelType() const { return modelType_; }
        /// @brief Checkpoint directory (holds config.json / tokenizer.json).
        const std::string &directory() const { return directory_; }
        /// @brief True when the checkpoint's weights use EXL3 trellis coding.
        bool hasTrellis() const { return hasTrellis_; }

        /// @brief Native packed-trellis metadata for a materialized GGML-named
        /// linear (nullptr for plain/F16 tensors or unsupported bitrates).
        bool isLazyLinear(const std::string &ggmlName) const {
            return lazy_.find(ggmlName) != lazy_.end();
        }

        const Exl3LinearInfo *nativeLinear(const std::string &ggmlName) const {
            auto it = native_.find(ggmlName);
            return it == native_.end() ? nullptr : &it->second;
        }

    private:
        // Resolved source for one logical (HF) linear: either an EXL3 trellis or
        // a plain safetensors weight.
        struct Src {
            std::string base; // HF base name (e.g. "model.layers.0.self_attn.q_proj")
            bool exl3 = false;
            bool valid = false;
        };

        bool discoverFiles(const std::string &path, std::string &configPath,
                           std::string &tokenizerPath, std::vector<std::string> &shards);
        bool parseConfig(const std::string &configPath);
        void mapArchitecture();

        /// @brief Materialize every GGUF-named tensor the engine expects for the
        /// detected architecture from the open safetensors shards.
        bool emitAllTensors();

        Src resolve(const std::vector<std::string> &candidates) const;

        /// @brief Read one reconstructed linear weight [out, in] (row-major) from
        /// a resolved source. When `recon` is non-null it receives the F32
        /// reconstruction (used by the fused-expert builders); the returned
        /// buffer is always the requested F16/F32 bytes.
        bool decodeLinear(const Src &src, uint32_t out, uint32_t in, bool forceF32,
                          std::vector<uint8_t> &outBytes, std::vector<float> *recon = nullptr);

        // Materializers. `expectedOut`/`expectedIn` are the engine's [rows, cols].
        struct LazyLinear {
            std::vector<std::string> hf;
            uint32_t out = 0;
            uint32_t in = 0;
            bool forceF32 = false;
        };
        void materializeLazy(const std::string &name);

        bool emitLinear(const std::string &ggmlName,
                        const std::vector<std::string> &hfCandidates,
                        uint32_t expectedOut, uint32_t expectedIn, bool forceF32 = false);
        // Stack `count` experts of `hfBasePrefix + "<e>." + proj` into one tensor
        // with rows = count * expectedOut (per-expert rows concatenated).
        bool emitExpertStack(const std::string &ggmlName, const std::string &hfBasePrefix,
                             const std::string &proj, uint32_t count,
                             uint32_t expectedOut, uint32_t expectedIn);
        // Fused gate+up experts: rows = count * 2 * perExpertOut, gate block first.
        bool emitFusedGateUpExperts(const std::string &ggmlName,
                                    const std::string &hfBasePrefix, uint32_t count,
                                    uint32_t perExpertOut, uint32_t expectedIn);
        // Stack `count` per-head sub-linears (e.g. linear_attn q/k/v convs),
        // each [expectedOut, expectedIn], rows concatenated.
        bool emitHeadStack(const std::string &ggmlName, const std::string &hfBasePrefix,
                           uint32_t count, uint32_t expectedOut, uint32_t expectedIn);
        // Emit the qwen35/qwen35moe gated-delta-net (linear-attention) tensors
        // for one recurrent layer.
        bool emitGdnLayer(const std::string &ggmlPrefix, const std::string &hfPrefix);

        // Record the packed trellis + signs for `ggmlName` so the CUDA adapter
        // can upload the compressed linear instead of the F16 materialization.
        bool recordNative(const std::string &ggmlName,
                          const std::vector<std::string> &hfCandidates,
                          uint32_t out, uint32_t in);
        bool loadSigns(const Src &src, uint32_t out, uint32_t in,
                       std::vector<float> &suh, std::vector<float> &svh);
        bool emit1D(const std::string &ggmlName, const std::vector<std::string> &hfCandidates,
                    bool bumpNorm = false);
        bool emitEmbedding(const std::string &ggmlName, const std::vector<std::string> &hfCandidates,
                           uint32_t rows, uint32_t cols, bool latentUntie);

        // Store a freshly built buffer and register its TensorInfo/record.
        void addTensor(const std::string &name, uint32_t ggmlType, uint32_t rows,
                       uint32_t cols, std::vector<uint8_t> bytes);
        void addTensorShape(const std::string &name, uint32_t ggmlType,
                            std::vector<uint32_t> shape, std::vector<uint8_t> bytes);

        // Decode one EXL3 trellis (or copy one plain weight) into a [out, in]
        // row-major F16 buffer. Shared by emitLinear/emitExpertStack/etc.
        bool buildLinearF16(const Src &src, uint32_t expectedOut, uint32_t expectedIn,
                            std::vector<uint8_t> &outBytes, bool forceF32 = false);

        ModelConfig config_;
        PrismMetadata prism_;// always empty (EXL3 is not Prism-folded)

        std::string directory_;
        std::string modelType_;
        std::string detectedArch_;
        std::string tokenizerPath_;

        mutable bool hasTrellis_ = false;
        bool loaded_ = false;

        std::unordered_map<std::string, Exl3LinearInfo> native_;
        std::unordered_map<std::string, LazyLinear> lazy_;
        bool lazyF16_ = false;
        std::vector<std::unique_ptr<std::vector<float>>> nativeSigns_;

        // Config-derived dims used by the tensor builders.
        uint32_t hidden_ = 0;
        uint32_t nLayers_ = 0;
        uint32_t nHeads_ = 0;
        uint32_t nKvHeads_ = 0;
        uint32_t headDim_ = 0;
        uint32_t inter_ = 0;             // dense FFN intermediate
        uint32_t vocab_ = 0;
        uint32_t nExperts_ = 0;
        uint32_t nExpertsUsed_ = 0;
        uint32_t expertInter_ = 0;
        uint32_t nSharedExperts_ = 0;
        uint32_t sharedInter_ = 0;
        uint32_t convChannels_ = 0;      // qwen35 linear-attention conv width
        uint32_t ssmInner_ = 0;          // qwen35 value dimension (dInner)
        uint32_t nVHeads_ = 0;           // qwen35 value heads
        float normEps_ = 1e-6f;          // config.json rms_norm_eps
        uint32_t fullAttentionInterval_ = 4;
        uint32_t nextnPredictLayers_ = 0;

        std::unique_ptr<SafetensorsReader> st_;
        std::vector<TensorInfo> records_;
        std::unordered_map<std::string, size_t> recIndex_;
        std::vector<std::unique_ptr<std::vector<uint8_t>>> buffers_;
        mutable TensorInfo infoScratch_;
    };

}// namespace tinycoder
