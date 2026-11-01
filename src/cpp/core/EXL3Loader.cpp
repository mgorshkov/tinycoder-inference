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

#include "EXL3Loader.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#include <fstream>
#include <iostream>
#include <sstream>

#include "EXL3Codec.hpp"
#include "IModelLoader.hpp"
#include "ModelConfig.hpp"
#include "ModelInternal.hpp"

namespace tinycoder {

    namespace fs = std::filesystem;

    namespace {

        // ---- Small helpers ------------------------------------------------------

        std::string dirnameOf(const std::string &path) {
            fs::path p(path);
            if (fs::is_directory(p)) {
                return p.string();
            }
            return p.parent_path().string();
        }

        std::string basenameOf(const std::string &path) {
            return fs::path(path).filename().string();
        }

        bool endsWith(const std::string &s, const std::string &suffix) {
            return s.size() >= suffix.size() &&
                   s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        }

        uint16_t floatToHalf(float f) {
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(bits));
            const uint32_t sign = (bits >> 16) & 0x8000u;
            int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
            uint32_t mant = bits & 0x7fffffu;
            if (exp <= 0) {
                if (exp < -10) {
                    return static_cast<uint16_t>(sign);
                }
                mant |= 0x800000u;
                const uint32_t shift = static_cast<uint32_t>(14 - exp);
                uint32_t half = mant >> shift;
                if ((mant >> (shift - 1)) & 1u) {
                    ++half;
                }
                return static_cast<uint16_t>(sign | half);
            }
            if (exp >= 31) {
                if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) {
                    return static_cast<uint16_t>(sign | 0x7e00u);// NaN
                }
                return static_cast<uint16_t>(sign | 0x7c00u);   // Inf / overflow
            }
            uint32_t half = (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
            if (mant & 0x1000u) {
                ++half;
            }
            return static_cast<uint16_t>(sign | half);
        }

        void writeF32(std::vector<uint8_t> &out, const std::vector<float> &v) {
            out.resize(v.size() * sizeof(float));
            std::memcpy(out.data(), v.data(), out.size());
        }

        void writeF16(std::vector<uint8_t> &out, const std::vector<float> &v) {
            out.resize(v.size() * sizeof(uint16_t));
            uint16_t *dst = reinterpret_cast<uint16_t *>(out.data());
            for (size_t i = 0; i < v.size(); ++i) {
                dst[i] = floatToHalf(v[i]);
            }
        }

        // Config lookup: text_config.<key> then <key>.
        const JsonValue *cfgFind(const JsonValue &root, const std::string &key) {
            const JsonValue *tc = root.find("text_config");
            if (tc && tc->type == JsonValue::Type::Object) {
                if (const JsonValue *v = tc->find(key)) {
                    return v;
                }
            }
            return root.find(key);
        }

        uint32_t cfgU32(const JsonValue &root, const std::string &key, uint32_t def) {
            const JsonValue *v = cfgFind(root, key);
            return v ? static_cast<uint32_t>(v->asU64()) : def;
        }

        float cfgF32(const JsonValue &root, const std::string &key, float def) {
            const JsonValue *v = cfgFind(root, key);
            return v ? static_cast<float>(v->asNumber()) : def;
        }

        // RoPE settings live in a nested `rope_parameters` object in the newer
        // HF configs (Qwen3.5 / Qwen3.8 / Gemma4), but older checkpoints keep
        // `rope_theta` / `partial_rotary_factor` flat.  Search the same four
        // locations exllamav3's read_rope_settings_default() does:
        //   text_config.rope_parameters.<key>, text_config.<key>,
        //   root.rope_parameters.<key>, root.<key>
        const JsonValue *cfgFindRope(const JsonValue &root, const std::string &key) {
            auto sub = [&](const JsonValue &obj) -> const JsonValue * {
                const JsonValue *rp = obj.find("rope_parameters");
                if (rp && rp->type == JsonValue::Type::Object) {
                    if (const JsonValue *v = rp->find(key)) {
                        return v;
                    }
                }
                return nullptr;
            };
            const JsonValue *tc = root.find("text_config");
            if (tc && tc->type == JsonValue::Type::Object) {
                if (const JsonValue *v = sub(*tc)) {
                    return v;
                }
                if (const JsonValue *v = tc->find(key)) {
                    return v;
                }
            }
            if (const JsonValue *v = sub(root)) {
                return v;
            }
            return root.find(key);
        }

        uint32_t cfgU32Rope(const JsonValue &root, const std::string &key, uint32_t def) {
            const JsonValue *v = cfgFindRope(root, key);
            return v ? static_cast<uint32_t>(v->asU64()) : def;
        }

        float cfgF32Rope(const JsonValue &root, const std::string &key, float def) {
            const JsonValue *v = cfgFindRope(root, key);
            return v ? static_cast<float>(v->asNumber()) : def;
        }

    }// namespace

    // =======================================================================
    // Construction / IModelLoader surface
    // =======================================================================

    EXL3Loader::EXL3Loader() = default;
    EXL3Loader::~EXL3Loader() = default;

    bool EXL3Loader::isExl3Path(const std::string &path) {
        std::error_code ec;
        if (fs::is_directory(fs::path(path), ec)) {
            // A directory with a .gguf and no safetensors is treated as GGUF; a
            // directory holding *.safetensors with no .gguf is an EXL3/HF
            // checkpoint. If both are present the caller's file argument wins;
            // here we only auto-detect, so prefer the safetensors checkpoint.
            bool hasSafetensors = false;
            bool hasGguf = false;
            for (const auto &entry: fs::directory_iterator(path, ec)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                const std::string name = entry.path().filename().string();
                if (endsWith(name, ".safetensors")) {
                    hasSafetensors = true;
                } else if (endsWith(name, ".gguf")) {
                    hasGguf = true;
                }
            }
            return hasSafetensors && !hasGguf;
        }
        return endsWith(path, ".safetensors");
    }

    const uint8_t *EXL3Loader::getTensor(const std::string &name) const {
        auto it = recIndex_.find(name);
        if (it == recIndex_.end()) {
            return nullptr;
        }
        return buffers_[it->second]->data();
    }

    const TensorInfo *EXL3Loader::getTensorInfo(const std::string &name) const {
        auto it = recIndex_.find(name);
        if (it == recIndex_.end()) {
            return nullptr;
        }
        infoScratch_ = records_[it->second];
        return &infoScratch_;
    }

    // =======================================================================
    // File discovery / config
    // =======================================================================

    bool EXL3Loader::discoverFiles(const std::string &path, std::string &configPath,
                                   std::string &tokenizerPath,
                                   std::vector<std::string> &shards) {
        std::vector<std::string> stFiles;
        if (fs::is_directory(fs::path(path))) {
            directory_ = fs::path(path).string();
            if (fs::exists(fs::path(path) / "config.json")) {
                configPath = (fs::path(path) / "config.json").string();
            }
            fs::path tok = fs::path(path) / "tokenizer.json";
            if (fs::exists(tok)) {
                tokenizerPath = tok.string();
            }
            for (const auto &entry: fs::directory_iterator(path)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                std::string name = entry.path().filename().string();
                if (endsWith(name, ".safetensors")) {
                    stFiles.push_back(entry.path().string());
                }
            }
        } else if (endsWith(path, ".safetensors")) {
            directory_ = dirnameOf(path);
            stFiles.push_back(path);
            if (fs::exists(fs::path(directory_) / "config.json")) {
                configPath = (fs::path(directory_) / "config.json").string();
            }
            fs::path tok = fs::path(directory_) / "tokenizer.json";
            if (fs::exists(tok)) {
                tokenizerPath = tok.string();
            }
        } else {
            std::cerr << "[EXL3] Not a directory or .safetensors file: " << path << std::endl;
            return false;
        }

        if (stFiles.empty()) {
            std::cerr << "[EXL3] No *.safetensors shards found in " << directory_ << std::endl;
            return false;
        }
        // Deterministic, shard-index-aware order (the model.safetensors.index.json
        // mapping is only a convenience: loading every visible shard is correct).
        std::sort(stFiles.begin(), stFiles.end());

        // Emit the single-shard file first when present so a duplicated tensor
        // resolves in favour of the canonical weights file.
        auto firstIt = std::find_if(stFiles.begin(), stFiles.end(), [](const std::string &s) {
            return basenameOf(s) == "model.safetensors";
        });
        if (firstIt != stFiles.end() && firstIt != stFiles.begin()) {
            std::iter_swap(stFiles.begin(), firstIt);
        }

        shards = std::move(stFiles);
        return true;
    }

    bool EXL3Loader::parseConfig(const std::string &configPath) {
        std::ifstream f(configPath, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "[EXL3] Cannot open config: " << configPath << std::endl;
            return false;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string text = ss.str();

        JsonValue root;
        std::string err;
        if (!JsonValue::parse(text, root, &err) || root.type != JsonValue::Type::Object) {
            std::cerr << "[EXL3] config.json parse failed: " << err << std::endl;
            return false;
        }

        // Architecture: config.json lists the HF architecture; map it to the
        // engine architecture id.
        {
            const JsonValue *archs = root.find("architectures");
            if (archs && archs->type == JsonValue::Type::Array && !archs->array.empty()) {
                detectedArch_ = archs->array[0].asString();
            }
            const JsonValue *mt = root.find("model_type");
            if (mt && mt->type == JsonValue::Type::String) {
                modelType_ = mt->asString();
            }
        }
        mapArchitecture();
        config_.architecture = detectedArch_;

        // Dimensions (Qwen/Gemma style keys; text_config.* takes precedence).
        hidden_ = cfgU32(root, "hidden_size", 0);
        nLayers_ = cfgU32(root, "num_hidden_layers", 0);
        nHeads_ = cfgU32(root, "num_attention_heads", 0);
        nKvHeads_ = cfgU32(root, "num_key_value_heads", nHeads_);
        headDim_ = cfgU32(root, "head_dim", 0);
        if (headDim_ == 0 && nHeads_ > 0) {
            headDim_ = hidden_ / nHeads_;
        }
        vocab_ = cfgU32(root, "vocab_size", 0);
        inter_ = cfgU32(root, "intermediate_size", 0);
        nExperts_ = cfgU32(root, "num_experts", 0);
        if (nExperts_ == 0) {
            nExperts_ = cfgU32(root, "num_local_experts", 0);
        }
        nExpertsUsed_ = cfgU32(root, "num_experts_per_tok", 0);
        if (nExpertsUsed_ == 0) {
            nExpertsUsed_ = cfgU32(root, "num_experts_per_token", 0);
        }
        expertInter_ = cfgU32(root, "moe_intermediate_size", 0);
        nSharedExperts_ = cfgU32(root, "n_shared_experts", 0);
        if (nSharedExperts_ == 0) {
            nSharedExperts_ = cfgU32(root, "num_shared_experts", 0);
        }
        sharedInter_ = cfgU32(root, "shared_expert_intermediate_size", 0);
        if (sharedInter_ == 0) {
            sharedInter_ = cfgU32(root, "shared_expert_inter_dim", 0);
        }
        convChannels_ = cfgU32(root, "linear_conv_kernel_dim", 0);
        ssmInner_ = cfgU32(root, "linear_value_head_dim", 0);
        if (ssmInner_ == 0) {
            ssmInner_ = cfgU32(root, "linear_inner_size", 0);
        }
        nVHeads_ = cfgU32(root, "linear_num_value_heads", 0);

        // ---- Fill the engine ModelConfig -----------------------------------
        config_.modelPath = directory_;
        config_.tokenizerPath = tokenizerPath_;
        config_.modelName = fs::path(directory_).filename().string();
        config_.hiddenSize = hidden_;
        config_.numLayers = nLayers_;
        config_.numAttentionHeads = nHeads_;
        config_.numKVHeads = nKvHeads_;
        config_.headDim = headDim_;
        config_.vocabSize = vocab_;
        config_.intermediateSize = inter_;
        config_.expertCount = nExperts_;
        config_.expertUsedCount = nExpertsUsed_;
        config_.expertFeedForwardLength = expertInter_;
        config_.expertSharedFeedForwardLength = sharedInter_;
        // Qwen3.5/3.8 full attention uses theta 1e7; older Qwen2 checkpoints use
        // 1e6.  The nested rope_parameters object is the authoritative source.
        const float defaultRopeTheta =
                (detectedArch_ == ARCH_QWEN35 || detectedArch_ == ARCH_QWEN35MOE)
                        ? 10000000.0f
                        : 1000000.0f;
        config_.ropeTheta = cfgF32Rope(root, "rope_theta", defaultRopeTheta);
        config_.finalLogitSoftcapping = cfgF32(root, "final_logit_softcapping", 0.0f);
        config_.maxSeqLen = cfgU32(root, "max_position_embeddings", 32768);
        normEps_ = cfgF32(root, "rms_norm_eps", 1e-6f);

        // qwen35/qwen35moe derived fields.
        if (detectedArch_ == ARCH_QWEN35 || detectedArch_ == ARCH_QWEN35MOE) {
            config_.fullAttentionInterval = cfgU32(root, "full_attention_interval",
                                                  config_.fullAttentionInterval);
            fullAttentionInterval_ = config_.fullAttentionInterval;
            if (convChannels_ > 0) {
                config_.ssmConvKernel = convChannels_;
            }
            // nKHeads * headK (both needed to reconstruct attn_qkv rows): the
            // GGUF field ssm.group_count holds nKHeads, ssm.state_size the headK.
            uint32_t nKHeads = cfgU32(root, "linear_num_key_heads", 0);
            uint32_t headK = cfgU32(root, "linear_key_head_dim", 0);
            config_.ssmGroupCount = nKHeads;
            config_.ssmStateSize = headK;
            config_.ssmTimeStepRank = nVHeads_;
            // The GGUF ssm.inner_size is the *value dimension* = nVHeads * headV
            // (linear_value_head_dim), NOT the config's linear_inner_size key.
            uint32_t headV = cfgU32(root, "linear_value_head_dim", 0);
            ssmInner_ = (nVHeads_ > 0 && headV > 0) ? nVHeads_ * headV : 0;
            config_.ssmInnerSize = ssmInner_;
            // The HF Qwen3.5 checkpoint stores the Multi-Token-Prediction head
            // as a separate top-level `mtp.*` module (model._keys_to_ignore_on_
            // load_unexpected == [r"^mtp.*", ...]); the base model runs without
            // it, so there is no extra `nextn` block in the layer stack.
            nextnPredictLayers_ = 0;
            config_.nextnPredictLayers = 0;

            // Full-attention head geometry: Qwen3.5 stores head_dim = 256 and
            // uses a *partial* rotary factor of 0.25 -> 64 rotated dims.  The
            // GDN's linear_key_head_dim (128) is a different, unrelated size;
            // deriving the attention head length from it broke the MRoPE
            // frequency schedule and the per-head Q/K norms.
            config_.attentionKeyLength = headDim_;
            config_.attentionValueLength = headDim_;
            const float partialRotary =
                    cfgF32Rope(root, "partial_rotary_factor", 1.0f);
            uint32_t rotaryDim = cfgU32(root, "rotary_dim", 0);
            if (rotaryDim == 0) {
                rotaryDim = static_cast<uint32_t>(
                        std::lround(static_cast<double>(headDim_) * partialRotary));
            }
            if (rotaryDim > headDim_) {
                rotaryDim = headDim_;
            }
            config_.ropeDimensionCount = rotaryDim;
            // MRoPE section split (text -> [t,h,w,e]); all-zero means "no
            // sections", which the qwen35 kernels treat as a single-position
            // schedule (identical to the GGUF defaults for text).
            if (const JsonValue *secs = cfgFindRope(root, "mrope_section")) {
                if (secs->type == JsonValue::Type::Array) {
                    for (size_t si = 0; si < 4 && si < secs->array.size(); ++si) {
                        config_.ropeDimensionSections[si] =
                                static_cast<uint32_t>(secs->array[si].asU64());
                    }
                }
            }
        }

        return true;
    }

    void EXL3Loader::mapArchitecture() {
        // A HF architecture string ("Qwen3ForCausalLM") or model_type ("qwen3")
        // is mapped onto one of the engine's supported architecture ids.
        std::string key = detectedArch_;
        std::transform(key.begin(), key.end(), key.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        auto has = [&](const char *needle) { return key.find(needle) != std::string::npos; };

        // qwen35moe before qwen35 (the substring is contained in both).
        if (has("qwen35moe") || has("qwen3.5moe") || has("qwen3_5moe") ||
            (has("qwen3") && has("moe"))) {
            detectedArch_ = ARCH_QWEN35MOE;
        } else if (has("qwen35") || has("qwen3.5") || has("qwen3_5") || has("ornith")) {
            detectedArch_ = ARCH_QWEN35;
        } else if (has("gemma4") || has("gemma-4") || has("gemma_4")) {
            detectedArch_ = ARCH_GEMMA4;
        } else if (has("qwen2")) {
            detectedArch_ = ARCH_QWEN2;
        } else {
            // Leave as-is; isSupportedArchitecture() in Model::load rejects it.
            std::cerr << "[EXL3] Unrecognized architecture \"" << detectedArch_
                      << "\" (model_type \"" << modelType_ << "\")" << std::endl;
        }
    }

    // =======================================================================
    // Loading
    // =======================================================================

    bool EXL3Loader::loadMetadata(const std::string &path) {
        std::string configPath, tokenizerPath;
        std::vector<std::string> shards;
        if (!discoverFiles(path, configPath, tokenizerPath, shards)) {
            return false;
        }
        tokenizerPath_ = tokenizerPath;
        if (configPath.empty()) {
            std::cerr << "[EXL3] No config.json beside " << path << std::endl;
            return false;
        }
        return parseConfig(configPath);
    }

    bool EXL3Loader::load(const std::string &path) {
        std::string configPath, tokenizerPath;
        std::vector<std::string> shards;
        if (!discoverFiles(path, configPath, tokenizerPath, shards)) {
            return false;
        }
        tokenizerPath_ = tokenizerPath;
        if (configPath.empty() || !parseConfig(configPath)) {
            std::cerr << "[EXL3] Failed to parse model config" << std::endl;
            return false;
        }

        st_ = std::make_unique<SafetensorsReader>();
        if (!st_->open(shards)) {
            std::cerr << "[EXL3] Failed to open safetensors shards" << std::endl;
            return false;
        }
        std::cout << "[EXL3] Opened " << shards.size() << " shard(s), "
                  << st_->tensors().size() << " tensors; architecture=" << detectedArch_
                  << " model_type=" << modelType_ << std::endl;

        // The reconstruction is OpenMP-parallel.  TinyCoder's ThreadPool pins
        // the MAIN thread to CPU 0 for inference, and OpenMP workers inherit
        // that one-CPU affinity mask -- so a load started after
        // ThreadPool::initialize() would otherwise run single-threaded (a 27B
        // EXL3 checkpoint then takes >20 minutes instead of a few minutes).
        // Widen the mask for the duration of the (host-side, one-off) load and
        // restore it before returning.
        struct AffinityScope {
#if defined(__linux__)
            cpu_set_t saved{};
            bool haveSaved = false;
            AffinityScope() {
                if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t),
                                           &saved) == 0) {
                    haveSaved = true;
                    cpu_set_t all;
                    CPU_ZERO(&all);
                    const long n = sysconf(_SC_NPROCESSORS_ONLN);
                    for (long i = 0; i < n; ++i) {
                        CPU_SET(static_cast<int>(i), &all);
                    }
                    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &all);
                }
            }
            ~AffinityScope() {
                if (haveSaved) {
                    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
                                           &saved);
                }
            }
#else
            AffinityScope() = default;
#endif
        } affinityScope;
        if (!emitAllTensors()) {
            std::cerr << "[EXL3] Failed to materialize tensors" << std::endl;
            return false;
        }

        loaded_ = true;
        std::cout << "[EXL3] Materialized " << records_.size() << " GGUF-named tensors"
                  << std::endl;
        return true;
    }

    // =======================================================================
    // Source resolution / linear decode
    // =======================================================================

    EXL3Loader::Src EXL3Loader::resolve(const std::vector<std::string> &candidates) const {
        for (const auto &base: candidates) {
            if (st_->find(base + ".trellis")) {
                Src s;
                s.base = base;
                s.exl3 = true;
                s.valid = true;
                hasTrellis_ = true;
                return s;
            }
            if (st_->find(base + ".weight")) {
                Src s;
                s.base = base;
                s.exl3 = false;
                s.valid = true;
                return s;
            }
        }
        return Src{};
    }

    bool EXL3Loader::decodeLinear(const Src &src, uint32_t out, uint32_t in, bool forceF32,
                                  std::vector<uint8_t> &outBytes, std::vector<float> *recon) {
        // Reads a sign vector (suh/svh, ±1) into `dst`, either from the stored
        // unpacked F16 form or by expanding the packed ±1 bitfield (su/sv).
        auto loadSignVector = [&](const std::string &unpackedSuffix, uint32_t count,
                                  std::vector<float> &dst) -> bool {
            if (const auto *t = st_->find(src.base + unpackedSuffix)) {
                const uint8_t *raw = st_->data(*t);
                if (raw == nullptr || SafetensorsReader::numel(*t) < count) {
                    return false;
                }
                dst.resize(count);
                if (t->dtype == SafetensorsReader::DType::F32) {
                    std::memcpy(dst.data(), raw, count * sizeof(float));
                } else if (t->dtype == SafetensorsReader::DType::F16 ||
                           t->dtype == SafetensorsReader::DType::BF16) {
                    const uint16_t *h = reinterpret_cast<const uint16_t *>(raw);
                    for (uint32_t i = 0; i < count; ++i) {
                        if (t->dtype == SafetensorsReader::DType::BF16) {
                            uint32_t bits = static_cast<uint32_t>(h[i]) << 16;
                            std::memcpy(&dst[i], &bits, sizeof(float));
                        } else {
                            dst[i] = EXL3Codec::halfToFloat(h[i]);
                        }
                    }
                } else {
                    return false;
                }
                return true;
            }
            const std::string packedSuffix =
                    unpackedSuffix == ".suh" ? ".su" : ".sv";
            if (const auto *t = st_->find(src.base + packedSuffix)) {
                const uint32_t groups = static_cast<uint32_t>(SafetensorsReader::numel(*t));
                dst.assign(static_cast<size_t>(groups) * 16, 1.0f);
                EXL3Codec::expandPackedSigns(
                        reinterpret_cast<const uint16_t *>(st_->data(*t)), groups, dst.data());
                if (dst.size() > count) {
                    dst.resize(count);
                }
                return true;
            }
            return false;
        };

        std::vector<float> w;
        if (src.exl3) {
            const auto *trellis = st_->find(src.base + ".trellis");
            if (out % 16 != 0 || in % 16 != 0) {
                std::cerr << "[EXL3] " << src.base
                          << ": trellis dims must be 16-divisible (" << out << "x" << in << ")"
                          << std::endl;
                return false;
            }
            const uint64_t tileWords =
                    trellis->shape.size() >= 3 ? trellis->shape.back() : 0;
            if (trellis->shape.size() != 3 || tileWords % 16 != 0) {
                std::cerr << "[EXL3] " << src.base << ": malformed trellis shape" << std::endl;
                return false;
            }
            const float Kf = static_cast<float>(tileWords) / 16.0f;
            const bool frac = (Kf != static_cast<float>(static_cast<int>(Kf)));
            const uint32_t KA = static_cast<uint32_t>(Kf);

            int codebook = EXL3Codec::CB_3INST;
            if (st_->find(src.base + ".mcg")) {
                codebook = EXL3Codec::CB_MCG;
            } else if (st_->find(src.base + ".mul1")) {
                codebook = EXL3Codec::CB_MUL1;
            }
            if (frac && codebook != EXL3Codec::CB_MUL1) {
                std::cerr << "[EXL3] " << src.base << ": half-integer K=" << Kf
                          << " requires the mul1 codebook" << std::endl;
                return false;
            }

            // Sign vectors: prefer the unpacked suh/svh, else expand the packed su/sv.
            // The reconstruct is [k=in_features, n=out_features]: suh is the
            // per-row (input) sign vector of length in, svh the per-column
            // (output) sign vector of length out. NOTE: the stored suh has `in`
            // elements and svh has `out`, i.e. the opposite of the engine's
            // [rows=out, cols=in] view here.
            std::vector<float> suh, svh;
            const bool haveSu = loadSignVector(".suh", in, suh);
            const bool haveSv = loadSignVector(".svh", out, svh);

            const uint16_t *packed = reinterpret_cast<const uint16_t *>(st_->data(*trellis));
            if (packed == nullptr) {
                std::cerr << "[EXL3] " << src.base << ": trellis data missing" << std::endl;
                return false;
            }
            // reconstructMatrix produces row-major [in][out] (in_features rows,
            // out_features cols, as the reference indexes the weight). The engine
            // wants row-major [out][in], so build [in][out] then transpose.
            std::vector<float> wRaw(static_cast<size_t>(out) * in);
            if (frac) {
                EXL3Codec::reconstructMatrixFrac(packed, KA, codebook,
                                                 haveSu ? suh.data() : nullptr,
                                                 haveSv ? svh.data() : nullptr,
                                                 in, out, wRaw.data());
            } else {
                EXL3Codec::reconstructMatrix(packed, KA, codebook,
                                             haveSu ? suh.data() : nullptr,
                                             haveSv ? svh.data() : nullptr,
                                             in, out, wRaw.data());
            }
            w.resize(static_cast<size_t>(out) * in);
#pragma omp parallel for schedule(static)
            for (uint32_t r = 0; r < out; ++r) {
                for (uint32_t c = 0; c < in; ++c) {
                    w[static_cast<size_t>(r) * in + c] =
                            wRaw[static_cast<size_t>(c) * out + r];
                }
            }
        } else {
            const auto *t = st_->find(src.base + ".weight");
            const uint64_t n = SafetensorsReader::numel(*t);
            if (n != static_cast<uint64_t>(out) * in) {
                std::cerr << "[EXL3] " << src.base << ".weight shape mismatch: got " << n
                          << " want " << (static_cast<uint64_t>(out) * in) << std::endl;
                return false;
            }
            const uint8_t *raw = st_->data(*t);
            w.resize(n);
            switch (t->dtype) {
                case SafetensorsReader::DType::F32:
                    std::memcpy(w.data(), raw, n * sizeof(float));
                    break;
                case SafetensorsReader::DType::F16: {
                    const uint16_t *src16 = reinterpret_cast<const uint16_t *>(raw);
                    for (uint64_t i = 0; i < n; ++i) {
                        w[i] = EXL3Codec::halfToFloat(src16[i]);
                    }
                    break;
                }
                case SafetensorsReader::DType::BF16: {
                    const uint16_t *src16 = reinterpret_cast<const uint16_t *>(raw);
                    for (uint64_t i = 0; i < n; ++i) {
                        uint32_t bits = static_cast<uint32_t>(src16[i]) << 16;
                        std::memcpy(&w[i], &bits, sizeof(float));
                    }
                    break;
                }
                default:
                    std::cerr << "[EXL3] " << src.base
                              << ".weight: unsupported dtype "
                              << SafetensorsReader::dtypeName(t->dtype) << std::endl;
                    return false;
            }
        }

        if (recon) {
            *recon = w;
        }
        if (forceF32) {
            writeF32(outBytes, w);
        } else {
            writeF16(outBytes, w);
        }
        return true;
    }

    // =======================================================================
    // Tensor registration
    // =======================================================================

    void EXL3Loader::addTensorShape(const std::string &name, uint32_t ggmlType,
                                    std::vector<uint32_t> shape,
                                    std::vector<uint8_t> bytes) {
        TensorInfo info;
        info.name = name;
        info.type = ggmlType;
        info.shape = std::move(shape);
        info.size = bytes.size();

        auto buf = std::make_unique<std::vector<uint8_t>>(std::move(bytes));
        const uint8_t *ptr = buf->data();
        buffers_.push_back(std::move(buf));

        info.offset = reinterpret_cast<uint64_t>(ptr);
        recIndex_[name] = records_.size();
        records_.push_back(std::move(info));
    }

    void EXL3Loader::addTensor(const std::string &name, uint32_t ggmlType, uint32_t rows,
                               uint32_t cols, std::vector<uint8_t> bytes) {
        // GGUF stores shapes reversed (numpy order): 2-D shape = {cols, rows}.
        addTensorShape(name, ggmlType, {cols, rows}, std::move(bytes));
    }

    // =======================================================================
    // Emit helpers
    // =======================================================================

    bool EXL3Loader::buildLinearF16(const Src &src, uint32_t expectedOut, uint32_t expectedIn,
                                    std::vector<uint8_t> &outBytes, bool forceF32) {
        return decodeLinear(src, expectedOut, expectedIn, forceF32, outBytes, nullptr);
    }

    bool EXL3Loader::loadSigns(const Src &src, uint32_t out, uint32_t in,
                               std::vector<float> &suh, std::vector<float> &svh) {
        auto load = [&](const std::string &suffix, uint32_t count,
                        std::vector<float> &dst) -> bool {
            if (const auto *t = st_->find(src.base + suffix)) {
                const uint8_t *raw = st_->data(*t);
                if (raw == nullptr || SafetensorsReader::numel(*t) < count) {
                    return false;
                }
                dst.resize(count);
                if (t->dtype == SafetensorsReader::DType::F32) {
                    std::memcpy(dst.data(), raw, count * sizeof(float));
                } else if (t->dtype == SafetensorsReader::DType::F16 ||
                           t->dtype == SafetensorsReader::DType::BF16) {
                    const uint16_t *h = reinterpret_cast<const uint16_t *>(raw);
                    for (uint32_t i = 0; i < count; ++i) {
                        if (t->dtype == SafetensorsReader::DType::BF16) {
                            uint32_t bits = static_cast<uint32_t>(h[i]) << 16;
                            std::memcpy(&dst[i], &bits, sizeof(float));
                        } else {
                            dst[i] = EXL3Codec::halfToFloat(h[i]);
                        }
                    }
                } else {
                    return false;
                }
                return true;
            }
            const std::string packed = (suffix == ".suh") ? ".su" : ".sv";
            if (const auto *t = st_->find(src.base + packed)) {
                const uint32_t groups =
                        static_cast<uint32_t>(SafetensorsReader::numel(*t));
                dst.assign(static_cast<size_t>(groups) * 16, 1.0f);
                EXL3Codec::expandPackedSigns(
                        reinterpret_cast<const uint16_t *>(st_->data(*t)), groups,
                        dst.data());
                if (dst.size() > count) dst.resize(count);
                return true;
            }
            return false;
        };
        // suh is input-side (length in), svh output-side (length out).
        load(".suh", in, suh);
        load(".svh", out, svh);
        return true;
    }

    bool EXL3Loader::recordNative(const std::string &ggmlName,
                                  const std::vector<std::string> &hfCandidates,
                                  uint32_t out, uint32_t in) {
        Src src = resolve(hfCandidates);
        if (!src.valid || !src.exl3) {
            return false;
        }
        const auto *trellis = st_->find(src.base + ".trellis");
        if (trellis == nullptr || trellis->shape.size() != 3 ||
            out % 16 != 0 || in % 16 != 0) {
            return false;
        }
        const uint32_t tileU16 = trellis->shape.back();
        if (tileU16 % 16 != 0) {
            return false;
        }
        const uint32_t K = tileU16 / 16;
        // Half-integer bitrates are not implemented by the fused GPU kernel.
        if (K == 0 || K > 8) {
            return false;
        }
        Exl3LinearInfo info;
        info.trellis = reinterpret_cast<const uint16_t *>(st_->data(*trellis));
        if (info.trellis == nullptr) {
            return false;
        }
        info.trellisBytes = static_cast<uint64_t>(in / 16) * (out / 16) * tileU16 *
                            sizeof(uint16_t);
        info.k = in;
        info.n = out;
        info.K = K;
        info.tileU16 = tileU16;
        info.frac = false;
        info.codebook = EXL3Codec::CB_3INST;
        if (st_->find(src.base + ".mcg")) {
            info.codebook = EXL3Codec::CB_MCG;
        } else if (st_->find(src.base + ".mul1")) {
            info.codebook = EXL3Codec::CB_MUL1;
        }
        std::vector<float> suh, svh;
        loadSigns(src, out, in, suh, svh);
        auto suhOwner = std::make_unique<std::vector<float>>(std::move(suh));
        auto svhOwner = std::make_unique<std::vector<float>>(std::move(svh));
        info.suh = suhOwner->empty() ? nullptr : suhOwner->data();
        info.svh = svhOwner->empty() ? nullptr : svhOwner->data();
        nativeSigns_.push_back(std::move(suhOwner));
        nativeSigns_.push_back(std::move(svhOwner));
        native_[ggmlName] = info;
        return true;
    }

    bool EXL3Loader::emitLinear(const std::string &ggmlName,
                                const std::vector<std::string> &hfCandidates,
                                uint32_t expectedOut, uint32_t expectedIn, bool forceF32) {
        Src src = resolve(hfCandidates);
        if (!src.valid) {
            return false;
        }
        std::vector<uint8_t> bytes;
        if (!decodeLinear(src, expectedOut, expectedIn, forceF32, bytes, nullptr)) {
            return false;
        }
        addTensor(ggmlName, forceF32 ? GGML_TYPE_F32 : GGML_TYPE_F16, expectedOut, expectedIn,
                  std::move(bytes));
        if (!forceF32) {
            recordNative(ggmlName, hfCandidates, expectedOut, expectedIn);
        }
        return true;
    }

    bool EXL3Loader::emit1D(const std::string &ggmlName,
                            const std::vector<std::string> &hfCandidates, bool bumpNorm) {
        for (const std::string &base: hfCandidates) {
            // Accept either the exact tensor name or a bare base name (append
            // ".weight"), so callers may pass e.g. "linear_attn.norm".
            const auto *t = st_->find(base);
            if (!t) {
                t = st_->find(base + ".weight");
            }
            if (!t) {
                continue;
            }
            const uint64_t n = SafetensorsReader::numel(*t);
            const uint8_t *raw = st_->data(*t);
            std::vector<float> f(n);
            switch (t->dtype) {
                case SafetensorsReader::DType::F32:
                    std::memcpy(f.data(), raw, n * sizeof(float));
                    break;
                case SafetensorsReader::DType::F16: {
                    const uint16_t *s = reinterpret_cast<const uint16_t *>(raw);
                    for (uint64_t i = 0; i < n; ++i) {
                        f[i] = EXL3Codec::halfToFloat(s[i]);
                    }
                    break;
                }
                case SafetensorsReader::DType::BF16: {
                    const uint16_t *s = reinterpret_cast<const uint16_t *>(raw);
                    for (uint64_t i = 0; i < n; ++i) {
                        uint32_t bits = static_cast<uint32_t>(s[i]) << 16;
                        std::memcpy(&f[i], &bits, sizeof(float));
                    }
                    break;
                }
                default:
                    continue;
            }
            if (bumpNorm) {
                // HF Qwen3.5 RMSNorm is zero-centered: forward() multiplies by
                // (1 + weight). The GGUF/engine RMSNorm kernels multiply by the
                // stored weight directly, so fold the +1 in here.
                for (uint64_t i = 0; i < n; ++i) {
                    f[i] += 1.0f;
                }
            }
            std::vector<uint8_t> bytes;
            writeF32(bytes, f);
            addTensorShape(ggmlName, GGML_TYPE_F32,
                           {static_cast<uint32_t>(n)}, std::move(bytes));
            return true;
        }
        return false;
    }

    bool EXL3Loader::emitExpertStack(const std::string &ggmlName,
                                     const std::string &hfBasePrefix,
                                     const std::string &proj, uint32_t count,
                                     uint32_t expectedOut, uint32_t expectedIn) {
        std::vector<uint8_t> stacked(static_cast<size_t>(count) * expectedOut * expectedIn *
                                     sizeof(uint16_t));
        const size_t perExpert = static_cast<size_t>(expectedOut) * expectedIn * sizeof(uint16_t);
        for (uint32_t e = 0; e < count; ++e) {
            std::string base = hfBasePrefix + std::to_string(e) + "." + proj;
            Src src = resolve({base});
            if (!src.valid) {
                std::cerr << "[EXL3] Missing expert tensor " << base << std::endl;
                return false;
            }
            std::vector<uint8_t> bytes;
            if (!decodeLinear(src, expectedOut, expectedIn, false, bytes, nullptr)) {
                return false;
            }
            std::memcpy(stacked.data() + e * perExpert, bytes.data(), perExpert);
        }
        // GGUF 3-D expert tensor shape is reversed → {cols, perExpertRows, count}.
        addTensorShape(ggmlName, GGML_TYPE_F16, {expectedIn, expectedOut, count},
                       std::move(stacked));
        return true;
    }

    bool EXL3Loader::emitFusedGateUpExperts(const std::string &ggmlName,
                                            const std::string &hfBasePrefix, uint32_t count,
                                            uint32_t perExpertOut, uint32_t expectedIn) {
        const uint32_t fusedRows = perExpertOut * 2;
        const size_t perExpertBytes =
                static_cast<size_t>(fusedRows) * expectedIn * sizeof(uint16_t);
        std::vector<uint8_t> stacked(static_cast<size_t>(count) * perExpertBytes);
        // F32 reconstruction scratch, reused across experts.
        std::vector<uint8_t> gateBytes, upBytes;
        for (uint32_t e = 0; e < count; ++e) {
            std::string gateBase = hfBasePrefix + std::to_string(e) + ".gate_proj";
            std::string upBase = hfBasePrefix + std::to_string(e) + ".up_proj";
            Src gateSrc = resolve({gateBase});
            Src upSrc = resolve({upBase});
            if (!gateSrc.valid || !upSrc.valid) {
                std::cerr << "[EXL3] Missing fused expert gate/up for expert " << e << std::endl;
                return false;
            }
            if (!decodeLinear(gateSrc, perExpertOut, expectedIn, false, gateBytes, nullptr)) {
                return false;
            }
            if (!decodeLinear(upSrc, perExpertOut, expectedIn, false, upBytes, nullptr)) {
                return false;
            }
            uint8_t *dst = stacked.data() + static_cast<size_t>(e) * perExpertBytes;
            std::memcpy(dst, gateBytes.data(), gateBytes.size());
            std::memcpy(dst + static_cast<size_t>(perExpertOut) * expectedIn * sizeof(uint16_t),
                        upBytes.data(), upBytes.size());
        }
        addTensorShape(ggmlName, GGML_TYPE_F16, {expectedIn, fusedRows, count},
                       std::move(stacked));
        return true;
    }

    bool EXL3Loader::emitHeadStack(const std::string &ggmlName,
                                   const std::string &hfBasePrefix, uint32_t count,
                                   uint32_t expectedOut, uint32_t expectedIn) {
        std::vector<uint8_t> stacked(static_cast<size_t>(count) * expectedOut * expectedIn *
                                     sizeof(uint16_t));
        const size_t perHead = static_cast<size_t>(expectedOut) * expectedIn * sizeof(uint16_t);
        for (uint32_t h = 0; h < count; ++h) {
            std::string base = hfBasePrefix + std::to_string(h);
            Src src = resolve({base});
            if (!src.valid) {
                std::cerr << "[EXL3] Missing head tensor " << base << std::endl;
                return false;
            }
            std::vector<uint8_t> bytes;
            if (!decodeLinear(src, expectedOut, expectedIn, false, bytes, nullptr)) {
                return false;
            }
            std::memcpy(stacked.data() + h * perHead, bytes.data(), perHead);
        }
        addTensorShape(ggmlName, GGML_TYPE_F16, {expectedIn, expectedOut, count},
                       std::move(stacked));
        return true;
    }

    bool EXL3Loader::emitEmbedding(const std::string &ggmlName,
                                   const std::vector<std::string> &hfCandidates,
                                   uint32_t rows, uint32_t cols, bool /*latentUntie*/) {
        for (const auto &cand: hfCandidates) {
            const auto *t = st_->find(cand);
            if (!t) {
                continue;
            }
            const uint64_t n = SafetensorsReader::numel(*t);
            if (n != static_cast<uint64_t>(rows) * cols) {
                std::cerr << "[EXL3] embedding " << cand << " shape mismatch: " << n << " vs "
                          << (static_cast<uint64_t>(rows) * cols) << std::endl;
                return false;
            }
            const uint8_t *raw = st_->data(*t);
            std::vector<uint8_t> bytes;
            if (t->dtype == SafetensorsReader::DType::F16) {
                bytes.assign(raw, raw + n * sizeof(uint16_t));
            } else if (t->dtype == SafetensorsReader::DType::F32) {
                std::vector<float> f(n);
                std::memcpy(f.data(), raw, n * sizeof(float));
                writeF16(bytes, f);
            } else if (t->dtype == SafetensorsReader::DType::BF16) {
                std::vector<float> f(n);
                const uint16_t *s = reinterpret_cast<const uint16_t *>(raw);
                for (uint64_t i = 0; i < n; ++i) {
                    uint32_t bits = static_cast<uint32_t>(s[i]) << 16;
                    std::memcpy(&f[i], &bits, sizeof(float));
                }
                writeF16(bytes, f);
            } else {
                std::cerr << "[EXL3] embedding " << cand << ": unsupported dtype" << std::endl;
                return false;
            }
            addTensor(ggmlName, GGML_TYPE_F16, rows, cols, std::move(bytes));
            return true;
        }
        return false;
    }

    // =======================================================================
    // Architecture-specific tensor emission
    // =======================================================================

    bool EXL3Loader::emitAllTensors() {
        const uint32_t V = vocab_;
        const uint32_t H = hidden_;

        // ---- Token embeddings (always F16 for the embedding path) -----------
        if (!emitEmbedding("token_embd.weight",
                           {"model.language_model.embed_tokens",
                            "model.language_model.embed_tokens.weight", "model.embed_tokens",
                            "model.embed_tokens.weight", "transformer.wte"},
                           V, H, false)) {
            std::cerr << "[EXL3] Failed to load token embeddings" << std::endl;
            return false;
        }

        // ---- LM head -------------------------------------------------------
        // The EXL3 checkpoint stores the head as an EXL3 linear (lm_head.trellis
        // + lm_head.suh/svh/mul1); older exports use lm_head.weight.
        {
            if (resolve({"lm_head"}).valid) {
                if (!emitLinear("output.weight", {"lm_head"}, V, H, false)) {
                    return false;
                }
            } else {
                // Tied: alias output.weight to the token embedding buffer.
                auto it = recIndex_.find("token_embd.weight");
                TensorInfo info = records_[it->second];
                info.name = "output.weight";
                recIndex_["output.weight"] = records_.size();
                records_.push_back(std::move(info));
            }
        }

        // ---- Final norm ----------------------------------------------------
        // Qwen3.5/HF `model.language_model.norm` is a zero-centered RMSNorm
        // (multiply by 1 + weight); Gemma uses the stored weight directly.
        {
            const std::string hfNorm = detectedArch_ == ARCH_GEMMA4 ? "model.norm" : "model.language_model.norm";
            if (!emit1D("output_norm.weight",
                        {hfNorm, hfNorm + ".weight", "model.norm", "model.norm.weight"},
                        detectedArch_ != ARCH_GEMMA4)) {
                std::cerr << "[EXL3] Missing final norm" << std::endl;
                return false;
            }
        }

        // ---- Per-layer -----------------------------------------------------
        for (uint32_t i = 0; i < nLayers_; ++i) {
            const std::string prefix = "blk." + std::to_string(i) + ".";
            // HF Qwen3.5 checkpoints nest the text stack under
            // `model.language_model.layers.`; Gemma/Qwen2 use `model.layers.`.
            const std::string hf = detectedArch_ == ARCH_GEMMA4 || detectedArch_ == ARCH_QWEN2
                                           ? "model.layers." + std::to_string(i) + "."
                                           : "model.language_model.layers." + std::to_string(i) + ".";

            // Norms (shared by every architecture).
            const bool bump = detectedArch_ != ARCH_GEMMA4 && detectedArch_ != ARCH_QWEN2;
            emit1D(prefix + "attn_norm.weight",
                   {hf + "input_layernorm", hf + "input_layernorm.weight"}, bump);

            if (detectedArch_ == ARCH_QWEN2) {
                emitLinear(prefix + "attn_q.weight", {hf + "self_attn.q_proj"}, nHeads_ * headDim_, H);
                emitLinear(prefix + "attn_k.weight", {hf + "self_attn.k_proj"}, nKvHeads_ * headDim_, H);
                emitLinear(prefix + "attn_v.weight", {hf + "self_attn.v_proj"}, nKvHeads_ * headDim_, H);
                emitLinear(prefix + "attn_output.weight", {hf + "self_attn.o_proj"}, H, nHeads_ * headDim_);
                emit1D(prefix + "attn_q.bias", {hf + "self_attn.q_proj.bias"});
                emit1D(prefix + "attn_k.bias", {hf + "self_attn.k_proj.bias"});
                emit1D(prefix + "attn_v.bias", {hf + "self_attn.v_proj.bias"});
                emit1D(prefix + "ffn_norm.weight", {hf + "post_attention_layernorm"});
                emitLinear(prefix + "ffn_gate.weight", {hf + "mlp.gate_proj"}, inter_, H);
                emitLinear(prefix + "ffn_up.weight", {hf + "mlp.up_proj"}, inter_, H);
                emitLinear(prefix + "ffn_down.weight", {hf + "mlp.down_proj"}, H, inter_);
            } else if (detectedArch_ == ARCH_QWEN35) {
                const bool recurrent = detail::isQwen35RecurrentLayer(config_, i);
                emitLinear(prefix + "ffn_gate.weight", {hf + "mlp.gate_proj"}, inter_, H);
                emitLinear(prefix + "ffn_up.weight", {hf + "mlp.up_proj"}, inter_, H);
                emitLinear(prefix + "ffn_down.weight", {hf + "mlp.down_proj"}, H, inter_);
                emit1D(prefix + "post_attention_norm.weight",
                       {hf + "post_attention_layernorm"}, bump);

                if (recurrent) {
                    if (!emitGdnLayer(prefix, hf)) {
                        std::cerr << "[EXL3] Missing recurrent tensors for layer " << i
                                  << std::endl;
                        return false;
                    }
                } else {
                    // Qwen3.5 self-attention uses a fused Q+gate projection:
                    // q_proj outputs 2 * nHeads * headDim (Q block, then gate).
                    emitLinear(prefix + "attn_q.weight", {hf + "self_attn.q_proj"},
                               nHeads_ * 2 * headDim_, H);
                    emitLinear(prefix + "attn_k.weight", {hf + "self_attn.k_proj"},
                               nKvHeads_ * headDim_, H);
                    emitLinear(prefix + "attn_v.weight", {hf + "self_attn.v_proj"},
                               nKvHeads_ * headDim_, H);
                    emitLinear(prefix + "attn_output.weight", {hf + "self_attn.o_proj"}, H,
                               nHeads_ * headDim_);
                    emit1D(prefix + "attn_q_norm.weight", {hf + "self_attn.q_norm"}, bump);
                    emit1D(prefix + "attn_k_norm.weight", {hf + "self_attn.k_norm"}, bump);
                }
            } else if (detectedArch_ == ARCH_QWEN35MOE) {
                const bool recurrent = detail::isQwen35RecurrentLayer(config_, i);
                emit1D(prefix + "post_attention_norm.weight",
                       {hf + "post_attention_layernorm"}, bump);

                // Routed experts.
                emitLinear(prefix + "ffn_gate_inp.weight", {hf + "mlp.gate"}, nExperts_, H);
                if (!emitFusedGateUpExperts(prefix + "ffn_gate_up_exps.weight",
                                            hf + "mlp.experts.", nExperts_, expertInter_, H)) {
                    // Fall back to separate gate/up stacks when the fused form is absent.
                    if (!emitExpertStack(prefix + "ffn_gate_exps.weight",
                                         hf + "mlp.experts.", "gate_proj", nExperts_,
                                         expertInter_, H) ||
                        !emitExpertStack(prefix + "ffn_up_exps.weight", hf + "mlp.experts.",
                                         "up_proj", nExperts_, expertInter_, H)) {
                        return false;
                    }
                }
                emitExpertStack(prefix + "ffn_down_exps.weight", hf + "mlp.experts.",
                                "down_proj", nExperts_, H, expertInter_);

                // Shared expert + (optional) sigmoid router.
                if (nSharedExperts_ > 0 && sharedInter_ > 0) {
                    emitLinear(prefix + "ffn_gate_inp_shexp.weight", {hf + "mlp.shared_expert_gate"},
                               nSharedExperts_, H);
                    emitLinear(prefix + "ffn_gate_shexp.weight", {hf + "mlp.shared_expert.gate_proj"},
                               sharedInter_, H);
                    emitLinear(prefix + "ffn_up_shexp.weight", {hf + "mlp.shared_expert.up_proj"},
                               sharedInter_, H);
                    emitLinear(prefix + "ffn_down_shexp.weight", {hf + "mlp.shared_expert.down_proj"},
                               H, sharedInter_);
                }

                if (recurrent) {
                    if (!emitGdnLayer(prefix, hf)) {
                        std::cerr << "[EXL3] Missing recurrent tensors for layer " << i
                                  << std::endl;
                        return false;
                    }
                    // MTP (nextn) block. HF Qwen3.5 checkpoints keep the MTP head
                    // in a separate top-level `mtp.*` module the base model
                    // ignores (nextnPredictLayers_ == 0), so nothing to emit.
                    if (nextnPredictLayers_ > 0) {
                        emitLinear(prefix + "nextn.eh_proj.weight", {hf + "mtp.eh_proj"}, H, 2 * H);
                        emit1D(prefix + "nextn.enorm.weight", {hf + "mtp.enorm"}, false);
                        emit1D(prefix + "nextn.hnorm.weight", {hf + "mtp.hnorm"}, false);
                        emit1D(prefix + "nextn.shared_head_norm.weight", {hf + "mtp.norm"}, bump);
                    }
                } else {
                    // Qwen3.5 self-attention uses a fused Q+gate projection.
                    emitLinear(prefix + "attn_q.weight", {hf + "self_attn.q_proj"},
                               nHeads_ * 2 * headDim_, H);
                    emitLinear(prefix + "attn_k.weight", {hf + "self_attn.k_proj"},
                               nKvHeads_ * headDim_, H);
                    emitLinear(prefix + "attn_v.weight", {hf + "self_attn.v_proj"},
                               nKvHeads_ * headDim_, H);
                    emitLinear(prefix + "attn_output.weight", {hf + "self_attn.o_proj"}, H,
                               nHeads_ * headDim_);
                    emit1D(prefix + "attn_q_norm.weight", {hf + "self_attn.q_norm"}, bump);
                    emit1D(prefix + "attn_k_norm.weight", {hf + "self_attn.k_norm"}, bump);
                }
            } else if (detectedArch_ == ARCH_GEMMA4) {
                emitLinear(prefix + "attn_q.weight", {hf + "self_attn.q_proj"}, nHeads_ * headDim_, H);
                emitLinear(prefix + "attn_k.weight", {hf + "self_attn.k_proj"}, nKvHeads_ * headDim_, H);
                emitLinear(prefix + "attn_v.weight", {hf + "self_attn.v_proj"}, nKvHeads_ * headDim_, H);
                emitLinear(prefix + "attn_output.weight", {hf + "self_attn.o_proj"}, H, nHeads_ * headDim_);
                emit1D(prefix + "attn_q_norm.weight", {hf + "self_attn.q_norm"});
                emit1D(prefix + "attn_k_norm.weight", {hf + "self_attn.k_norm"});
                emit1D(prefix + "post_attention_norm.weight", {hf + "post_attention_layernorm"});
                emit1D(prefix + "ffn_norm.weight", {hf + "pre_feedforward_layernorm"});
                emit1D(prefix + "post_ffw_norm.weight", {hf + "post_feedforward_layernorm"});
                emit1D(prefix + "layer_output_scale.weight", {hf + "layer_output_scale"});
                if (nExperts_ > 0) {
                    emitLinear(prefix + "ffn_gate_inp.weight", {hf + "mlp.gate"}, nExperts_, H);
                    if (!emitFusedGateUpExperts(prefix + "ffn_gate_up_exps.weight",
                                                hf + "mlp.experts.", nExperts_, expertInter_, H)) {
                        emitExpertStack(prefix + "ffn_gate_exps.weight", hf + "mlp.experts.",
                                        "gate_proj", nExperts_, expertInter_, H);
                        emitExpertStack(prefix + "ffn_up_exps.weight", hf + "mlp.experts.",
                                        "up_proj", nExperts_, expertInter_, H);
                    }
                    emitExpertStack(prefix + "ffn_down_exps.weight", hf + "mlp.experts.",
                                    "down_proj", nExperts_, H, expertInter_);
                    emit1D(prefix + "pre_ffw_norm_2.weight", {hf + "pre_feedforward_layernorm_2"});
                    emit1D(prefix + "post_ffw_norm_1.weight", {hf + "post_feedforward_layernorm_1"});
                    emit1D(prefix + "post_ffw_norm_2.weight", {hf + "post_feedforward_layernorm_2"});
                } else {
                    emitLinear(prefix + "ffn_gate.weight", {hf + "mlp.gate_proj"}, inter_, H);
                    emitLinear(prefix + "ffn_up.weight", {hf + "mlp.up_proj"}, inter_, H);
                    emitLinear(prefix + "ffn_down.weight", {hf + "mlp.down_proj"}, H, inter_);
                }
            } else {
                std::cerr << "[EXL3] Unknown architecture \"" << detectedArch_ << "\"" << std::endl;
                return false;
            }
        }

        return true;
    }

    bool EXL3Loader::emitGdnLayer(const std::string &prefix, const std::string &hf) {
        // Qwen3.5/3.8 gated-delta-net (linear attention) tensors.
        //
        // HF source (Qwen3_5GatedDeltaNet):
        //   key_dim   = linear_num_key_heads   * linear_key_head_dim     (16*128 = 2048)
        //   value_dim = linear_num_value_heads * linear_value_head_dim   (48*128 = 6144)
        //   conv_dim  = 2*key_dim + value_dim                            (10240)
        //   in_proj_qkv: hidden -> conv_dim ; in_proj_z: hidden -> value_dim
        //   in_proj_b:   hidden -> num_v_heads (beta = sigmoid(b))
        //   in_proj_a:   hidden -> num_v_heads (a, used for the decay gate)
        //   beta = sigmoid(b); g = -A_log.exp() * softplus(a + dt_bias)
        const uint32_t H = hidden_;
        const uint32_t nKHeads = config_.ssmGroupCount; // linear_num_key_heads
        const uint32_t headK = config_.ssmStateSize;    // linear_key_head_dim
        const uint32_t nVHeads = config_.ssmTimeStepRank;// linear_num_value_heads
        const uint32_t headV = nVHeads > 0 ? ssmInner_ / nVHeads : 0;
        const uint32_t keyDim = nKHeads * headK;
        const uint32_t valueDim = nVHeads * headV;
        const uint32_t convDim = 2 * keyDim + valueDim;

        emitLinear(prefix + "attn_qkv.weight",
                   {hf + "linear_attn.in_proj_qkv", hf + "linear_attn.qkv_proj"}, convDim, H);
        emitLinear(prefix + "attn_gate.weight",
                   {hf + "linear_attn.in_proj_z", hf + "linear_attn.gate_proj"}, valueDim, H);
        // ssm_out is the linear_attn output projection (value-dim → hidden).
        emitLinear(prefix + "ssm_out.weight", {hf + "linear_attn.out_proj"}, H, valueDim);
        // Depthwise conv over convDim channels; the raw HF weight is [convDim, 1, K].
        // The engine reads it as F32 with shape {width(K), channels}: data as
        // [channel * K + tap]. decodeLinear yields [out=K][in=convDim] row-major
        // which is exactly that layout, so no extra transpose is needed.
        {
            std::vector<uint8_t> bytes;
            Src src = resolve({hf + "linear_attn.conv1d"});
            if (src.valid && decodeLinear(src, convChannels_, convDim, true, bytes, nullptr)) {
                addTensorShape(prefix + "ssm_conv1d.weight", GGML_TYPE_F32,
                               {convChannels_, convDim}, std::move(bytes));
            } else {
                std::cerr << "[EXL3] Missing linear_attn.conv1d for " << hf << std::endl;
                return false;
            }
        }
        // Gated RMSNorm over the value head (standard RMSNorm, weight used as-is).
        emit1D(prefix + "ssm_norm.weight", {hf + "linear_attn.norm"}, false);
        emit1D(prefix + "ssm_dt.bias", {hf + "linear_attn.dt_bias"}, false);
        // ssm_a holds the per-head log-decay scale. The engine computes
        // decay = exp(softplus(a + dt_bias) * ssm_a) and expects ssm_a < 0;
        // HF stores A_log and uses g = -exp(A_log) * softplus(...), so ssm_a
        // must be -exp(A_log) (a small negative number).
        if (const auto *t = st_->find(hf + "linear_attn.A_log")) {
            // BF16 A_log -> -exp(A_log).
            const uint64_t n = SafetensorsReader::numel(*t);
            const uint8_t *raw = st_->data(*t);
            std::vector<float> v(n);
            const uint16_t *s = reinterpret_cast<const uint16_t *>(raw);
            for (uint64_t i = 0; i < n; ++i) {
                uint32_t bits = static_cast<uint32_t>(s[i]) << 16;
                float x;
                std::memcpy(&x, &bits, sizeof(float));
                v[i] = -std::exp(x);
            }
            std::vector<uint8_t> bytes;
            writeF32(bytes, v);
            addTensorShape(prefix + "ssm_a", GGML_TYPE_F32,
                           {static_cast<uint32_t>(n)}, std::move(bytes));
        } else if (!emit1D(prefix + "ssm_a", {hf + "linear_attn.ssm_a"}, false)) {
            std::cerr << "[EXL3] Missing linear_attn.A_log/ssm_a for " << hf << std::endl;
            return false;
        }
        // ssm_alpha (a) and ssm_beta (b) are low-rank hidden -> nVHeads maps; the
        // engine reads them as F32 applied to the raw hidden state.
        // HF: beta = sigmoid(in_proj_b @ x); a = in_proj_a @ x.
        emitLinear(prefix + "ssm_alpha.weight",
                   {hf + "linear_attn.in_proj_a", hf + "linear_attn.alpha_proj"}, nVHeads, H, true);
        emitLinear(prefix + "ssm_beta.weight",
                   {hf + "linear_attn.in_proj_b", hf + "linear_attn.beta_proj"}, nVHeads, H, true);
        (void) normEps_;
        return true;
    }

}// namespace tinycoder
