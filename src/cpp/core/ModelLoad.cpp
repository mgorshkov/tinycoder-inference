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

#include "MemHints.hpp"
#include "Model.hpp"
#include "gpu/InklingGPU.hpp"
#include "ThreadPool.hpp"

#include "EXL3Loader.hpp"
#include "GGMLDequantize.hpp"
#include "GGMLQuantize.hpp"
#include "GGUFLoader.hpp"
#include "ModelInternal.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/sysinfo.h>
#include <vector>

namespace tinycoder {

    // ---- Model loading ----

    namespace {
        // ------------------------------------------------------------------
        // PTQ1_0 (Prism ternary, group 128, 28 B/block) -> PQ2_0 (34 B/block).
        //
        // The two packs carry the SAME ternary values and the same fp16 group
        // scale; they differ only in bit packing, so this is lossless and lets the
        // whole PQ2_0 pipeline (ternary GEMV, Hadamard fold on activations, fp16
        // twin prefill, latent embedding) serve both files.  Block layouts, taken
        // from the reference implementation (PrismML-Eng/llama.cpp,
        // ggml-quants.c:dequantize_row_ptq1_0):
        //
        //   PTQ1_0: qs[24] (5 trits/byte) + qh[2] (4 trits/byte) + d(fp16) = 28 B
        //           element  16*n + m      = trit n of qs[m],      m <  16, n < 5
        //           element  80 + 8*n + m  = trit n of qs[16+m],   m <   8, n < 5
        //           element 120 + 2*n + h  = trit n of qh[h],      h <   2, n < 4
        //           trit n = ((uint16_t)(qs*pow3[n] as uint8) * 3) >> 8
        //   PQ2_0:  d(fp16) + qs[32]; element j at byte j/4, bits 2*(j%4); code q
        //           in 0..2 and value = (q - 1) * d in both packs.
        //
        // Verified by decoding a real PTQ1_0 block and comparing the recovered
        // symbols against the reference order above; the end-to-end check is that
        // the transcoded file reproduces the native PQ2_0 file's logits.
        // ------------------------------------------------------------------
        void transcodePTQ1_0ToPQ2_0(const uint8_t *src, uint8_t *dst, size_t blocks) {
            static const uint8_t pow3[5] = {1, 3, 9, 27, 81};
            for (size_t b = 0; b < blocks; ++b) {
                const uint8_t *s = src + b * 28u;
                uint8_t *o = dst + b * 34u;
                o[0] = s[26];// d (fp16), same value in both packs
                o[1] = s[27];
                std::memset(o + 2, 0, 32);
                uint8_t code[128];
                uint32_t idx = 0;
                for (uint32_t n = 0; n < 5; ++n) {
                    for (uint32_t m = 0; m < 16; ++m) {
                        const uint8_t q = static_cast<uint8_t>(s[m] * pow3[n]);
                        code[idx++] =
                                static_cast<uint8_t>((static_cast<uint16_t>(q) * 3u) >> 8);
                    }
                }
                for (uint32_t n = 0; n < 5; ++n) {
                    for (uint32_t m = 0; m < 8; ++m) {
                        const uint8_t q =
                                static_cast<uint8_t>(s[16u + m] * pow3[n]);
                        code[idx++] =
                                static_cast<uint8_t>((static_cast<uint16_t>(q) * 3u) >> 8);
                    }
                }
                for (uint32_t n = 0; n < 4; ++n) {
                    for (uint32_t h = 0; h < 2; ++h) {
                        const uint8_t q =
                                static_cast<uint8_t>(s[24u + h] * pow3[n]);
                        code[idx++] =
                                static_cast<uint8_t>((static_cast<uint16_t>(q) * 3u) >> 8);
                    }
                }
                for (uint32_t j = 0; j < 128; ++j) {
                    o[2u + (j >> 2)] |=
                            static_cast<uint8_t>((code[j] & 3u) << (2u * (j & 3u)));
                }
            }
        }

        bool ptq1NativeEnabled() {
            static const bool on = [] {
                const char *e = std::getenv("TINYCODER_PT_NATIVE");
                return e != nullptr && e[0] == '1';
            }();
            return on;
        }
    }// namespace

    namespace {
        void requantIQ2XSGateUpToQ4K(std::vector<Model::LayerWeights> &layers) {
            const char *e = std::getenv("TINYCODER_IQ2_GATEUP_Q4K");
            // Opt-in only, for the same reason as the IQ3/Q6 rewrites below:
            // it re-quantises lossless IQ2_S weights through quantizeQ4KBlock.
            if (e == nullptr || e[0] == '0') return;
            uint32_t converted = 0;
            std::vector<float> x(256);
            for (auto &l: layers) {
                for (QuantizedMatrix *m: {&l.ffnGate, &l.ffnUp}) {
                    if (m->type != 17 || m->cols == 0 || (m->cols & 255u) != 0 ||
                        m->data.empty())
                        continue;
                    const uint32_t blocks = m->cols / 256u;
                    const uint32_t srcBytes = ggmlTypeSize(17);
                    AlignedVector<uint8_t> out;
                    out.resize(static_cast<size_t>(m->rows) * blocks * 144u);
                    for (uint32_t r = 0; r < m->rows; ++r) {
                        const uint8_t *src = m->data.data() +
                                             static_cast<size_t>(r) * blocks * srcBytes;
                        uint8_t *dst = out.data() +
                                       static_cast<size_t>(r) * blocks * 144u;
                        for (uint32_t b = 0; b < blocks; ++b) {
                            GGMLDequantize::dequantizeIQ2_XSBlock(
                                    src + b * srcBytes, x.data());
                            GGMLQuantize::quantizeQ4KBlock(x.data(), dst + b * 144u);
                        }
                    }
                    m->data = std::move(out);
                    m->type = GGML_TYPE_Q4_K;
                    m->prepackedData.clear();
                    ++converted;
                }
            }
            if (converted != 0) {
                std::cout << "[TinyCoder] IQ2_XS gate/up requantized to Q4_K ("
                          << converted << " matrices)" << std::endl;
            }
        }
        // Requantize IQ3_XXS / IQ3_S projections to Q4_K for larger dense
        // models (Qwen2.5-7B): the Q4_K int8 dot is much cheaper per weight than
        // the IQ3 grid-lookup dot, and the 4.5 bpw form still fits VRAM.
        void requantIQ3ToQ4K(std::vector<Model::LayerWeights> &layers,
                             uint32_t hiddenSize) {
            const char *e = std::getenv("TINYCODER_IQ3_Q4K");
            // Opt-in only: these rewrites replace a lossless K-quant with
            // a re-quantised Q4_K.  That is a numerics change, not a
            // formatting one, and it silently corrupted every checkpoint
            // with hidden >= 2048 (7B incoherence, 9B, 35B-A3B).  Kept as
            // an explicit experiment until the quantiser passes a
            // round-trip test.
            if (e == nullptr || e[0] == '0') return;
            if (hiddenSize < 2048u) return;// keep the 1.5B IQ3 fast path
            uint32_t converted = 0;
            std::vector<float> x(256);
            for (auto &l: layers) {
                QuantizedMatrix *mats[] = {&l.attnQ, &l.attnK, &l.attnV,
                                           &l.attnO, &l.ffnGate, &l.ffnUp,
                                           &l.ffnDown};
                for (QuantizedMatrix *m: mats) {
                    if ((m->type != 18 && m->type != 21) || m->cols == 0 ||
                        (m->cols & 255u) != 0 || m->data.empty())
                        continue;
                    const uint32_t blocks = m->cols / 256u;
                    const uint32_t srcType = m->type;
                    const uint32_t srcBytes = ggmlTypeSize(srcType);
                    AlignedVector<uint8_t> out;
                    out.resize(static_cast<size_t>(m->rows) * blocks * 144u);
                    for (uint32_t r = 0; r < m->rows; ++r) {
                        const uint8_t *src = m->data.data() +
                                             static_cast<size_t>(r) * blocks * srcBytes;
                        uint8_t *dst = out.data() +
                                       static_cast<size_t>(r) * blocks * 144u;
                        for (uint32_t b = 0; b < blocks; ++b) {
                            if (srcType == 18) {
                                GGMLDequantize::dequantizeIQ3_XXSBlock(
                                        src + b * srcBytes, x.data());
                            } else {
                                GGMLDequantize::dequantizeIQ3_SBlock(
                                        src + b * srcBytes, x.data());
                            }
                            GGMLQuantize::quantizeQ4KBlock(x.data(), dst + b * 144u);
                        }
                    }
                    m->data = std::move(out);
                    m->type = GGML_TYPE_Q4_K;
                    m->prepackedData.clear();
                    ++converted;
                }
            }
            if (converted != 0) {
                std::cout << "[TinyCoder] IQ3 projections requantized to Q4_K ("
                          << converted << " matrices)" << std::endl;
            }
        }

        void requantQ6KToQ4K(std::vector<Model::LayerWeights> &layers,
                             uint32_t hiddenSize) {
            const char *e = std::getenv("TINYCODER_Q6_Q4K");
            // Opt-in only: these rewrites replace a lossless K-quant with
            // a re-quantised Q4_K.  That is a numerics change, not a
            // formatting one, and it silently corrupted every checkpoint
            // with hidden >= 2048 (7B incoherence, 9B, 35B-A3B).  Kept as
            // an explicit experiment until the quantiser passes a
            // round-trip test.
            if (e == nullptr || e[0] == '0') return;
            const bool onlyQ8 = std::getenv("TINYCODER_Q8_Q4K") != nullptr;
            // Dense 9B only: expert Q6_K (35B) is too large/quality-critical.
            if (hiddenSize != 4096u || layers.size() < 32u || layers.size() > 33u) return;
            uint32_t converted = 0;
            std::vector<float> x(256);
            for (auto &l: layers) {
                QuantizedMatrix *mats[] = {&l.attnQ, &l.attnK, &l.attnV,
                                           &l.attnO, &l.ffnGate, &l.ffnUp,
                                           &l.ffnDown, &l.attnQKV, &l.attnGate,
                                           &l.ssmOut};
                for (QuantizedMatrix *m: mats) {
                    if ((m->type != 14 && m->type != 13 &&
                         !(onlyQ8 && m->type == 8)) ||
                        m->cols == 0 ||
                        (m->cols & 255u) != 0 || m->data.empty())
                        continue;
                    const uint32_t blocks = m->cols / 256u;
                    const uint32_t srcType = m->type;
                    const uint32_t srcBytes = ggmlTypeSize(srcType);
                    AlignedVector<uint8_t> out;
                    out.resize(static_cast<size_t>(m->rows) * blocks * 144u);
                    for (uint32_t r = 0; r < m->rows; ++r) {
                        const uint8_t *src = m->data.data() +
                                             static_cast<size_t>(r) * blocks * srcBytes;
                        uint8_t *dst = out.data() +
                                       static_cast<size_t>(r) * blocks * 144u;
                        for (uint32_t b = 0; b < blocks; ++b) {
                            if (srcType == 14) {
                                GGMLDequantize::dequantizeQ6_KBlock(
                                        src + b * srcBytes, x.data());
                            } else if (srcType == 13) {
                                GGMLDequantize::dequantizeQ5_KBlock(
                                        src + b * srcBytes, x.data());
                            } else {
                                for (uint32_t sub = 0; sub < 8; ++sub)
                                    GGMLDequantize::dequantizeQ8_0Block(
                                            src + b * srcBytes + sub * 34u,
                                            x.data() + sub * 32u);
                            }
                            GGMLQuantize::quantizeQ4KBlock(x.data(), dst + b * 144u);
                        }
                    }
                    m->data = std::move(out);
                    m->type = GGML_TYPE_Q4_K;
                    m->prepackedData.clear();
                    ++converted;
                }
            }
            if (converted != 0) {
                std::cout << "[TinyCoder] Q6_K/Q5_K/Q8_0 matrices requantized to Q4_K ("
                          << converted << ")" << std::endl;
            }
        }

    }// namespace

    bool Model::setContextLength(uint32_t tokens) {
        if (loaded_) {
            // The host cache and the device K/V are sized from the window at
            // load time; changing it afterwards would desync every one of them.
            return false;
        }
        requestedSeqLen_ = tokens;
        config_.maxSeqLen = tokens;
        return true;
    }

    bool Model::load(const std::string &modelPath, std::string *outError,
                     ProgressCallback progressCb) {
        std::cout << "[TinyCoder] Loading model from: " << modelPath << std::endl;

        // The caller's window request (Model::setContextLength, or the
        // max_seq_len that tc_engine_load writes into config_).  It has to be
        // captured here: the two loader.config() assignments below overwrite
        // config_ wholesale with the file's metadata, which is how the request
        // used to disappear before being clamped to 2048.
        const uint32_t requestedSeqLen = requestedSeqLen_ ? requestedSeqLen_
                                                          : config_.maxSeqLen;

        auto reportProgress = [&](float p, const std::string &stage) {
            std::cout << "[TinyCoder] " << stage << " (" << static_cast<int>(p * 100) << "%)" << std::endl;
            if (progressCb) {
                progressCb(p, stage);
            }
        };

        // Helper to set error message
        auto setError = [outError](const std::string &msg) {
            std::cerr << "[TinyCoder] " << msg << std::endl;
            if (outError) {
                *outError = msg;
            }
        };

        reportProgress(0.0f, "Checking model file...");

        // Check file existence
        std::ifstream testFile(modelPath, std::ios::binary);
        if (!testFile.is_open()) {
            setError("File not found or cannot be opened: " + modelPath);
            return false;
        }
        testFile.close();

        // Estimate memory requirements before loading.
        // The path is either a GGUF file or an EXL3/HF checkpoint directory; for
        // a directory the "size" is the sum of its file sizes (shards + config).
        uint64_t fileSize = 0;
        {
            namespace fs = std::filesystem;
            std::error_code ec;
            if (fs::is_directory(fs::path(modelPath), ec)) {
                for (const auto &entry: fs::recursive_directory_iterator(
                             modelPath, fs::directory_options::skip_permission_denied, ec)) {
                    if (entry.is_regular_file(ec)) {
                        fileSize += entry.file_size(ec);
                    }
                }
            } else {
                std::ifstream sizeFile(modelPath, std::ios::binary | std::ios::ate);
                const std::streampos end = sizeFile.tellg();
                fileSize = end > 0 ? static_cast<uint64_t>(end) : 0;
            }
        }

        reportProgress(0.05f, "Reading model file metadata...");

        // Pick the loader from the path (extension/magic based): a *.gguf file
        // (GGUF v3) or an EXL3/HF safetensors checkpoint. config_ is the only
        // thing needed up front (for the memory estimate), so read metadata only.
        const bool isExl3 = EXL3Loader::isExl3Path(modelPath);
        std::unique_ptr<IModelLoader> metaLoader =
                isExl3 ? std::unique_ptr<IModelLoader>(new EXL3Loader())
                       : std::unique_ptr<IModelLoader>(new GGUFLoader());
        if (!metaLoader->loadMetadata(modelPath)) {
            setError(isExl3 ? "Failed to read EXL3 model config"
                            : "Failed to read GGUF metadata");
            return false;
        }
        config_ = metaLoader->config();

        // Estimate total memory needed
        uint64_t estimatedMem = config_.estimatedMemoryBytes();
        uint64_t fileMem = fileSize;

        // Get available system memory
        struct sysinfo si;
        uint64_t availableMem = 0;
        if (sysinfo(&si) == 0) {
            availableMem = static_cast<uint64_t>(si.freeram) * si.mem_unit;
        }

        std::cout << "[TinyCoder] Memory estimate: " << (estimatedMem / (1024 * 1024))
                  << " MB (weights: ~" << (fileMem / (1024 * 1024)) << " MB on disk)"
                  << ", available: " << (availableMem / (1024 * 1024)) << " MB"
                  << std::endl;

        // Bypass memory check for debugging
        if (availableMem > 0 && estimatedMem > availableMem) {
            std::cout << "[TinyCoder] WARNING: Insufficient memory: need ~"
                      << (estimatedMem / (1024 * 1024)) << " MB but only "
                      << (availableMem / (1024 * 1024))
                      << " MB available. Attempting to load anyway (may swap)..."
                      << std::endl;
        }

        reportProgress(0.1f, isExl3 ? "Loading EXL3 checkpoint..." : "Loading GGUF file...");

        // Load the full model file. The loader is kept alive for the Model's
        // lifetime in weightLoader_ (declared before every weight field in
        // Model.hpp so it is destroyed last): the quantized weight tensors are
        // adopted as EXTERNAL (zero-copy) buffers that alias this loader's
        // backing store (file-backed mmap for GGUF, loader-owned heap buffers
        // for EXL3), so the backing store must outlive every QuantizedMatrix
        // that borrows a pointer into it.
        weightLoader_ = isExl3 ? std::unique_ptr<IModelLoader>(new EXL3Loader())
                               : std::unique_ptr<IModelLoader>(new GGUFLoader());
        if (!weightLoader_->load(modelPath)) {
            setError(isExl3 ? "Failed to parse EXL3 checkpoint"
                            : "Failed to parse GGUF file (invalid or corrupted format)");
            return false;
        }
        IModelLoader &loader = *weightLoader_;

        config_ = loader.config();

        // ---- Resolve the context window -----------------------------------
        // The request survived the loader overwrite above; the checkpoint's own
        // advertised limit is what loader.config() brought back.  planContext
        // decides the window and the K/V precision from free VRAM (or RAM),
        // and its `reason` is what gets printed, so the message and the
        // decision cannot disagree.
        {
            const bool inklingArch = config_.architecture == ARCH_INKLING2 ||
                                     config_.architecture == ARCH_INKLING2_LOOP;
            const uint32_t kvLayers = config_.numKvSlots > 0 ? config_.numKvSlots
                                                             : config_.numLayers;
            const uint32_t interval = config_.fullAttentionInterval;
            const uint32_t numAttnLayers =
                    interval > 1u
                            ? (kvLayers + interval - 1u) / interval
                            : kvLayers;
            // qwen35/qwen35moe carry an MRoPE table next to the text one; both
            // are [maxSeqLen][dims/2] of cos+sin in f32.
            const uint32_t mropeDims =
                    (config_.architecture == ARCH_QWEN35 ||
                     config_.architecture == ARCH_QWEN35MOE)
                            ? (config_.ropeDimensionCount ? config_.ropeDimensionCount
                                                          : 64u)
                            : 0u;

            detail::ContextRequest req;
            req.requested = requestedSeqLen;
            req.modelMax = config_.maxSeqLen;
            req.numLayers = kvLayers;
            req.numAttentionLayers = numAttnLayers;
            req.numKVHeads = config_.numKVHeads;
            req.headDim = config_.headDim;
            req.ropeDims = config_.headDim + mropeDims;
            // An int8 K/V store needs a kernel that writes and reads the
            // quantized rows end to end.  Arch 1 (qwen35) has one: kQ35QuantKV8
            // + kQ35PrefillAttnSharedKV8 / kQ35WarpAttention8Prefill for the
            // prefill and the existing kQ35WarpAttention8* for the decode.  The
            // MoE arch (2) still only converts for decode, so granting int8
            // there would charge 1/4 of the K/V it actually allocates.
            req.int8KvSupported = config_.architecture == ARCH_QWEN35;
            // TINYCODER_CONTEXT_KV8=1|0 pins the K/V store precision, so the same
            // window can be served both ways for a numerics comparison.
            if (const char *fk = std::getenv("TINYCODER_CONTEXT_KV8");
                fk != nullptr && fk[0] != '\0' && req.int8KvSupported) {
                req.forceKv = fk[0] == '0' ? detail::KvPrecision::F32
                                           : detail::KvPrecision::I8;
                req.forceKvSet = true;
            }
            req.weightBytes = fileSize;
            // Inkling's CPU path owns the whole model and both K/V and conv
            // state live in host RAM (there is no GPU Inkling kernel yet), so
            // plan the window against RAM even when the weights would fit VRAM.
            req.freeVram = inklingArch ? 0 : detail::probeFreeVramBytes();
            // The RAM probe runs after the weights are resident, so it already
            // accounts for them; planContext must not subtract them again.
            req.availableRam = req.freeVram == 0 ? detail::probeAvailableRamBytes() : 0;
            contextPlan_ = detail::planContext(req);
            // A checkpoint whose weights do not fit the CARD AT ALL (a Q8 35B on
            // an 11 GB device) must not refuse to load: it is perfectly usable
            // with its weights in RAM, where the window is bounded by RAM instead.
            //
            // The test is "could this ever fit here", not "does it fit now": a
            // card that is merely busy (another process, a stale allocation from
            // the previous test in this process) also reports a small freeVram,
            // and silently dropping such a session to a 2 tok/s CPU path is far
            // worse than the reduction the ladder already handles.  Only when
            // the weights exceed the device's total capacity is the host the
            // only place they can live.
            const uint64_t totalVram = detail::probeTotalVramBytes();
            const bool cannotFitCard =
                    totalVram != 0 && req.weightBytes + req.gpuFixedOverhead > totalVram;
            if (contextPlan_.seqLen == 0 && req.freeVram != 0 && cannotFitCard) {
                const uint64_t ram = detail::probeAvailableRamBytes();
                if (ram > 0) {
                    detail::ContextRequest ramReq = req;
                    ramReq.freeVram = 0;
                    ramReq.availableRam = ram;
                    const detail::ContextPlan ramPlan = detail::planContext(ramReq);
                    if (ramPlan.seqLen > 0) contextPlan_ = ramPlan;
                }
            }
            // TINYCODER_CONTEXT_DEBUG=1 prints the inputs the grant was made
            // from.  Every number here has moved a ceiling at least once, so the
            // report line alone is not enough to explain a surprising window.
            {
                const char *dbg = std::getenv("TINYCODER_CONTEXT_DEBUG");
                if (dbg != nullptr && dbg[0] != '\0' && dbg[0] != '0') {
                    std::fprintf(stderr,
                                 "[ctx] freeVram=%llu MiB weights=%llu MiB fixed=%llu MiB "
                                 "attnLayers=%u kvHeads=%u headDim=%u ropeDims=%u int8=%d\n",
                                 (unsigned long long) (req.freeVram >> 20),
                                 (unsigned long long) (req.weightBytes >> 20),
                                 (unsigned long long) (req.gpuFixedOverhead >> 20),
                                 req.numAttentionLayers, req.numKVHeads, req.headDim,
                                 req.ropeDims, req.int8KvSupported ? 1 : 0);
                    std::fprintf(stderr,
                                 "[ctx] perToken f32=%llu f32+mirror=%llu int8=%llu\n",
                                 (unsigned long long) detail::kvBytesPerToken(
                                         req, detail::KvPrecision::F32, false),
                                 (unsigned long long) detail::kvBytesPerToken(
                                         req, detail::KvPrecision::F32, true),
                                 (unsigned long long) detail::kvBytesPerToken(
                                         req, detail::KvPrecision::I8, false));
                }
            }

            if (contextPlan_.seqLen == 0) {
                setError("cannot grant a context window: " + contextPlan_.reason);
                return false;
            }
            config_.maxSeqLen = contextPlan_.seqLen;

            std::cout << "[TinyCoder] context window: " << contextPlan_.seqLen
                      << " tokens (" << contextPlan_.reason << "; "
                      << (contextPlan_.kv == detail::KvPrecision::I8 ? "K/V int8"
                                                                     : "K/V f32")
                      << (contextPlan_.kvMirror ? " + int8 mirror" : "")
                      << ", " << (contextPlan_.kvBytes / (1024 * 1024)) << " MB, "
                      << (contextPlan_.kvOnGpu ? "on GPU" : "in host RAM") << ")"
                      << std::endl;
        }

        // Validate model architecture
        if (!isSupportedArchitecture(config_.architecture)) {
            setError("Unsupported model architecture: \"" + config_.architecture +
                     "\". Supported architectures: " + ARCH_QWEN2 + ", " +
                     ARCH_GEMMA4 + ", " + ARCH_QWEN35MOE + ", " + ARCH_QWEN35);
            return false;
        }

        // Validate model name is in the supported list (optional check)
        if (!config_.modelName.empty() && !isSupportedModel(config_.modelName)) {
            std::cout << "[TinyCoder] Warning: model name \"" << config_.modelName
                      << "\" not in known list, but architecture \""
                      << config_.architecture << "\" is supported. Proceeding..."
                      << std::endl;
        }

        std::cout << "[TinyCoder] Model: \"" << config_.modelName << "\" ("
                  << config_.architecture << ")" << std::endl;
        std::cout << "[TinyCoder] Model config: " << config_.numLayers << " layers, "
                  << config_.hiddenSize << " hidden, " << config_.numAttentionHeads
                  << " heads, " << config_.numKVHeads << " KV heads" << std::endl;

        // Validate config
        if (config_.numLayers == 0) {
            setError("Model has zero layers \u2014 GGUF metadata may be incomplete or "
                     "unsupported architecture");
            return false;
        }
        if (config_.hiddenSize == 0 || config_.vocabSize == 0) {
            setError("Model has invalid dimensions (hiddenSize=" +
                     std::to_string(config_.hiddenSize) +
                     ", vocabSize=" + std::to_string(config_.vocabSize) + ")");
            return false;
        }

        reportProgress(0.2f, "Loading tokenizer...");

        // Configure tokenizer for the model architecture FIRST (sets defaults)
        tokenizer_.configureForArchitecture(config_.architecture);

        if (loader.isGguf()) {
            // Load tokenizer from GGUF (overrides BOS/EOS/PAD from metadata)
            if (!tokenizer_.loadFromGGUF(modelPath)) {
                std::cerr << "[TinyCoder] Failed to load tokenizer, using embedded data"
                          << std::endl;
            }

            // Sync vocab size to the tokenizer's actual vocabulary. Some GGUF
            // writers (e.g. Unsloth Qwen3.8-27B-UD) omit tokenizer.ggml.vocab_size,
            // leaving the Qwen2.5-Coder default here; the real vocab count comes
            // from tokenizer.ggml.tokens (which matches token_embd.weight rows /
            // output.weight rows = the LM head width).
            if (tokenizer_.vocabSize() > 0) {
                config_.vocabSize = static_cast<uint32_t>(tokenizer_.vocabSize());
            }
        } else {
            // EXL3 checkpoints ship a Hugging Face tokenizer.json. Load it so
            // the model-dependent tests (and generation) can tokenize prompts.
            // config_.tokenizerPath is set by EXL3Loader when the file exists.
            std::string tokPath = config_.tokenizerPath;
            if (tokPath.empty()) {
                const auto *exl3 = dynamic_cast<const EXL3Loader *>(&loader);
                if (exl3 != nullptr) {
                    const std::string candidate = exl3->directory() + "/tokenizer.json";
                    if (std::filesystem::exists(candidate)) {
                        tokPath = candidate;
                    }
                }
            }
            if (!tokPath.empty() &&
                tokenizer_.loadFromHFTokenizer(tokPath, config_.architecture)) {
                // Keep config_.vocabSize from the EXL3 loader: it is the LM-head /
                // embedding row count (config.json / tensor shapes), which is the
                // authoritative generation width.  The tokenizer's own vocab may
                // differ by a few added tokens and must not resize the LM head.
            } else {
                std::cerr << "[TinyCoder] EXL3 tokenizer.json loading failed; "
                             "using the architecture's embedded tokenizer defaults."
                          << std::endl;
            }
        }

        reportProgress(0.25f, "Loading model weights...");

        // Load weights (stored in native quantized format)
        if (!loadWeights(loader)) {
            setError("Failed to load model weights");
            return false;
        }

        requantIQ2XSGateUpToQ4K(layers_);
        requantIQ3ToQ4K(layers_, config_.hiddenSize);
        requantQ6KToQ4K(layers_, config_.hiddenSize);

        reportProgress(0.7f, "Building pre-dequantized embeddings...");

        // Build pre-dequantized embedding matrix for fast LM head computation
        // This eliminates the expensive dequantization from every forward pass
        std::cout << "[TinyCoder] Building pre-dequantized embedding matrix..."
                  << std::endl;
        // When the CUDA path is available the full-offload upload frees these
        // tables anyway (they are CPU-LM-head-only), so building them first and
        // freeing them after leaves the *peak* RSS unchanged — the rented 3090
        // report measured 22.5 GiB of host RSS on a 34 GB model for exactly
        // this.  Defer the build to first CPU use instead (Model::forward
        // rebuilds when the table is empty); a CPU-only build still builds it
        // here, where it is the only path.
        // A CUDA build with the GPU path left enabled will upload the LM head to
        // the device, so the host tables are dead weight until something falls
        // back; if it does, Model::forward rebuilds them.  `TINYCODER_GPU=0`
        // (and a non-CUDA build) keeps the eager build.
        const bool deferEmbTables = [&] {
            // Inkling runs its LM head from the quantized output matrix and
            // never reads the F32/F16/Q8_K embedding tables.
            if (config_.architecture == ARCH_INKLING2 ||
                config_.architecture == ARCH_INKLING2_LOOP) {
                return true;
            }
#ifdef USE_CUDA
            const char *e = std::getenv("TINYCODER_GPU");
            return !(e != nullptr && e[0] == '0');
#else
            return false;
#endif
        }();
        if (deferEmbTables) {
            std::cout << "[TinyCoder] embedding tables deferred to first CPU use "
                         "(GPU path owns the LM head)"
                      << std::endl;
        }
        if (!deferEmbTables && !buildDequantizedEmbeddings()) {
            // The dequantized table is not optional: the generation-time LM
            // head and the sampler read it, and continuing without it produced
            // coherent-looking logits but garbage text (caught by the text gate,
            // not by the logits probe).
            setError("Failed to build pre-dequantized embeddings");
            return false;
        }

        reportProgress(0.9f, "Initializing KV cache...");

        // Initialize KV cache
        if (!initKVCache()) {
            setError("Failed to initialize KV cache (out of memory?)");
            return false;
        }

        // Create this model's own CPU thread pool. One pool per Model (rather
        // than the process-wide singleton) is what lets two resident models in
        // the same process dispatch work without clobbering each other's task
        // publication, and it makes n_threads a per-model setting.
        //
        // Size: config_.nThreads when set. Otherwise inherit the size of the
        // process-wide instance() if a caller already initialized it (the bench
        // and unit tests do, so --threads keeps working), else the recommended
        // logical-CPU count.
        if (!pool_) {
            size_t n = 0;
            if (const char *te = std::getenv("TINYCODER_NTHREADS");
                te != nullptr && te[0] != '\0') {
                n = static_cast<size_t>(std::atoi(te));
            } else if (config_.nThreads > 0) {
                n = static_cast<size_t>(config_.nThreads);
            } else if (ThreadPool::instance().isInitialized()) {
                n = ThreadPool::instance().numThreads();
            } else {
                n = ThreadPool::recommendedThreadCount();
            }
            pool_ = std::make_unique<ThreadPool>();
            pool_->initialize(n);
            std::cout << "[TinyCoder] CPU thread pool initialized with " << n
                      << " threads" << std::endl;
        }

        loaded_ = true;
        reportProgress(1.0f, "Model loaded successfully");
        std::cout << "[TinyCoder] Model loaded successfully" << std::endl;
        return true;
    }

    uint64_t Model::estimateMemory(const std::string &modelPath) {
        std::unique_ptr<IModelLoader> loader =
                EXL3Loader::isExl3Path(modelPath)
                        ? std::unique_ptr<IModelLoader>(new EXL3Loader())
                        : std::unique_ptr<IModelLoader>(new GGUFLoader());
        if (!loader->loadMetadata(modelPath)) {
            return 0;
        }
        return loader->config().estimatedMemoryBytes();
    }

    bool Model::loadWeights(IModelLoader &loader) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // ---- Prism Hadamard fold setup (Ternary-Bonsai-2-27B PQ2_0) ----
        // When prism.hadamard.* metadata is present the listed weights were
        // stored folded (W_folded = W @ (H S P)); the matching activations are
        // transformed at runtime (Model::applyPrismTransform). Sign vectors are
        // keyed by input width and sliced out of the concatenated sign_values by
        // sign_widths.
        {
            const auto &pm = loader.prism();
            if (pm.present) {
                prismFold_ = true;
                prismBlockSize_ = pm.blockSize;
                prismGdnVGrouped_ = pm.gdnVGrouped;
                config_.prismFold = true;
                config_.prismBlockSize = pm.blockSize;
                config_.prismGdnVGrouped = pm.gdnVGrouped;
                prismWeightNames_ = pm.foldedWeights;
                size_t off = 0;
                for (uint32_t w: pm.signWidths) {
                    std::vector<float> v(w, 1.0f);
                    for (uint32_t i = 0; i < w && off + i < pm.signValues.size(); ++i) {
                        v[i] = static_cast<float>(pm.signValues[off + i]);
                    }
                    prismSigns_[w] = std::move(v);
                    off += w;
                }
                std::cout << "[TinyCoder] Prism Hadamard fold: block=" << pm.blockSize
                          << " folded weights=" << pm.foldedWeights.size()
                          << " latent tables=" << pm.inverseWeights.size()
                          << " sign widths=";
                for (size_t si = 0; si < pm.signWidths.size(); ++si) {
                    std::cout << (si ? "," : "") << pm.signWidths[si];
                }
                std::cout << " gdnVGrouped=" << (pm.gdnVGrouped ? 1 : 0) << std::endl;
            }
        }
        // True when `name` is a folded weight whose activation must be
        // transformed before the matmul.
        auto isFoldedWeight = [&](const std::string &name) {
            return prismFold_ &&
                   std::find(prismWeightNames_.begin(), prismWeightNames_.end(),
                             name) != prismWeightNames_.end();
        };

        // Helper to convert a packed FP16 weight matrix to a row-major Q8_K block
        // array (rows x blocksPerRow Q8KBlock), for the prefill Q8_K batch GEMM.
        auto buildQ8K = [](const uint16_t *W_f16, uint32_t rows, uint32_t cols) {
            constexpr uint32_t BLOCK_SIZE = 256;
            uint32_t blocksPerRow = (cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
            std::vector<Q8KBlock> out(static_cast<size_t>(rows) * blocksPerRow);
            std::vector<float> rowF32(cols);
            // Manual FP16 -> FP32 (Model.cpp is not compiled with F16C).
            for (uint32_t r = 0; r < rows; ++r) {
                const uint16_t *src = W_f16 + static_cast<size_t>(r) * cols;
                for (uint32_t i = 0; i < cols; ++i) {
                    uint16_t h = src[i];
                    uint32_t sign = (h >> 15) & 1;
                    uint32_t exp = (h >> 10) & 0x1F;
                    uint32_t mant = h & 0x3FF;
                    uint32_t f32;
                    if (exp == 0) {
                        if (mant == 0) {
                            f32 = sign << 31;
                        } else {
                            int n = 0;
                            while ((mant & 0x200) == 0 && n < 10) {
                                mant <<= 1;
                                ++n;
                            }
                            mant &= 0x3FF;
                            exp = 112 - n;
                            f32 = (sign << 31) | (exp << 23) | ((mant - 512) << 14);
                        }
                    } else if (exp == 31) {
                        f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
                    } else {
                        exp = exp + (127 - 15);
                        f32 = (sign << 31) | (exp << 23) | (mant << 13);
                    }
                    float v;
                    std::memcpy(&v, &f32, sizeof(float));
                    rowF32[i] = v;
                }
                GGMLDequantize::quantizeQ8K(rowF32.data(), cols,
                                            out.data() + static_cast<size_t>(r) * blocksPerRow);
            }
            return out;
        };

        // Helper to load a quantized weight tensor into a QuantizedMatrix.
        // Copies the raw quantized data without dequantizing.
        auto loadQuantized = [&](const std::string &name) -> QuantizedMatrix {
            auto info = loader.getTensorInfo(name);
            if (!info) {
                // Optional tensors are probed across all layers (e.g. the
                // qwen35 MTP `nextn.*` block exists only in models that were
                // exported with multi-token prediction; Qwen3.6-27B-Q5_K_M has
                // none).  A missing optional tensor is normal, so stay quiet —
                // required tensors are validated by the callers that need them.
                return QuantizedMatrix{};
            }

            // EXL3 linears are uploaded from the packed trellis; the host F16
            // copy is optional and materialized lazily only for CPU fallback.
            const auto *exl3Loader = dynamic_cast<const EXL3Loader *>(&loader);
            const bool lazyLinear = exl3Loader && exl3Loader->isLazyLinear(name);
            const Exl3LinearInfo *native =
                    exl3Loader ? exl3Loader->nativeLinear(name) : nullptr;
            const uint8_t *data = lazyLinear ? nullptr : loader.getTensor(name);
            if (!data && native == nullptr) {
                std::cerr << "[TinyCoder] Tensor data not found: " << name << std::endl;
                return QuantizedMatrix{};
            }

            // GGUF stores shapes in reverse (numpy) order.
            // For a 2D tensor, shape[0] = cols, shape[1] = rows.
            // For a 3D tensor [D0, D1, D2] in numpy → GGUF shape [D2, D1, D0]:
            //   shape[0] = D2 (cols), shape[1] = D1, shape[2] = D0
            //   total rows = D0 * D1 = shape[1] * shape[2]
            uint32_t rows =
                    info->shape.size() >= 2 ? static_cast<uint32_t>(info->shape[1]) : 1;
            uint32_t cols =
                    info->shape.size() >= 1 ? static_cast<uint32_t>(info->shape[0]) : 1;
            if (info->shape.size() == 1) {
                rows = 1;
                cols = static_cast<uint32_t>(info->shape[0]);
            }
            // For 3D tensors (expert weights), multiply rows by the third dimension
            if (info->shape.size() >= 3) {
                rows *= static_cast<uint32_t>(info->shape[2]);
            }

            uint32_t blockSize = ggmlBlockSize(info->type);
            uint32_t typeSize = ggmlTypeSize(info->type);
            uint64_t dataBytes = static_cast<uint64_t>(rows) * cols * sizeof(float);// fallback

            if (blockSize > 1 && typeSize > 0) {
                // Blocks are per-row: each row has (cols + blockSize - 1) / blockSize blocks.
                // Total blocks = rows * blocksPerRow.
                uint64_t blocksPerRow = (static_cast<uint64_t>(cols) + blockSize - 1) / blockSize;
                uint64_t totalBlocks = static_cast<uint64_t>(rows) * blocksPerRow;
                dataBytes = totalBlocks * typeSize;
            } else if (typeSize > 0) {
                dataBytes = static_cast<uint64_t>(rows) * cols * typeSize;
            }

            std::cout << "[TinyCoder] Loading " << name << ": type=" << info->type
                      << " shape=[";
            for (size_t si = 0; si < info->shape.size(); ++si) {
                if (si > 0)
                    std::cout << ",";
                std::cout << info->shape[si];
            }
            std::cout << "] rows=" << rows << " cols=" << cols
                      << " dataBytes=" << (dataBytes / (1024 * 1024)) << " MB"
                      << std::endl;

            QuantizedMatrix qm;
            qm.rows = rows;
            qm.cols = cols;
            qm.type = info->type;
            // EXL3 checkpoints: keep the packed trellis available so the CUDA
            // adapter can upload the native compressed linear instead of the
            // F16 materialization (the CPU path still uses `data`).
            qm.exl3 = native;
            // Mark folded (Prism-rotated) weights so the forward pass routes
            // their activation through applyPrismTransform.
            if (isFoldedWeight(name)) {
                qm.foldedWeight = true;
            }
            // Zero-copy: adopt the tensor bytes DIRECTLY from the loader's
            // file-backed mmap (kept alive by Model::weightLoader_) instead of
            // heap-copying them. The previous qm.data.assign() copied every
            // quantized tensor (~36.9 GB) into the heap on top of the resident
            // mmap pages, doubling RSS and OOM-killing the process at ~layer
            // 16/40 on the 31 GB host. The SIMD kernels only use unaligned
            // loads, so the mmap pointer (which is not 64-byte aligned) is
            // safe to hand to them directly.
            if (data != nullptr) {
                qm.data.setExternal(data, static_cast<size_t>(dataBytes));
            }
            // PTQ1_0 is transcoded to the PQ2_0 block layout here, once, so the
            // rest of the engine only ever sees one ternary format.  Lossless
            // (same trits, same fp16 group scale); costs the 28 -> 34 byte
            // expansion in RAM/VRAM, which is the same footprint the native
            // PQ2_0 file already needs on this box.
            if (qm.type == GGML_TYPE_PTQ1_0 && data != nullptr &&
                !ptq1NativeEnabled()) {
                const uint64_t blocksPerRowT =
                        (static_cast<uint64_t>(cols) + 127u) / 128u;
                const uint64_t blocks = static_cast<uint64_t>(rows) * blocksPerRowT;
                std::vector<uint8_t> out(static_cast<size_t>(blocks * 34u));
                transcodePTQ1_0ToPQ2_0(data, out.data(),
                                       static_cast<size_t>(blocks));
                qm.data.assign(out);
                qm.type = GGML_TYPE_PQ2_0;
            }
            return qm;
        };

        // Helper to load a 1D F32 tensor (norms, etc.)
        auto loadF32_1D = [&](const std::string &name) -> tinycoder::Array<float> {
            auto info = loader.getTensorInfo(name);
            if (!info)
                return tinycoder::Array<float>{};

            const uint8_t *data = loader.getTensor(name);
            if (!data)
                return tinycoder::Array<float>{};

            uint32_t n = static_cast<uint32_t>(info->shape[0]);
            std::vector<float> vec(n);
            std::memcpy(vec.data(), data, n * sizeof(float));
            return tinycoder::Array<float>(vec, tinycoder::Shape{n});
        };

        // Helper to load any-rank F32 tensor as a flat vector.  Inkling's
        // clip vectors and depth-attention queries are 2-D F32 ([headDim,
        // nHeads], [rows, hidden]) and never go through a quantized matmul.
        auto loadF32Flat = [&](const std::string &name) -> tinycoder::Array<float> {
            auto info = loader.getTensorInfo(name);
            if (!info)
                return tinycoder::Array<float>{};
            const uint8_t *data = loader.getTensor(name);
            if (!data)
                return tinycoder::Array<float>{};
            uint64_t n = 1;
            for (auto d: info->shape) n *= d;
            if (info->type != GGML_TYPE_F32 || n == 0) {
                std::cerr << "[TinyCoder] " << name << ": expected F32, got type "
                          << info->type << std::endl;
                return tinycoder::Array<float>{};
            }
            std::vector<float> vec(n);
            std::memcpy(vec.data(), data, n * sizeof(float));
            return tinycoder::Array<float>(vec, tinycoder::Shape{static_cast<size_t>(n)});
        };

        // Load a quantized weight, converting a BF16 (type 30) tensor to F32.
        // The Prism Bonsai export stores ssm_alpha / ssm_beta as BF16; the engine
        // has no BF16 kernel, so convert the 2-byte bf16 values to F32 at load
        // (F32 takes the fast AVX2 matmul path; these matrices are tiny). Other
        // types pass through unchanged.
        auto loadQuantizedBF16ToF32 = [&](const std::string &name) -> QuantizedMatrix {
            QuantizedMatrix qm = loadQuantized(name);
            if (qm.empty() || qm.type != GGML_TYPE_BF16) {
                return qm;
            }
            const uint64_t n = static_cast<uint64_t>(qm.rows) * qm.cols;
            const uint16_t *src = reinterpret_cast<const uint16_t *>(qm.data.data());
            std::vector<float> f32(n);
            for (uint64_t i = 0; i < n; ++i) {
                f32[i] = GGMLDequantize::bf16ToFloat(src[i]);
            }
            QuantizedMatrix out;
            out.rows = qm.rows;
            out.cols = qm.cols;
            out.type = GGML_TYPE_F32;
            out.data.assign(reinterpret_cast<const uint8_t *>(f32.data()),
                            reinterpret_cast<const uint8_t *>(f32.data()) + n * sizeof(float));
            std::cout << "[TinyCoder] " << name << ": converted BF16 -> F32 ("
                      << qm.rows << "x" << qm.cols << ")" << std::endl;
            return out;
        };

        // Load token embeddings (keep quantized)
        {
            auto info = loader.getTensorInfo("token_embd.weight");
            if (!info) {
                std::cerr << "[TinyCoder] Missing tensor: token_embd.weight" << std::endl;
                return false;
            }

            const uint8_t *data = loader.getTensor("token_embd.weight");
            if (!data) {
                std::cerr << "[TinyCoder] Failed to get tensor data: token_embd.weight"
                          << std::endl;
                return false;
            }

            // GGUF stores shapes in reverse (numpy) order.
            // For a 2D tensor, shape[0] = cols (hiddenSize), shape[1] = rows (vocabSize).
            uint32_t rows =
                    info->shape.size() >= 2 ? static_cast<uint32_t>(info->shape[1]) : 1;
            uint32_t cols =
                    info->shape.size() >= 1 ? static_cast<uint32_t>(info->shape[0]) : 1;
            if (info->shape.size() == 1) {
                rows = 1;
                cols = static_cast<uint32_t>(info->shape[0]);
            }

            // Calculate compressed size based on quantization type
            uint32_t blockSize = ggmlBlockSize(info->type);
            uint32_t typeSize = ggmlTypeSize(info->type);
            uint64_t numElements = static_cast<uint64_t>(rows) * cols;
            uint64_t numBlocks = (numElements + blockSize - 1) / blockSize;
            uint64_t compressedBytes = numBlocks * typeSize;

            // Prism stores the token table "latent" (rows rotated by H S); the
            // looked-up rows are restored to the primal basis in getRow().
            bool latent = false;
            if (prismFold_) {
                const auto &inv = loader.prism().inverseWeights;
                latent = std::find(inv.begin(), inv.end(), std::string("token_embd.weight")) != inv.end();
            }
            quantizedEmbeddings_.data.assign(data, data + compressedBytes);
            quantizedEmbeddings_.vocabSize = rows;
            quantizedEmbeddings_.hiddenSize = cols;
            quantizedEmbeddings_.type = info->type;
            // PTQ1_0 table: transcode to the PQ2_0 block layout once, exactly as
            // loadQuantized() does for every other ternary tensor, so the latent
            // embedding dequant + inverse Hadamard path is shared.
            // The embedding table stays in the PQ2_0 layout: it is read once
            // per token, so the 28 B saving is irrelevant, and the shared
            // latent dequant / inverse-Hadamard path is the validated one.
            if (quantizedEmbeddings_.type == GGML_TYPE_PTQ1_0) {
                std::vector<uint8_t> out(static_cast<size_t>(numBlocks * 34u));
                transcodePTQ1_0ToPQ2_0(data, out.data(),
                                       static_cast<size_t>(numBlocks));
                quantizedEmbeddings_.data.assign(out.begin(), out.end());
                quantizedEmbeddings_.type = GGML_TYPE_PQ2_0;
            }
            if (latent) {
                quantizedEmbeddings_.isLatentPrism = true;
                quantizedEmbeddings_.hadamardBlockSize = prismBlockSize_;
                const float *s = signFor(cols);
                if (s != nullptr) {
                    quantizedEmbeddings_.signs.assign(s, s + cols);
                }
            }
            std::cout << "[TinyCoder] Token embeddings: " << rows << " x " << cols
                      << " (type " << info->type << ", "
                      << (compressedBytes / (1024 * 1024)) << " MB quantized"
                      << (latent ? ", Prism latent)" : ")") << std::endl;
        }

        // Check if LM head is tied (output.weight exists and has same pointer as
        // token_embd.weight)
        auto lmHeadInfo = loader.getTensorInfo("output.weight");
        if (lmHeadInfo) {
            const auto *exl3Loader = dynamic_cast<const EXL3Loader *>(&loader);
            const bool nativeLm =
                    exl3Loader && exl3Loader->nativeLinear("output.weight") != nullptr;
            auto lmHeadData = loader.getTensor("output.weight");
            auto embData = loader.getTensor("token_embd.weight");
            if (lmHeadData && embData && lmHeadData == embData) {
                // Same data pointer = weight tying
                lmHeadTied_ = true;
                std::cout << "[TinyCoder] LM head: tied with token embeddings"
                          << std::endl;
            } else if (nativeLm || lmHeadData) {
                // Separate LM head - load as quantized matrix
                lmHead_ = loadQuantized("output.weight");
                if (!lmHead_.empty()) {
                    lmHeadTied_ = false;
                    std::cout << "[TinyCoder] LM head: separate, "
                              << ((lmHead_.exl3 ? lmHead_.exl3->trellisBytes
                                                : lmHead_.data.size()) /
                                  (1024 * 1024))
                              << " MB quantized"
                              << std::endl;
                } else {
                    lmHeadTied_ = true;
                    std::cout << "[TinyCoder] LM head: tied (failed to load separate)"
                              << std::endl;
                }
            }
        } else {
            lmHeadTied_ = true;
            std::cout << "[TinyCoder] LM head: tied (no output.weight tensor)"
                      << std::endl;
        }

        // Load final norm (F32)
        finalNorm_ = loadF32_1D("output_norm.weight");
        if (finalNorm_.empty()) {
            finalNorm_ = loadF32_1D("token_embd_norm.weight");
        }

        if (config_.architecture == ARCH_INKLING2 ||
            config_.architecture == ARCH_INKLING2_LOOP) {
            inklingTokNorm_ = loadF32_1D("token_embd_norm.weight");
            inklingOutputResQuery_ = loadF32_1D("output_res_query.weight");
            if (config_.inklingExpandedLayout) {
                // Legacy/expanded layout: 2 queries per core block, grouped
                // into one tensor each for (Q, FFN) and (K, V).
                inklingLoopResQuery_ = loadF32Flat("loop_res_query.weight");
                inklingLoopKvResQuery_ = loadF32Flat("loop_kv_res_query.weight");
                const uint32_t core = config_.loopEnd - config_.loopStart;
                if (inklingLoopResQuery_.size() != size_t(2) * core * config_.hiddenSize ||
                    inklingLoopKvResQuery_.size() != size_t(2) * core * config_.hiddenSize) {
                    std::cerr << "[TinyCoder] inkling2: bad legacy loop query sizes ("
                              << inklingLoopResQuery_.size() << ", "
                              << inklingLoopKvResQuery_.size() << ")" << std::endl;
                    return false;
                }
            }
            if (inklingTokNorm_.empty() || inklingOutputResQuery_.empty()) {
                std::cerr << "[TinyCoder] inkling2: missing token/output norm or "
                             "output_res_query"
                          << std::endl;
                return false;
            }
        }

        // Load per-layer weights (architecture-aware)
        layers_.resize(config_.numLayers);
        for (uint32_t i = 0; i < config_.numLayers; ++i) {
            std::string prefix = "blk." + std::to_string(i) + ".";

            if (config_.architecture == ARCH_QWEN2) {
                // ---- Qwen2 architecture ----
                // Quantized attention weights
                layers_[i].attnQ = loadQuantized(prefix + "attn_q.weight");
                layers_[i].attnK = loadQuantized(prefix + "attn_k.weight");
                layers_[i].attnV = loadQuantized(prefix + "attn_v.weight");
                layers_[i].attnO = loadQuantized(prefix + "attn_output.weight");

                // Dequantize attnO to FP16 for exact float dot product (halves memory bandwidth vs F32)
                {
                    auto &qm = layers_[i].attnO;
                    uint64_t numElements = static_cast<uint64_t>(qm.rows) * qm.cols;
                    layers_[i].attnO_deq_f16 = GGMLDequantize::dequantizeToF16(qm.type, qm.data.data(), numElements);
                }
                // Pre-pack attnO as Q8_K for the prefill Q8_K batch GEMM (P4).
                layers_[i].attnO_q8k =
                        buildQ8K(layers_[i].attnO_deq_f16.data(),
                                 layers_[i].attnO.rows, layers_[i].attnO.cols);

                // Pre-pack attnQ and attnK for the register-tiled Q8_K batch GEMM
                // prefill path (Q2_K only). This lets the prefill Q/K projections
                // reuse each weight row across all prompt tokens and use the int8
                // _mm256_maddubs_epi16 kernel, instead of the generic row-parallel
                // quantized mat-mul path (see BENCHMARK_REPORT §7.2 "remaining
                // prefill bottleneck"). attnV is usually Q4_K and stays on the
                // generic path.
                if (layers_[i].attnQ.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].attnQ.rows) * layers_[i].attnQ.cols;
                    layers_[i].attnQ.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].attnQ.data.data(), numElements);
                }
                if (layers_[i].attnK.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].attnK.rows) * layers_[i].attnK.cols;
                    layers_[i].attnK.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].attnK.data.data(), numElements);
                }

                // Quantized FFN weights (SwiGLU)
                layers_[i].ffnGate = loadQuantized(prefix + "ffn_gate.weight");
                layers_[i].ffnUp = loadQuantized(prefix + "ffn_up.weight");
                layers_[i].ffnDown = loadQuantized(prefix + "ffn_down.weight");

                // Dequantize ffnDown to FP16
                {
                    auto &qm = layers_[i].ffnDown;
                    uint64_t numElements = static_cast<uint64_t>(qm.rows) * qm.cols;
                    layers_[i].ffnDown_deq_f16 = GGMLDequantize::dequantizeToF16(qm.type, qm.data.data(), numElements);
                }
                // Pre-pack ffnDown as Q8_K for the prefill Q8_K batch GEMM (P4).
                layers_[i].ffnDown_q8k =
                        buildQ8K(layers_[i].ffnDown_deq_f16.data(),
                                 layers_[i].ffnDown.rows, layers_[i].ffnDown.cols);

                // Pre-pack ffnGate and ffnUp for gather-free SIMD access (Q2_K only)
                // This eliminates the 2-bit extraction overhead in the SIMD kernel.
                if (layers_[i].ffnGate.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].ffnGate.rows) * layers_[i].ffnGate.cols;
                    layers_[i].ffnGate.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].ffnGate.data.data(), numElements);
                }
                if (layers_[i].ffnUp.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].ffnUp.rows) * layers_[i].ffnUp.cols;
                    layers_[i].ffnUp.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].ffnUp.data.data(), numElements);
                }
                // Pre-pack ffnDown for the register-tiled Q8_K batch GEMM prefill
                // path (Q2_K only). This lets the prefill ffnDown projection reuse
                // each weight row across all tokens and use the int8
                // _mm256_maddubs_epi16 kernel.
                if (layers_[i].ffnDown.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].ffnDown.rows) * layers_[i].ffnDown.cols;
                    layers_[i].ffnDown.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].ffnDown.data.data(), numElements);
                }


                // F32 norms
                layers_[i].rmsNormAttn = loadF32_1D(prefix + "attn_norm.weight");
                layers_[i].rmsNormFFN = loadF32_1D(prefix + "ffn_norm.weight");

                // F32 Q, K, V biases (Qwen2.5-Coder has these)
                layers_[i].attnQBias = loadF32_1D(prefix + "attn_q.bias");
                layers_[i].attnKBias = loadF32_1D(prefix + "attn_k.bias");
                layers_[i].attnVBias = loadF32_1D(prefix + "attn_v.bias");

                // Validate all tensors loaded
                if (layers_[i].attnQ.empty() || layers_[i].attnK.empty() ||
                    layers_[i].attnV.empty() || layers_[i].attnO.empty()) {
                    std::cerr << "[TinyCoder] Missing attention weights for layer " << i
                              << std::endl;
                    return false;
                }
            } else if (config_.architecture == ARCH_GEMMA4) {
                // ---- Gemma4 architecture (dense and MoE) ----
                // Quantized attention weights (no biases)
                layers_[i].attnQ = loadQuantized(prefix + "attn_q.weight");
                layers_[i].attnK = loadQuantized(prefix + "attn_k.weight");
                layers_[i].attnV = loadQuantized(prefix + "attn_v.weight");
                layers_[i].attnO = loadQuantized(prefix + "attn_output.weight");

                // Dequantize attnO to FP16 for exact float dot product (halves memory bandwidth vs F32)
                {
                    auto &qm = layers_[i].attnO;
                    uint64_t numElements = static_cast<uint64_t>(qm.rows) * qm.cols;
                    layers_[i].attnO_deq_f16 = GGMLDequantize::dequantizeToF16(qm.type, qm.data.data(), numElements);
                }

                // Quantized FFN weights (GeGLU: gate+up, down)
                layers_[i].ffnGate = loadQuantized(prefix + "ffn_gate.weight");
                layers_[i].ffnUp = loadQuantized(prefix + "ffn_up.weight");
                layers_[i].ffnDown = loadQuantized(prefix + "ffn_down.weight");

                // Dequantize ffnDown to FP16
                {
                    auto &qm = layers_[i].ffnDown;
                    uint64_t numElements = static_cast<uint64_t>(qm.rows) * qm.cols;
                    layers_[i].ffnDown_deq_f16 = GGMLDequantize::dequantizeToF16(qm.type, qm.data.data(), numElements);
                }

                // Pre-pack ffnGate and ffnUp for gather-free SIMD access (Q2_K only)
                if (layers_[i].ffnGate.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].ffnGate.rows) * layers_[i].ffnGate.cols;
                    layers_[i].ffnGate.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].ffnGate.data.data(), numElements);
                }
                if (layers_[i].ffnUp.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].ffnUp.rows) * layers_[i].ffnUp.cols;
                    layers_[i].ffnUp.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].ffnUp.data.data(), numElements);
                }
                // Pre-pack ffnDown for the register-tiled Q8_K batch GEMM prefill
                // path (Q2_K only).
                if (layers_[i].ffnDown.type == GGML_TYPE_Q2_K) {
                    uint64_t numElements = static_cast<uint64_t>(layers_[i].ffnDown.rows) * layers_[i].ffnDown.cols;
                    layers_[i].ffnDown.prepackedData = GGMLDequantize::prepackQ2_K(layers_[i].ffnDown.data.data(), numElements);
                }

                // F32 norms
                layers_[i].rmsNormAttn = loadF32_1D(prefix + "attn_norm.weight");
                layers_[i].rmsNormFFN = loadF32_1D(prefix + "ffn_norm.weight");

                // Gemma4-specific: Q/K norms (RMSNorm before RoPE)
                layers_[i].attnQNorm = loadF32_1D(prefix + "attn_q_norm.weight");
                layers_[i].attnKNorm = loadF32_1D(prefix + "attn_k_norm.weight");

                // Gemma4-specific: post-attention and post-FFW norms
                layers_[i].postAttnNorm = loadF32_1D(prefix + "post_attention_norm.weight");
                layers_[i].postFFWNorm = loadF32_1D(prefix + "post_ffw_norm.weight");

                // Gemma4-specific: layer output scale
                layers_[i].layerOutputScale = loadF32_1D(prefix + "layer_output_scale.weight");

                // Gemma4 MoE-specific weights
                if (config_.expertCount > 0) {
                    // Expert router
                    layers_[i].ffnGateInp = loadQuantized(prefix + "ffn_gate_inp.weight");
                    // Expert scales (loaded as F32 1D)
                    {
                        auto info = loader.getTensorInfo(prefix + "ffn_gate_inp.scale");
                        if (info) {
                            const uint8_t *data = loader.getTensor(prefix + "ffn_gate_inp.scale");
                            if (data) {
                                uint32_t n = static_cast<uint32_t>(info->shape[0]);
                                std::vector<float> vec(n);
                                std::memcpy(vec.data(), data, n * sizeof(float));
                                // Store in layerOutputScale (reuse field)
                                // ffn_gate_inp.scale is a per-expert scale factor
                            }
                        }
                    }
                    // Fused gate+up expert weights: [expertCount, expertFF * 2, hiddenSize]
                    layers_[i].ffnGateUpExps = loadQuantized(prefix + "ffn_gate_up_exps.weight");
                    // Down expert weights: [expertCount, hiddenSize, expertFF]
                    layers_[i].ffnDownExps = loadQuantized(prefix + "ffn_down_exps.weight");

                    // Additional norms for MoE
                    layers_[i].preFFWNorm2 = loadF32_1D(prefix + "pre_ffw_norm_2.weight");
                    layers_[i].postFFWNorm1 = loadF32_1D(prefix + "post_ffw_norm_1.weight");
                    layers_[i].postFFWNorm2 = loadF32_1D(prefix + "post_ffw_norm_2.weight");
                }

                // Validate all tensors loaded
                if (layers_[i].attnQ.empty() || layers_[i].attnK.empty() ||
                    layers_[i].attnV.empty() || layers_[i].attnO.empty()) {
                    std::cerr << "[TinyCoder] Missing attention weights for layer " << i
                              << std::endl;
                    return false;
                }
            } else if (config_.architecture == ARCH_QWEN35MOE) {
                const bool isRecurrent = detail::isQwen35RecurrentLayer(config_, i);

                // Common norms
                layers_[i].rmsNormAttn = loadF32_1D(prefix + "attn_norm.weight");
                layers_[i].postAttnNorm = loadF32_1D(prefix + "post_attention_norm.weight");

                // ---- MoE FFN weights (routed experts) ----
                layers_[i].ffnGateInpMoe = loadQuantized(prefix + "ffn_gate_inp.weight");
                // When fused, the compute path slices rows [e*ff*2, e*ff*2+ff)
                // as gate and [e*ff*2+ff, (e+1)*ff*2) as up from
                // ffnGateUpExpsMoe.
                layers_[i].ffnGateUpExpsMoe = loadQuantized(prefix + "ffn_gate_up_exps.weight");
                if (layers_[i].ffnGateUpExpsMoe.empty()) {
                    layers_[i].ffnGateExps = loadQuantized(prefix + "ffn_gate_exps.weight");
                    layers_[i].ffnUpExps = loadQuantized(prefix + "ffn_up_exps.weight");
                }
                layers_[i].ffnDownExpsMoe = loadQuantized(prefix + "ffn_down_exps.weight");

                // Shared expert weights + router (sigmoid gate)
                layers_[i].ffnGateInpShexp = loadQuantized(prefix + "ffn_gate_inp_shexp.weight");
                layers_[i].ffnGateShexp = loadQuantized(prefix + "ffn_gate_shexp.weight");
                layers_[i].ffnUpShexp = loadQuantized(prefix + "ffn_up_shexp.weight");
                layers_[i].ffnDownShexp = loadQuantized(prefix + "ffn_down_shexp.weight");

                if (isRecurrent) {
                    // ---- Gated delta net (linear attention) tensors ----
                    layers_[i].attnQKV = loadQuantized(prefix + "attn_qkv.weight");
                    layers_[i].attnGate = loadQuantized(prefix + "attn_gate.weight");
                    layers_[i].ssmConv1d = loadQuantized(prefix + "ssm_conv1d.weight");// F32 [4, 8192]
                    layers_[i].ssmABroadcast = loadF32_1D(prefix + "ssm_a");
                    layers_[i].ssmAlphaQ = loadQuantized(prefix + "ssm_alpha.weight");// F32 [2048, 32]
                    layers_[i].ssmBetaQ = loadQuantized(prefix + "ssm_beta.weight");  // F32 [2048, 32]
                    layers_[i].ssmDtBiasFull = loadF32_1D(prefix + "ssm_dt.bias");
                    layers_[i].ssmNorm = loadF32_1D(prefix + "ssm_norm.weight");// [headV]
                    layers_[i].ssmOut = loadQuantized(prefix + "ssm_out.weight");

                    if (layers_[i].attnQKV.empty() || layers_[i].attnGate.empty() ||
                        layers_[i].ssmConv1d.empty() || layers_[i].ssmOut.empty()) {
                        std::cerr << "[TinyCoder] Missing recurrent tensors for layer "
                                  << i << std::endl;
                        return false;
                    }
                } else {
                    // ---- Full-attention tensors ----
                    layers_[i].attnQ = loadQuantized(prefix + "attn_q.weight");// Q+gate fused
                    layers_[i].attnK = loadQuantized(prefix + "attn_k.weight");
                    layers_[i].attnV = loadQuantized(prefix + "attn_v.weight");
                    layers_[i].attnO = loadQuantized(prefix + "attn_output.weight");
                    layers_[i].attnQNorm = loadF32_1D(prefix + "attn_q_norm.weight");
                    layers_[i].attnKNorm = loadF32_1D(prefix + "attn_k_norm.weight");

                    if (layers_[i].attnQ.empty() || layers_[i].attnK.empty() ||
                        layers_[i].attnV.empty() || layers_[i].attnO.empty()) {
                        std::cerr << "[TinyCoder] Missing attention weights for layer "
                                  << i << std::endl;
                        return false;
                    }
                }

                // ---- MTP (Multi-Token Prediction) weights ----
                layers_[i].nextnEhProj = loadQuantized(prefix + "nextn.eh_proj.weight");
                layers_[i].nextnEnorm = loadF32_1D(prefix + "nextn.enorm.weight");
                layers_[i].nextnHnorm = loadF32_1D(prefix + "nextn.hnorm.weight");
                layers_[i].nextnSharedHeadNorm = loadF32_1D(prefix + "nextn.shared_head_norm.weight");
            } else if (config_.architecture == ARCH_QWEN35) {
                // ---- Qwen35 architecture (dense: gated delta net + full attention) ----
                //
                // All layers share attn_norm, post_attention_norm,
                // ffn_gate/up/down.
                const bool isRecurrent = detail::isQwen35RecurrentLayer(config_, i);

                // Common norms
                layers_[i].rmsNormAttn = loadF32_1D(prefix + "attn_norm.weight");
                layers_[i].postAttnNorm = loadF32_1D(prefix + "post_attention_norm.weight");

                // Common FFN (SwiGLU): quantized path (no FP16/Q8_K prepacks here)
                layers_[i].ffnGate = loadQuantized(prefix + "ffn_gate.weight");
                layers_[i].ffnUp = loadQuantized(prefix + "ffn_up.weight");
                layers_[i].ffnDown = loadQuantized(prefix + "ffn_down.weight");

                if (isRecurrent) {
                    // ---- Gated delta net (linear attention) tensors ----
                    layers_[i].attnQKV = loadQuantized(prefix + "attn_qkv.weight");
                    layers_[i].attnGate = loadQuantized(prefix + "attn_gate.weight");
                    layers_[i].ssmConv1d = loadQuantized(prefix + "ssm_conv1d.weight");// F32 [4, 10240]
                    layers_[i].ssmABroadcast = loadF32_1D(prefix + "ssm_a");
                    layers_[i].ssmAlphaQ = loadQuantizedBF16ToF32(prefix + "ssm_alpha.weight");
                    layers_[i].ssmBetaQ = loadQuantizedBF16ToF32(prefix + "ssm_beta.weight");
                    layers_[i].ssmDtBiasFull = loadF32_1D(prefix + "ssm_dt.bias");
                    layers_[i].ssmNorm = loadF32_1D(prefix + "ssm_norm.weight");
                    layers_[i].ssmOut = loadQuantized(prefix + "ssm_out.weight");

                    if (layers_[i].attnQKV.empty() || layers_[i].attnGate.empty() ||
                        layers_[i].ssmConv1d.empty() || layers_[i].ssmOut.empty()) {
                        std::cerr << "[TinyCoder] Missing recurrent tensors for layer "
                                  << i << std::endl;
                        return false;
                    }
                } else {
                    // ---- Full-attention tensors (incl. MTP block) ----
                    layers_[i].attnQ = loadQuantized(prefix + "attn_q.weight");
                    layers_[i].attnK = loadQuantized(prefix + "attn_k.weight");
                    layers_[i].attnV = loadQuantized(prefix + "attn_v.weight");
                    layers_[i].attnO = loadQuantized(prefix + "attn_output.weight");
                    layers_[i].attnQNorm = loadF32_1D(prefix + "attn_q_norm.weight");
                    layers_[i].attnKNorm = loadF32_1D(prefix + "attn_k_norm.weight");

                    // MTP (nextn) tensors (only present in the blk.<nLayer> block)
                    layers_[i].nextnEhProj = loadQuantized(prefix + "nextn.eh_proj.weight");
                    layers_[i].nextnEnorm = loadF32_1D(prefix + "nextn.enorm.weight");
                    layers_[i].nextnHnorm = loadF32_1D(prefix + "nextn.hnorm.weight");
                    layers_[i].nextnSharedHeadNorm =
                            loadF32_1D(prefix + "nextn.shared_head_norm.weight");

                    if (layers_[i].attnQ.empty() || layers_[i].attnK.empty() ||
                        layers_[i].attnV.empty() || layers_[i].attnO.empty()) {
                        std::cerr << "[TinyCoder] Missing full-attention tensors for layer "
                                  << i << std::endl;
                        return false;
                    }
                }
            } else if (config_.architecture == ARCH_INKLING2 ||
                       config_.architecture == ARCH_INKLING2_LOOP) {
                // ---- Inkling-2-Mini (looped MoE) ----
                auto &w = layers_[i];
                const bool isSwa = i < config_.slidingWindowPattern.size()
                                           ? config_.slidingWindowPattern[i] != 0
                                           : true;

                w.attnQ = loadQuantized(prefix + "attn_q.weight");
                w.attnK = loadQuantized(prefix + "attn_k.weight");
                w.attnV = loadQuantized(prefix + "attn_v.weight");
                w.attnO = loadQuantized(prefix + "attn_output.weight");
                w.inkling.attnGate = loadQuantized(prefix + "attn_gate.weight");
                w.inkling.attnRel = loadQuantized(prefix + "attn_rel.weight");
                w.inkling.attnRelBias = loadQuantized(prefix + "attn_rel_bias.weight");
                w.inkling.attnSinkProj =
                        loadQuantizedBF16ToF32(prefix + "attn_sink_proj.weight");

                w.inkling.attnQClip = loadF32Flat(prefix + "attn_q_clip.weight");
                w.inkling.attnRelClip = loadF32Flat(prefix + "attn_rel_clip.weight");

                // Short convolutions (F32 [n_conv, channels]).
                w.inkling.attnKConv = loadQuantized(prefix + "attn_k_conv.weight");
                w.inkling.attnVConv = loadQuantized(prefix + "attn_v_conv.weight");
                w.inkling.attnPostConv = loadQuantized(prefix + "attn_post_conv.weight");
                w.inkling.ffnPostConv = loadQuantized(prefix + "ffn_post_conv.weight");
                if (isSwa) {
                    w.rmsNormAttn = loadF32_1D(prefix + "attn_norm.weight");
                } else {
                    w.inkling.attnQInpNorm =
                            loadF32_1D(prefix + "attn_q_inp_norm.weight");
                    w.inkling.attnKInpNorm =
                            loadF32_1D(prefix + "attn_k_inp_norm.weight");
                    w.inkling.attnVInpNorm =
                            loadF32_1D(prefix + "attn_v_inp_norm.weight");
                    w.inkling.attnKConvProj =
                            loadQuantized(prefix + "attn_k_conv_proj.weight");
                    w.inkling.attnVConvProj =
                            loadQuantized(prefix + "attn_v_conv_proj.weight");
                }
                if (i > 0) {
                    w.inkling.attnResQuery =
                            loadF32Flat(prefix + "attn_res_query.weight");
                }
                w.inkling.ffnResQuery = loadF32_1D(prefix + "ffn_res_query.weight");

                w.rmsNormFFN = loadF32_1D(prefix + "ffn_norm.weight");
                w.ffnGateInp = loadQuantized(prefix + "ffn_gate_inp.weight");
                w.ffnGateUpExps = loadQuantized(prefix + "ffn_gate_up_exps.weight");
                w.ffnDownExps = loadQuantized(prefix + "ffn_down_exps.weight");
                w.inkling.expertProbsBias =
                        loadF32_1D(prefix + "exp_probs_b.bias");
                w.inkling.expertGateUpScale =
                        loadF32_1D(prefix + "ffn_gate_up_exps.scale");
                w.inkling.expertDownScale =
                        loadF32_1D(prefix + "ffn_down_exps.scale");

                if (i >= uint32_t(config_.loopStart) &&
                    i < uint32_t(config_.loopEnd)) {
                    if (!config_.inklingExpandedLayout) {
                        w.inkling.msgQuery =
                                loadQuantized(prefix + "loop_msg_query.weight");
                    }
                    // Control tensors.  The two layouts only rename them:
                    //   mainline: loop_q_ctrl_* / loop_k_ctrl_* / loop_v_ctrl_* /
                    //             loop_ffn_ctrl_*
                    //   legacy:   loop_attn_*   / loop_k_*      / loop_v_*     /
                    //             loop_mlp_*
                    static const char *ctrlMain[4] = {"q", "k", "v", "ffn"};
                    static const char *ctrlLegacy[4] = {"attn", "k", "v", "mlp"};
                    for (int c = 0; c < 4; ++c) {
                        const std::string base =
                                config_.inklingExpandedLayout
                                        ? "loop_" + std::string(ctrlLegacy[c])
                                        : "loop_" + std::string(ctrlMain[c]) + "_ctrl";
                        w.inkling.ctrlDown[c] =
                                loadQuantized(prefix + base + "_down.weight");
                        w.inkling.ctrlGain[c] =
                                loadQuantized(prefix + base + "_gain.weight");
                        w.inkling.ctrlForget[c] =
                                loadQuantized(prefix + base + "_forget.weight");
                    }
                }

                if (w.attnQ.empty() || w.attnK.empty() || w.attnV.empty() ||
                    w.attnO.empty() || w.inkling.attnGate.empty() ||
                    w.inkling.attnRel.empty() || w.inkling.attnRelBias.empty() ||
                    w.ffnGateInp.empty() || w.ffnGateUpExps.empty() ||
                    w.ffnDownExps.empty() || w.rmsNormFFN.empty()) {
                    std::cerr << "[TinyCoder] Missing Inkling attention/MoE tensors "
                                 "for block "
                              << i << std::endl;
                    return false;
                }
            } else {
                std::cerr << "[TinyCoder] Unknown architecture: " << config_.architecture
                          << " for layer " << i << std::endl;
                return false;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "[TinyCoder] Weights loaded in " << ms << " ms" << std::endl;

        return true;
    }

    bool Model::ensureHostKv(uint32_t firstCpuLayer) {
        // Inkling owns one K/V slot per logical layer (the loop passes each
        // keep their own cache), so the slot count is numKvSlots, not the
        // number of physical weight blocks.
        const uint32_t nLayers = config_.numKvSlots > 0 ? config_.numKvSlots
                                                        : config_.numLayers;
        if (firstCpuLayer >= nLayers) {
            // The device owns every layer; no host slots are needed.
            hostKvFirst_ = firstCpuLayer;
            hostKvBaseSlot_ = 0;
            hostKvCount_ = 0;
            hostKvReady_ = false;
            return true;
        }
        if (hostKvReady_ && hostKvFirst_ == firstCpuLayer) return true;

        // Slot map, identical to the device's kvLayerOffset(): the K/V slots
        // belong to the full-attention layers, numbered layer/interval.  A
        // recurrent layer's number collides with an attention layer's, but a
        // recurrent layer never addresses K/V.
        const uint32_t interval = config_.fullAttentionInterval;
        const auto slotOf = [interval](uint32_t layer) {
            return interval > 1u ? layer / interval : layer;
        };
        const uint32_t lastSlot = interval > 1u
                                          ? (nLayers + interval - 1u) / interval
                                          : nLayers;
        const uint32_t baseSlot = slotOf(firstCpuLayer);
        const uint32_t count = lastSlot > baseSlot ? lastSlot - baseSlot : 0;

        const uint32_t maxSeq = config_.maxSeqLen;
        if (count == 0) {
            hostKvFirst_ = firstCpuLayer;
            hostKvBaseSlot_ = baseSlot;
            hostKvCount_ = 0;
            hostKvReady_ = true;
            return true;
        }
        const uint64_t kvBytes = static_cast<uint64_t>(count) * maxSeq *
                                 config_.numKVHeads * config_.headDim *
                                 sizeof(float) * 2;
        // A partial offload keeps the CPU tail's K/V in host RAM, and the plan
        // charged that window at the GRANTED precision (int8 is 4x smaller) and
        // budgets RAM only on the CPU-only path.  Refuse with the numbers
        // instead of letting `bad_alloc` surface as an empty-logits forward.
        const uint64_t ramAvail = detail::probeAvailableRamBytes();
        if (ramAvail != 0 && kvBytes + (512ull << 20) > ramAvail) {
            std::cerr << "[TinyCoder] host K/V for " << count << " slots x "
                      << maxSeq << " tokens needs " << (kvBytes >> 20)
                      << " MiB, but only " << (ramAvail >> 20)
                      << " MiB of RAM is available (GPU offload leaves no host "
                         "cache room): request a smaller window or offload more "
                         "layers"
                      << std::endl;
            return false;
        }
        try {
            kvCache_.k = tinycoder::Array<float>(
                    tinycoder::Shape{count, maxSeq, config_.numKVHeads, config_.headDim});
            kvCache_.v = tinycoder::Array<float>(
                    tinycoder::Shape{count, maxSeq, config_.numKVHeads, config_.headDim});
        } catch (const std::exception &e) {
            std::cerr << "[TinyCoder] host K/V allocation failed (" << e.what()
                      << "): " << count << " slots x " << maxSeq << " tokens"
                      << std::endl;
            return false;
        }
        hostKvFirst_ = firstCpuLayer;
        hostKvBaseSlot_ = baseSlot;
        hostKvCount_ = count;
        hostKvReady_ = true;
        const uint64_t bytes = static_cast<uint64_t>(count) * maxSeq *
                               config_.numKVHeads * config_.headDim *
                               sizeof(float) * 2;
        std::cout << "[TinyCoder] host K/V: " << count << " slots (layers "
                  << firstCpuLayer << ".." << (nLayers - 1) << ") x " << maxSeq
                  << " tokens = " << (bytes / (1024 * 1024)) << " MB (CPU path)"
                  << std::endl;
        return true;
    }

    uint32_t Model::hostKvSlot(uint32_t layer) const {
        const uint32_t interval = config_.fullAttentionInterval;
        const uint32_t slot = interval > 1u ? layer / interval : layer;
        if (!hostKvReady_ || slot < hostKvBaseSlot_) {
            // Only reachable from a debug dump taken before any forward, or by
            // addressing a layer the host does not own.
            std::fprintf(stderr,
                         "[TinyCoder] hostKvSlot(%u) with no host K/V for it "
                         "(base %u, count %u)\n",
                         layer, hostKvBaseSlot_, hostKvCount_);
            return 0;
        }
        const uint32_t idx = slot - hostKvBaseSlot_;
        if (idx >= hostKvCount_) {
            // A slot-map mismatch used to alias the last slot, which corrupts
            // attention and returns plausible garbage -- the worst failure mode
            // here.  Say it once per layer and use slot 0 so the session stays
            // deterministic while the message names the defect.
            std::fprintf(stderr,
                         "[TinyCoder] hostKvSlot(%u): slot %u outside the host "
                         "cache [%u, %u]; the host/device slot maps disagree\n",
                         layer, slot, hostKvBaseSlot_,
                         hostKvBaseSlot_ + hostKvCount_);
            return 0;
        }
        return idx;
    }

    bool Model::initKVCache() {
        uint32_t nLayers = config_.numLayers;
        // Qwen35 layers are hybrids: full-attention layers use the KV cache,
        // recurrent (gated-delta-net) layers use the SSM/conv state. The KV
        // cache for the recurrent layers is unused (they don't store K/V); only
        // the full-attention layers (and the MTP block) read/write it. Allocate
        // the full numLayers x maxSeq cache anyway to keep the indexing simple.

        // The host K/V cache is NOT allocated here: a GPU session never runs a
        // CPU attention layer, and a CPU-only session only needs the layers it
        // actually runs (and, on a qwen35 hybrid, only their full-attention
        // ones).  ensureHostKv() allocates it on the first CPU-path forward,
        // sized from the granted window.
        kvCache_.pos = 0;

        // ---- Qwen35/Qwen35MoE (gated delta net) recurrent state ----
        // Both architectures use the same GDN recurrence: per-layer conv state
        // [(convKernel-1)*convChannels] and per-value-head square [headV, headV]
        // GDN state.  Only recurrent layers allocate it.
        if ((config_.architecture == ARCH_QWEN35 ||
             config_.architecture == ARCH_QWEN35MOE) &&
            config_.ssmInnerSize > 0 && config_.ssmConvKernel > 1) {
            const uint32_t convChannels =
                    2 * config_.ssmStateSize * config_.ssmGroupCount +
                    config_.ssmInnerSize;
            kvCache_.q35ConvState.resize(nLayers);
            kvCache_.q35GdnState.resize(nLayers);
            uint32_t headV = config_.ssmInnerSize / config_.ssmTimeStepRank;
            uint32_t nRecurrent = 0;
            for (uint32_t i = 0; i < nLayers; ++i) {
                if (!detail::isQwen35RecurrentLayer(config_, i)) {
                    continue;
                }
                kvCache_.q35ConvState[i].assign(
                        (config_.ssmConvKernel - 1) * convChannels, 0.0f);
                // one square [headV, headV] state per value-head
                kvCache_.q35GdnState[i].assign(
                        static_cast<size_t>(config_.ssmTimeStepRank) *
                                static_cast<size_t>(headV) * headV,
                        0.0f);
                ++nRecurrent;
            }
            uint64_t memBytes =
                    static_cast<uint64_t>(nRecurrent) *
                    (static_cast<uint64_t>(config_.ssmConvKernel - 1) * convChannels +
                     static_cast<uint64_t>(config_.ssmTimeStepRank) * headV * headV) *
                    sizeof(float);
            std::cout << "[TinyCoder] Qwen35 recurrent state: " << nRecurrent
                      << " layers (conv " << config_.ssmConvKernel - 1 << "x"
                      << convChannels << " + gdn " << config_.ssmTimeStepRank
                      << "x" << headV << "x" << headV << ") = "
                      << (memBytes / (1024 * 1024)) << " MB" << std::endl;
        }

        if (config_.architecture == ARCH_INKLING2 ||
            config_.architecture == ARCH_INKLING2_LOOP) {
            const uint32_t kvDim = config_.numKVHeads * config_.headDim;
            const uint32_t channels = 2 * kvDim + 2 * config_.hiddenSize;
            const uint32_t nSlots = config_.numKvSlots > 0 ? config_.numKvSlots
                                                           : config_.numLayers;
            const uint32_t hist = config_.shortconvLCache > 1
                                          ? config_.shortconvLCache - 1
                                          : 0;
            inklingConvState_.assign(
                    nSlots, std::vector<float>(
                                    static_cast<size_t>(hist) * channels, 0.0f));
            std::cout << "[TinyCoder] Inkling short-conv state: " << nSlots
                      << " slots x " << hist << " x " << channels << " = "
                      << ((static_cast<uint64_t>(nSlots) * hist * channels *
                           sizeof(float)) /
                          (1024 * 1024))
                      << " MB" << std::endl;
        }

        return true;
    }

    bool Model::buildDequantizedEmbeddings() {
        if (quantizedEmbeddings_.empty()) {
            std::cerr << "[TinyCoder] No quantized embeddings to build" << std::endl;
            return false;
        }

        uint32_t vocabSize = quantizedEmbeddings_.vocabSize;
        uint32_t hiddenSize = quantizedEmbeddings_.hiddenSize;

        // Compute total memory needed
        uint64_t embedElements = static_cast<uint64_t>(vocabSize) * hiddenSize;
        uint64_t embedMemBytes = embedElements * sizeof(float);

        std::cout << "[TinyCoder] Pre-dequantizing embeddings: "
                  << vocabSize << " x " << hiddenSize
                  << " = " << (embedElements / (1024 * 1024)) << "M elements ("
                  << (embedMemBytes / (1024 * 1024)) << " MB)" << std::endl;

        auto t0 = std::chrono::high_resolution_clock::now();

        // Which host-side table to build.  The LM-head dispatch in
        // ModelForward tries, in this order: direct quantized block dot (Q2_K),
        // Q8_K, FP16, F32 -- so at most ONE of the three tables is ever read.
        // All three used to be resident for the whole process: for
        // 248320 x 5120 that is 5.1 GB (F32) + 2.5 GB (FP16) + 1.3 GB (Q8_K)
        // = 8.9 GB of anonymous RSS, which is why a 6.9 GB checkpoint showed
        // 9.7 GiB of heap (Bonsai pq2_0) and an 8.6 GB one showed 18.2 GiB
        // (Qwen3.8 27B EXL3).  Build only the table the chain selects; the
        // others are dead weight.  The chain's last resort (the generic
        // per-block dequantize at the end of the dispatch) reads the quantized
        // source directly, so no table is required for correctness.
        //   TINYCODER_EMB_TABLE=f32   also build the F32 table
        //   TINYCODER_EMB_TABLE=f16   also build the FP16 table
        //   TINYCODER_EMB_TABLE=all   build all three (legacy behaviour)
        const char *embEnv = std::getenv("TINYCODER_EMB_TABLE");
        const std::string embSel = (embEnv != nullptr) ? embEnv : "";
        const bool embAll = (embSel == "all");
        const bool embWantF32 = embAll || embSel == "f32";
        const bool embWantF16 = embAll || embSel == "f16";
        const bool embQ8K = quantizedEmbeddings_.type != GGML_TYPE_PQ2_0 &&
                            quantizedEmbeddings_.type != GGML_TYPE_BF16;

        dequantizedEmbeddings_.vocabSize = vocabSize;
        dequantizedEmbeddings_.hiddenSize = hiddenSize;

        // F32 table: opt-in only.  With Q8_K available it is never selected
        // (Q8_K wins on bandwidth), and with PQ2_0/BF16 the FP16 table below is
        // selected ahead of it -- so it is the third choice in every case.
        if (embWantF32) {
            auto deqData = GGMLDequantize::dequantize(
                    quantizedEmbeddings_.type,
                    quantizedEmbeddings_.data.data(),
                    embedElements);

            if (deqData.empty()) {
                std::cerr << "[TinyCoder] Failed to dequantize embeddings" << std::endl;
                return false;
            }

            dequantizedEmbeddings_.data = std::move(deqData);
            // Hint 2 MB huge pages for the F32 embedding table (streamed every
            // token by the LM head; collapses the dTLB working set — see
            // plans/generation_optimizations.md §6.10).
            tinycoder::adviseHugePages(dequantizedEmbeddings_.data.data(),
                                       dequantizedEmbeddings_.data.size() * sizeof(float));
        }

        // FP16 table: the selected fallback for types with no Q8_K dot kernel
        // (PQ2_0, BF16), otherwise opt-in only.  The LM head reads the whole
        // matrix every token, and FP16 halves that traffic versus F32.
        if (embWantF16 || !embQ8K) {
            auto deqF16 = GGMLDequantize::dequantizeToF16(
                    quantizedEmbeddings_.type,
                    quantizedEmbeddings_.data.data(),
                    embedElements);
            if (!deqF16.empty()) {
                dequantizedEmbeddings_.dataF16 = std::move(deqF16);
                tinycoder::adviseHugePages(dequantizedEmbeddings_.dataF16.data(),
                                           dequantizedEmbeddings_.dataF16.size() * sizeof(uint16_t));
            }
        }

        // Q8_K table: the chain's preference whenever the type supports it.  It
        // cuts bandwidth ~2x versus FP16 and ~4x versus F32 (each 256-element
        // block is 257 bytes), and since the source embeddings are Q2_K/EXL3
        // (~2 bit) the quantisation is lossless in practice.
        //
        // Rows are dequantized into one reusable F32 row buffer instead of into
        // a full [vocabSize][hiddenSize] matrix: the previous order built the
        // 5.1 GB F32 table only to throw it away when Q8_K replaced it, which
        // doubled the load-time peak for nothing.
        if (embQ8K) {
            const uint32_t srcBlockSize = ggmlBlockSize(quantizedEmbeddings_.type);
            const uint32_t srcTypeSize = ggmlTypeSize(quantizedEmbeddings_.type);
            const uint64_t srcBlocksPerRow = (hiddenSize + srcBlockSize - 1) / srcBlockSize;
            const uint32_t blocksPerRow = (hiddenSize + 255) / 256;
            std::vector<Q8KBlock> q8k(static_cast<size_t>(vocabSize) * blocksPerRow);
            // A block never crosses a row boundary (ggml lays the matrix out as
            // rows of whole blocks), so per-row dequantisation is bit-identical
            // to dequantising the whole matrix and slicing it.
            if ((hiddenSize % srcBlockSize) == 0) {
                std::vector<float> rowF32(hiddenSize);
                for (uint32_t r = 0; r < vocabSize; ++r) {
                    const uint8_t *rowSrc = quantizedEmbeddings_.data.data() +
                                            static_cast<uint64_t>(r) * srcBlocksPerRow * srcTypeSize;
                    for (uint64_t b = 0; b < srcBlocksPerRow; ++b) {
                        GGMLDequantize::dequantizeBlock(
                                quantizedEmbeddings_.type, rowSrc + b * srcTypeSize,
                                rowF32.data() + b * srcBlockSize, srcBlockSize);
                    }
                    GGMLDequantize::quantizeQ8K(
                            rowF32.data(), hiddenSize,
                            q8k.data() + static_cast<uint64_t>(r) * blocksPerRow);
                }
            } else {
                // Odd geometry (hidden size not a whole number of source
                // blocks): fall back to the whole-matrix dequantise.
                auto deq = GGMLDequantize::dequantize(
                        quantizedEmbeddings_.type,
                        quantizedEmbeddings_.data.data(),
                        embedElements);
                if (deq.empty()) {
                    std::cerr << "[TinyCoder] Failed to dequantize embeddings" << std::endl;
                    return false;
                }
                for (uint32_t r = 0; r < vocabSize; ++r) {
                    GGMLDequantize::quantizeQ8K(
                            deq.data() + static_cast<uint64_t>(r) * hiddenSize, hiddenSize,
                            q8k.data() + static_cast<uint64_t>(r) * blocksPerRow);
                }
            }
            dequantizedEmbeddings_.dataQ8K = std::move(q8k);
            tinycoder::adviseHugePages(dequantizedEmbeddings_.dataQ8K.data(),
                                       dequantizedEmbeddings_.dataQ8K.size() * sizeof(Q8KBlock));
        }

        // P0 (BENCHMARK_REPORT §5/P0): build per-16-element-subgroup dequantized
        // (min,max) bounds for a separate (non-tied) LM head. K-quants store a
        // per-16-element scale, so per-16 bounds are tight (candidate set
        // approximates top-K). The pruning pass reads only this small table
        // instead of the full ~182 MB Q6_K matrix to determine candidates, then
        // computes exact top-K logits only for those candidates.
        if (!lmHeadTied_ && !lmHead_.empty()) {
            uint32_t bSize = ggmlBlockSize(lmHead_.type);
            uint32_t tSize = ggmlTypeSize(lmHead_.type);
            if (bSize == 256) {
                uint32_t bpr = (lmHead_.cols + 255) / 256;
                constexpr uint32_t SCG = 16;
                uint32_t sgpb = bSize / SCG;// subgroups per block
                lmHeadBoundsBlocksPerRow_ = bpr;
                lmHeadBoundsSubgroupsPerBlock_ = sgpb;
                uint32_t totalSubgroups = bpr * sgpb;
                lmHeadBounds_.assign(static_cast<size_t>(lmHead_.rows) *
                                             totalSubgroups * 2,
                                     0.0f);
                const uint8_t *base = lmHead_.data.data();
                for (uint32_t r = 0; r < lmHead_.rows; ++r) {
                    float *boundsRow = lmHeadBounds_.data() +
                                       static_cast<size_t>(r) * totalSubgroups * 2;
                    for (uint32_t b = 0; b < bpr; ++b) {
                        const uint8_t *block = base +
                                               (static_cast<uint64_t>(r) * bpr + b) * tSize;
                        float blockOut[256];
                        GGMLDequantize::dequantizeBlock(lmHead_.type, block, blockOut, bSize);
                        // Per-16-subgroup min/max over the actual valid columns of
                        // the last (partial) block.
                        uint32_t blkCols = std::min<uint32_t>(bSize, lmHead_.cols - b * bSize);
                        for (uint32_t sg = 0; sg < sgpb; ++sg) {
                            uint32_t start = sg * SCG;
                            uint32_t n = std::min(SCG, blkCols - start);
                            float mn = blockOut[start], mx = blockOut[start];
                            for (uint32_t i = 1; i < n; ++i) {
                                mn = std::min(mn, blockOut[start + i]);
                                mx = std::max(mx, blockOut[start + i]);
                            }
                            boundsRow[(b * sgpb + sg) * 2 + 0] = mn;
                            boundsRow[(b * sgpb + sg) * 2 + 1] = mx;
                        }
                    }
                }
                std::cout << "[TinyCoder] Built LM head bounds: "
                          << lmHead_.rows << " rows x " << bpr << " blocks x "
                          << sgpb << " subgroups ("
                          << (lmHeadBounds_.size() * sizeof(float) / (1024 * 1024))
                          << " MB) for top-K pruning" << std::endl;
            } else {
                std::cerr << "[TinyCoder] P0 pruning requires 256-block quant type (got "
                          << bSize << "); pruning disabled" << std::endl;
            }
        }

        // Plan §3: pre-quantize the separate (non-tied) LM head to Q8_K. The LM
        // head mat-vec is the largest single memory read per token (vocabSize x
        // hiddenSize); storing it as Q8_K (256 int8 + block scale per block) cuts
        // that traffic ~2x vs on-the-fly dequantization, and the dot products
        // run through the int8 _mm256_maddubs_epi16 kernels with the hidden
        // vector quantized once per token. Since the source LM head uses K-quant
        // types, Q8_K is lossless. Mirrors the tied-embedding Q8_K build above.
        if (!lmHeadTied_ && !lmHead_.empty() && !lmHead_.data.empty()) {
            uint32_t srcBlockSize = ggmlBlockSize(lmHead_.type);
            uint32_t srcTypeSize = ggmlTypeSize(lmHead_.type);
            uint32_t srcBlocksPerRow = (lmHead_.cols + srcBlockSize - 1) / srcBlockSize;
            uint32_t bpr = (lmHead_.cols + 255) / 256;
            std::vector<Q8KBlock> q8k(static_cast<size_t>(lmHead_.rows) * bpr);
            std::vector<float> rowF32(lmHead_.cols);
            const uint8_t *base = lmHead_.data.data();
            float blockBuf[256];
            for (uint32_t r = 0; r < lmHead_.rows; ++r) {
                float *row = rowF32.data();
                for (uint32_t b = 0; b < srcBlocksPerRow; ++b) {
                    const uint8_t *block =
                            base + (static_cast<uint64_t>(r) * srcBlocksPerRow + b) * srcTypeSize;
                    uint32_t start = b * srcBlockSize;
                    uint32_t n = std::min(srcBlockSize, lmHead_.cols - start);
                    GGMLDequantize::dequantizeBlock(lmHead_.type, block, blockBuf, n);
                    std::memcpy(row + start, blockBuf, n * sizeof(float));
                }
                GGMLDequantize::quantizeQ8K(rowF32.data(), lmHead_.cols,
                                            q8k.data() + static_cast<size_t>(r) * bpr);
            }
            lmHeadQ8K_ = std::move(q8k);
            std::cout << "[TinyCoder] Pre-quantized separate LM head to Q8_K: "
                      << lmHead_.rows << " rows x " << bpr << " blocks ("
                      << (lmHeadQ8K_.size() * sizeof(Q8KBlock) / (1024 * 1024))
                      << " MB)" << std::endl;
        }


        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        std::cout << "[TinyCoder] Pre-dequantized embeddings in " << ms << " ms"
                  << std::endl;

        return true;
    }

    void Model::clearKVCache() {
        // Clear KV cache position and reset RNG state
        kvCache_.pos = 0;
        // A cleared window is usable again.
        contextFull_ = false;
        gpuContextFull_ = false;
        // Reset RNG state so generation is reproducible from a clean cache
        rngInitialized_ = false;

#ifdef USE_CUDA
        // If the GPU engine is active, reset its (persistent) KV cache too so a
        // new generation session starts from a clean state on both sides.
        gpuClearKV();
#endif

        // Zero out KV cache tensors to prevent residual data from previous generations
        std::fill(kvCache_.k.data(), kvCache_.k.data() + kvCache_.k.size(), 0.0f);
        std::fill(kvCache_.v.data(), kvCache_.v.data() + kvCache_.v.size(), 0.0f);

        // Clear SSM state for Qwen35MoE architecture
        for (auto &buf: kvCache_.ssmConvBuf) {
            std::fill(buf.begin(), buf.end(), 0.0f);
        }
        for (auto &state: kvCache_.ssmState) {
            std::fill(state.begin(), state.end(), 0.0f);
        }
        // Clear Qwen35 (dense gated delta net) recurrent state
        for (auto &buf: kvCache_.q35ConvState) {
            std::fill(buf.begin(), buf.end(), 0.0f);
        }
        for (auto &state: kvCache_.q35GdnState) {
            std::fill(state.begin(), state.end(), 0.0f);
        }
#ifdef USE_CUDA
        if (inklingGpu_) inklingGpu_->clearKVCache();
#endif
        // Inkling short-convolution history is recurrent state: without this,
        // a second generate() would see the previous conversation's last three
        // K/V/attention/FFN inputs.
        for (auto &buf: inklingConvState_) {
            std::fill(buf.begin(), buf.end(), 0.0f);
        }
    }

    void Model::resetCpuKVState() {
        // Same CPU-side reset body as clearKVCache(), but NEVER touches the GPU
        // engine (no gpuClearKV).  Used by the GPU adapter's Option-D profiling
        // pass, which runs with the adapter mutex held — calling clearKVCache()
        // -> gpuClearKV() -> adapter->clearKVCache() would deadlock.
        kvCache_.pos = 0;
        rngInitialized_ = false;
        // An empty cache is not full: the latches describe the last forward.
        contextFull_ = false;
        gpuContextFull_ = false;
        std::fill(kvCache_.k.data(), kvCache_.k.data() + kvCache_.k.size(), 0.0f);
        std::fill(kvCache_.v.data(), kvCache_.v.data() + kvCache_.v.size(), 0.0f);
        for (auto &buf: kvCache_.ssmConvBuf) {
            std::fill(buf.begin(), buf.end(), 0.0f);
        }
        for (auto &state: kvCache_.ssmState) {
            std::fill(state.begin(), state.end(), 0.0f);
        }
        for (auto &buf: kvCache_.q35ConvState) {
            std::fill(buf.begin(), buf.end(), 0.0f);
        }
        for (auto &state: kvCache_.q35GdnState) {
            std::fill(state.begin(), state.end(), 0.0f);
        }
        for (auto &buf: inklingConvState_) {
            std::fill(buf.begin(), buf.end(), 0.0f);
        }
    }

}// namespace tinycoder
