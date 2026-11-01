// exl3_model_smoke.cpp
//
// Loads a real EXL3 (ExLlamaV3) checkpoint through EXL3Loader and validates the
// materialized GGUF-named tensor inventory + numeric sanity.  This exercises
// exactly the path Model::loadWeights() consumes, without needing a tokenizer,
// and is the fast way to confirm a downloaded EXL3 checkpoint maps cleanly.
//
// Build (CPU-only, no SIMD):
//   cmake --build build-exl3 --target exl3_model_smoke
// Run:
//   ./build-exl3/exl3_model_smoke ~/qwen3.8-27B-exl3-1.60bpw
//
// Exits 0 when every required tensor is present, finite and non-degenerate.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "EXL3Loader.hpp"
#include "IModelLoader.hpp"
#include "ModelConfig.hpp"

using namespace tinycoder;

namespace {

    struct Stats {
        bool ok = true;
        uint64_t count = 0;
        double sumAbs = 0.0;
        float minV = 0.0f;
        float maxV = 0.0f;
        bool finite = true;
    };

    // Inspect a materialized F16/F32 1-D or 2-D tensor.
    Stats inspect(const IModelLoader &loader, const std::string &name) {
        Stats st;
        const TensorInfo *info = loader.getTensorInfo(name);
        const uint8_t *data = loader.getTensor(name);
        if (!info || !data) {
            st.ok = false;
            return st;
        }
        uint64_t n = 1;
        for (uint32_t d: info->shape) {
            n *= d;
        }
        st.count = n;
        st.minV = 1e30f;
        st.maxV = -1e30f;
        if (info->type == GGML_TYPE_F32) {
            const float *p = reinterpret_cast<const float *>(data);
            for (uint64_t i = 0; i < n; ++i) {
                float v = p[i];
                if (!std::isfinite(v)) {
                    st.finite = false;
                }
                st.sumAbs += std::fabs(v);
                st.minV = std::min(st.minV, v);
                st.maxV = std::max(st.maxV, v);
            }
        } else {
            // F16 (and everything else treated as 2-byte halves).
            const uint16_t *p = reinterpret_cast<const uint16_t *>(data);
            for (uint64_t i = 0; i < n; ++i) {
                uint32_t sign = (p[i] >> 15) & 1;
                uint32_t exp = (p[i] >> 10) & 0x1F;
                uint32_t mant = p[i] & 0x3FF;
                float v;
                if (exp == 0) {
                    v = mant ? std::ldexp(static_cast<float>(mant), -24) : 0.0f;
                } else if (exp == 31) {
                    st.finite = false;
                    v = 0.0f;
                } else {
                    v = std::ldexp(static_cast<float>(mant + 1024), static_cast<int>(exp) - 25);
                }
                if (sign) {
                    v = -v;
                }
                st.sumAbs += std::fabs(v);
                st.minV = std::min(st.minV, v);
                st.maxV = std::max(st.maxV, v);
            }
        }
        return st;
    }

    int failures = 0;

    void require(const IModelLoader &loader, const std::string &name, uint32_t expShape2 = 0,
                 uint32_t expShape1 = 0) {
        Stats st = inspect(loader, name);
        if (!st.ok) {
            std::cerr << "  MISSING  " << name << std::endl;
            ++failures;
            return;
        }
        const TensorInfo *info = loader.getTensorInfo(name);
        double mean = st.count ? st.sumAbs / static_cast<double>(st.count) : 0.0;
        bool shapeOk = true;
        if (expShape2 != 0) {
            // GGUF 2-D shape is reversed: {cols, rows}.
            shapeOk = info->shape.size() == 2 && info->shape[0] == expShape2 &&
                      info->shape[1] == expShape1;
        }
        if (!st.finite || mean == 0.0 || !shapeOk) {
            std::cerr << "  BAD      " << name << " shape=[";
            for (size_t i = 0; i < info->shape.size(); ++i) {
                std::cerr << (i ? "," : "") << info->shape[i];
            }
            std::cerr << "] finite=" << st.finite << " meanAbs=" << mean
                      << " range=[" << st.minV << "," << st.maxV << "]" << std::endl;
            ++failures;
            return;
        }
        std::cout << "  ok       " << name << "  n=" << st.count << " meanAbs=" << mean
                  << " range=[" << st.minV << "," << st.maxV << "]" << std::endl;
    }

}// namespace

int main(int argc, char **argv) {
    const std::string path = argc > 1 ? argv[1] : "";
    if (path.empty()) {
        std::cerr << "usage: exl3_model_smoke <checkpoint-dir-or-safetensors>" << std::endl;
        return 2;
    }

    EXL3Loader loader;
    std::cout << "[smoke] loading " << path << " ..." << std::endl;
    if (!loader.load(path)) {
        std::cerr << "[smoke] EXL3Loader::load failed" << std::endl;
        return 1;
    }

    const ModelConfig &cfg = loader.config();
    std::cout << "[smoke] arch=" << loader.detectedArchitecture() << " model_type="
              << loader.modelType() << " trellis=" << (loader.hasTrellis() ? "yes" : "no")
              << std::endl;
    std::cout << "[smoke] hidden=" << cfg.hiddenSize << " layers=" << cfg.numLayers
              << " heads=" << cfg.numAttentionHeads << " kvHeads=" << cfg.numKVHeads
              << " headDim=" << cfg.headDim << " vocab=" << cfg.vocabSize
              << " inter=" << cfg.intermediateSize << " nextn=" << cfg.nextnPredictLayers
              << std::endl;
    std::cout << "[smoke] gdn: nKHeads=" << cfg.ssmGroupCount << " headK=" << cfg.ssmStateSize
              << " inner=" << cfg.ssmInnerSize << " nVHeads=" << cfg.ssmTimeStepRank
              << " convK=" << cfg.ssmConvKernel << " fullAttnInterval=" << cfg.fullAttentionInterval
              << std::endl;

    const uint32_t H = cfg.hiddenSize;
    const uint32_t V = cfg.vocabSize;
    const uint32_t I = cfg.intermediateSize;
    const uint32_t HD = cfg.headDim;
    const uint32_t nH = cfg.numAttentionHeads;
    const uint32_t nKV = cfg.numKVHeads;
    const uint32_t keyDim = cfg.ssmGroupCount * cfg.ssmStateSize;
    const uint32_t valDim = cfg.ssmInnerSize;
    const uint32_t convDim = 2 * keyDim + valDim;

    std::cout << "[smoke] globals:" << std::endl;
    require(loader, "token_embd.weight", H, V);
    require(loader, "output.weight", H, V);
    require(loader, "output_norm.weight");
    if (cfg.nextnPredictLayers > 0) {
        std::cerr << "[smoke] WARNING: expected nextnPredictLayers=0 for HF Qwen3.5" << std::endl;
    }

    std::cout << "[smoke] layer 0 (recurrent / linear-attention):" << std::endl;
    require(loader, "blk.0.attn_norm.weight");
    require(loader, "blk.0.post_attention_norm.weight");
    require(loader, "blk.0.ffn_gate.weight", H, I);
    require(loader, "blk.0.ffn_up.weight", H, I);
    require(loader, "blk.0.ffn_down.weight", I, H);
    require(loader, "blk.0.attn_qkv.weight", H, convDim);
    require(loader, "blk.0.attn_gate.weight", H, valDim);
    require(loader, "blk.0.ssm_out.weight", valDim, H);
    require(loader, "blk.0.ssm_conv1d.weight");
    require(loader, "blk.0.ssm_norm.weight");
    require(loader, "blk.0.ssm_dt.bias");
    require(loader, "blk.0.ssm_a");
    require(loader, "blk.0.ssm_alpha.weight", H, cfg.ssmTimeStepRank);
    require(loader, "blk.0.ssm_beta.weight", H, cfg.ssmTimeStepRank);

    std::cout << "[smoke] layer 3 (full attention; fused Q+gate):" << std::endl;
    require(loader, "blk.3.attn_q.weight", H, nH * 2 * HD);
    require(loader, "blk.3.attn_k.weight", H, nKV * HD);
    require(loader, "blk.3.attn_v.weight", H, nKV * HD);
    require(loader, "blk.3.attn_output.weight", nH * HD, H);
    require(loader, "blk.3.attn_q_norm.weight");
    require(loader, "blk.3.attn_k_norm.weight");

    std::cout << "[smoke] last layer (" << (cfg.numLayers - 1) << "):" << std::endl;
    {
        const std::string p = "blk." + std::to_string(cfg.numLayers - 1) + ".";
        require(loader, p + "attn_norm.weight");
        require(loader, p + "post_attention_norm.weight");
    }

    // Numeric sanity: the zero-centered RMSNorm fold means attn_norm/output_norm
    // have mean ~= 1.0 (they were ~0 in HF).
    {
        Stats an = inspect(loader, "blk.0.attn_norm.weight");
        Stats on = inspect(loader, "output_norm.weight");
        double anMean = an.count ? an.sumAbs / an.count : 0.0;
        double onMean = on.count ? on.sumAbs / on.count : 0.0;
        std::cout << "[smoke] norm-fold check: attn_norm meanAbs=" << anMean
                  << " output_norm meanAbs=" << onMean << std::endl;
        if (!(anMean > 0.8 && anMean < 1.2)) {
            std::cerr << "  BAD attn_norm mean not ~1 (zero-centered fold?)" << std::endl;
            ++failures;
        }
    }

    if (failures == 0) {
        std::cout << "[smoke] ALL CHECKS PASSED" << std::endl;
        return 0;
    }
    std::cerr << "[smoke] " << failures << " check(s) FAILED" << std::endl;
    return 1;
}
