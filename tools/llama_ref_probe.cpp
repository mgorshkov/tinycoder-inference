// llama_ref_probe.cpp
//
// Reference probe against llama.cpp public API + gguf/ggml.
//
// Dumps the authoritative reference data for the exact prompt string used by
// ReferenceCompareTest.Qwen35LogitsVsReference so we can diff it against
// TinyCoder's qwen35 path:
//
//   1. Tokenization (parse_special=true, add_special both true/false)
//   2. Raw token embedding rows dequantized with ggml's reference dequantizer
//      (from token_embd.weight in the GGUF) — per-token norm, leading values,
//      and a cheap 64-bit fingerprint
//   3. llama_decode with embeddings=true on the same tokens:
//        - final hidden (after output_norm) embedding: norm + fingerprint
//        - final logits: argmax + top-5
//
// Build (against the llama.cpp tree):
//   g++ -O2 -std=c++17 tools/llama_ref_probe.cpp \
//       -I /home/mike/git/llama.cpp/include \
//       -I /home/mike/git/llama.cpp/ggml/include \
//       -L /home/mike/git/llama.cpp/build/bin \
//       -Wl,-rpath,/home/mike/git/llama.cpp/build/bin \
//       -lllama -lggml -lggml-base -lggml-cpu -o /tmp/llama_ref_probe
//
// Run:
//   /tmp/llama_ref_probe /data/models/qwen/Qwen3.8-27B-UD-Q4_K_M.gguf \
//       "What is the capital of France?<|im_end|>\n<|im_start|>assistant\n"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <map>

#include "llama.h"
#include "gguf.h"
#include "ggml.h"

// ---- tiny helpers -----------------------------------------------------------

static double vecNorm2(const float *v, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) s += (double) v[i] * v[i];
    return s;
}

static uint64_t fnv1a64(const float *v, int64_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (int64_t i = 0; i < n; ++i) {
        uint32_t bits; std::memcpy(&bits, &v[i], sizeof(bits));
        h ^= bits; h *= 1099511628211ULL;
    }
    return h;
}

static std::string pieceToString(const llama_vocab *vocab, llama_token tok, bool special) {
    char buf[256];
    const int len = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, special);
    return std::string(buf, len < 0 ? 0 : len);
}

// ---- raw GGUF embedding dump ------------------------------------------------

static void dumpEmbeddingRows(const char *fname, const std::vector<llama_token> &toks,
                              int64_t n_embd, int64_t n_vocab) {
    gguf_init_params params = { .no_alloc = true, .ctx = nullptr };
    gguf_context *gctx = gguf_init_from_file(fname, params);
    if (!gctx) {
        fprintf(stderr, "  [embed] gguf_init_from_file failed\n");
        return;
    }
    const int64_t tid = gguf_find_tensor(gctx, "token_embd.weight");
    if (tid < 0) {
        fprintf(stderr, "  [embed] token_embd.weight not found\n");
        gguf_free(gctx);
        return;
    }
    const char *tname   = gguf_get_tensor_name(gctx, tid);
    // gguf_get_tensor_ne() is not exported from the linkable object in this
    // ggml build, but token_embd.weight is always [n_embd, n_vocab]; both dims
    // are known from the llama model API. The tensor type is exported.
    const ggml_type type = gguf_get_tensor_type(gctx, tid);
    const size_t offset = gguf_get_tensor_offset(gctx, tid);
    const size_t daddr  = gguf_get_data_offset(gctx);

    fprintf(stderr, "  [embed] tensor=%s ne=[%lld,%lld] type=%s (%d) offset=%zu\n",
            tname, (long long) n_embd, (long long) n_vocab, ggml_type_name(type),
            (int) type, offset);

    FILE *f = fopen(fname, "rb");
    if (!f) { gguf_free(gctx); return; }

    const ggml_type_traits *tr = ggml_get_type_traits(type);
    const size_t rowSize = ggml_row_size(type, n_embd);
    std::vector<float> row(n_embd);
    std::vector<uint8_t> raw(rowSize);

    printf("  [embed] raw token embedding rows (reference ggml dequant):\n");
    for (size_t i = 0; i < toks.size(); ++i) {
        const int64_t t = toks[i];
        if (t < 0 || t >= n_vocab) {
            printf("    tok[%zu]=%d (OOB)\n", i, (int) t);
            continue;
        }
        if (fseeko(f, (off_t)(daddr + offset + (uint64_t) t * rowSize), SEEK_SET) != 0) {
            printf("    tok[%zu]=%d seek fail\n", i, (int) t);
            continue;
        }
        if (fread(raw.data(), 1, rowSize, f) != rowSize) {
            printf("    tok[%zu]=%d read fail\n", i, (int) t);
            continue;
        }
        tr->to_float(raw.data(), row.data(), n_embd);
        printf("    tok[%zu]=%d norm=%12.6f rms=%.6f first8=[% .7f % .7f % .7f % .7f % .7f % .7f % .7f % .7f] fnv=%016llx\n",
               i, (int) t, std::sqrt(vecNorm2(row.data(), n_embd)),
               std::sqrt(vecNorm2(row.data(), n_embd) / n_embd),
               row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7],
               (unsigned long long) fnv1a64(row.data(), n_embd));
    }
    fclose(f);
    gguf_free(gctx);
}

// ---- raw GGUF weight-block dump (ggml reference dequant) --------------------

// Dump the first/last blocks of row 0 and row 10 of selected qwen35 weight
// matrices. TinyCoder's mirror test dequantizes the same rows with its own
// GGMLDequantize and prints the same fingerprints — if any per-block fnv or
// scale differs, the weight dequant is the divergence.
struct WeightDumpSpec {
    const char *name;
    int64_t cols;   // ne[0] (GGUF shape[0])
    int64_t rows;   // ne[1] (GGUF shape[1])
};

static void dumpWeightBlocks(const char *fname) {
    static const WeightDumpSpec specs[] = {
            {"blk.0.attn_qkv.weight",    5120,  10240},
            {"blk.0.attn_gate.weight",   5120,   6144},
            {"blk.0.ssm_out.weight",     6144,   5120},
            {"blk.0.ffn_gate.weight",    5120,  17408},
            {"blk.0.ffn_up.weight",      5120,  17408},
            {"blk.0.ffn_down.weight",   17408,   5120},
            {"blk.0.ssm_alpha.weight",   5120,     48},
            {"blk.0.ssm_beta.weight",    5120,     48},
            // Qwen2 dense: the GGUF is rowwise with ne[0]=cols. The 1.5B
            // model has hidden=1536/12 heads/2 KV heads/inter=8960 while the
            // 7B has 3584/28/4/18944. Both are listed; the probe prints the
            // tensors that exist in the model.
            {"blk.0.attn_q.weight",      3584,   3584},
            {"blk.0.attn_k.weight",       512,   3584},
            {"blk.0.attn_v.weight",       512,   3584},
            {"blk.0.attn_output.weight", 3584,   3584},
            {"blk.0.ffn_gate.weight",   18944,   3584},
            {"blk.0.ffn_up.weight",     18944,   3584},
            {"blk.0.ffn_down.weight",   18944,   3584},
            {"token_embd.weight",        3584,  152064},
            {"output.weight",            3584,  152064},
            // 1.5B qwen2.5-coder (iq3_xxs-imat): hidden=1536, 12 Q heads,
            // 2 KV heads, intermediate=8960.
            {"blk.0.attn_q.weight",      1536,   1536},
            {"blk.0.attn_k.weight",       256,   1536},
            {"blk.0.attn_v.weight",       256,   1536},
            {"blk.0.attn_output.weight", 1536,   1536},
            {"blk.0.ffn_gate.weight",    8960,   1536},
            {"blk.0.ffn_up.weight",      8960,   1536},
            {"blk.0.ffn_down.weight",    8960,   1536},
            {"token_embd.weight",        1536,  151936},
            {"output.weight",            1536,  151936},
    };
    gguf_init_params params = { .no_alloc = true, .ctx = nullptr };
    gguf_context *gctx = gguf_init_from_file(fname, params);
    if (!gctx) {
        fprintf(stderr, "  [weights] gguf_init_from_file failed\n");
        return;
    }
    const size_t daddr = gguf_get_data_offset(gctx);
    FILE *f = fopen(fname, "rb");
    if (!f) {
        gguf_free(gctx);
        return;
    }
    printf("  [weights] raw weight-block fingerprints (ggml reference dequant):\n");
    for (const auto &spec : specs) {
        const int64_t tid = gguf_find_tensor(gctx, spec.name);
        if (tid < 0) {
            printf("    %s: <tensor not found>\n", spec.name);
            continue;
        }
        const ggml_type type = gguf_get_tensor_type(gctx, tid);
        const size_t offset = gguf_get_tensor_offset(gctx, tid);
        const ggml_type_traits *tr = ggml_get_type_traits(type);
        const size_t rowSize = ggml_row_size(type, spec.cols);
        const int64_t blocksPerRow = (spec.cols + tr->blck_size - 1) / tr->blck_size;
        std::vector<uint8_t> raw(rowSize);

        printf("    [%s] ne=[%lld,%lld] type=%s blocksPerRow=%lld rowwise\n",
               spec.name, (long long) spec.cols, (long long) spec.rows,
               ggml_type_name(type), (long long) blocksPerRow);

        const int64_t rowsToDump[2] = {0, 10};
        for (int64_t ri = 0; ri < 2; ++ri) {
            const int64_t row = rowsToDump[ri];
            if (row >= spec.rows) {
                printf("      row %lld: OOB\n", (long long) row);
                continue;
            }
            if (fseeko(f, (off_t)(daddr + offset + (uint64_t) row * rowSize), SEEK_SET) != 0 ||
                fread(raw.data(), 1, rowSize, f) != rowSize) {
                printf("      row %lld: read fail\n", (long long) row);
                continue;
            }
            std::vector<float> dq(spec.cols);
            tr->to_float(raw.data(), dq.data(), spec.cols);
            double s = 0.0;
            for (int64_t j = 0; j < spec.cols; ++j) s += (double) dq[j] * dq[j];
            printf("      row %lld: norm=%12.6f fnv=%016llx first8=[% .6f % .6f % .6f % .6f % .6f % .6f % .6f % .6f]\n",
                   (long long) row, std::sqrt(s),
                   (unsigned long long) fnv1a64(dq.data(), spec.cols),
                   dq[0], dq[1], dq[2], dq[3], dq[4], dq[5], dq[6], dq[7]);
            printf("        blockFnv: ");
            const int64_t blockN = tr->blck_size;
            std::vector<int64_t> blocks;
            for (int64_t b = 0; b < blocksPerRow && b < 4; ++b) blocks.push_back(b);
            if (blocksPerRow > 4) blocks.push_back(blocksPerRow - 1);
            for (int64_t b : blocks) {
                printf("b%lld=%016llx ", (long long) b,
                       (unsigned long long) fnv1a64(dq.data() + b * blockN, blockN));
            }
            printf("\n");
        }
        // Exhaustive full-tensor rolling FNV (every element, every row) so a
        // single bad row/bock anywhere is detected. Also count rows matching
        // the dequant of row 0's input pattern is unnecessary; the whole-tensor
        // hash is decisive.
        {
            uint64_t allRowFnv = 1469598103934665603ULL;
            std::vector<float> dqRow(spec.cols);
            double totalSq = 0.0;
            for (int64_t row = 0; row < spec.rows; ++row) {
                if (fseeko(f, (off_t)(daddr + offset + (uint64_t) row * rowSize), SEEK_SET) != 0 ||
                    fread(raw.data(), 1, rowSize, f) != rowSize) {
                    break;
                }
                tr->to_float(raw.data(), dqRow.data(), spec.cols);
                for (int64_t j = 0; j < spec.cols; ++j) {
                    uint32_t bits; std::memcpy(&bits, &dqRow[j], sizeof(bits));
                    allRowFnv ^= bits; allRowFnv *= 1099511628211ULL;
                    totalSq += (double) dqRow[j] * dqRow[j];
                }
            }
            printf("      ALL: rowsTotalFnv=%016llx totalNorm=%12.6f\n",
                   (unsigned long long) allRowFnv, std::sqrt(totalSq));
        }
    }
    fclose(f);
    gguf_free(gctx);
}

// ---- main -------------------------------------------------------------------

static bool readTokensFile(const char *path, std::vector<llama_token> &out) {
    FILE *f = fopen(path, "r");
    if (!f) return false;
    int v;
    while (fscanf(f, "%d", &v) == 1) {
        if (v >= 0) out.push_back((llama_token) v);
    }
    fclose(f);
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> \"<prompt>\" | --tokens <ids-file> | --weights-only\n", argv[0]);
        return 1;
    }
    const char *modelPath = argv[1];
    if (argc >= 3 && strcmp(argv[2], "--weights-only") == 0) {
        // Fast path: dump only the raw weight-block fingerprints — no model load.
        dumpWeightBlocks(modelPath);
        return 0;
    }
    const bool useTokensFile = strcmp(argv[2], "--tokens") == 0;
    const std::string prompt = useTokensFile ? "" : argv[2];
    const char *tokensFilePath = useTokensFile && argc >= 4 ? argv[3] : nullptr;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model *model = llama_model_load_from_file(modelPath, mparams);
    if (!model) {
        fprintf(stderr, "model load failed\n");
        return 2;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);
    const int64_t n_vocab = llama_vocab_n_tokens(vocab);
    const int64_t n_embd  = llama_model_n_embd(model);
    const int64_t n_layer = llama_model_n_layer(model);
    char desc[256]; llama_model_desc(model, desc, sizeof(desc));
    printf("== llama.cpp reference probe ==\n");
    printf("  model: %s\n  n_vocab=%lld n_embd=%lld n_layer=%lld\n",
           desc, (long long) n_vocab, (long long) n_embd, (long long) n_layer);

    // ---- tokenization ----
    int nTok = 0;
    std::vector<llama_token> toks;

    if (useTokensFile) {
        if (!readTokensFile(tokensFilePath, toks)) {
            fprintf(stderr, "failed to read tokens file\n");
            llama_free_model(model);
            llama_backend_free();
            return 2;
        }
        nTok = (int) toks.size();
        printf("  [tokens-file] %d ids loaded\n", nTok);
        for (int i = 0; i < nTok; ++i) {
            std::string p = pieceToString(vocab, toks[i], true);
            printf("    %d(%s)\n", (int) toks[i], p.c_str());
        }
    } else {
        toks.resize((size_t) (prompt.size() + 64));
        for (int pass = 0; pass < 2; ++pass) {
            const bool add_special = (pass == 0);
            int n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                                   toks.data(), (int32_t) toks.size(), add_special, true);
            if (n < 0) toks.resize(-n);
            printf("  tokens(add_special=%s): (%d) ", add_special ? "true" : "false", n);
            for (int i = 0; i < n; ++i) {
                std::string p = pieceToString(vocab, toks[i], true);
                printf("%d(%s) ", (int) toks[i], p.c_str());
            }
            printf("\n");
        }

        // Tokenize deterministically (parse_special=true, add_special=false by
        // default for qwen3.8-like models without BOS).
        nTok = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                              toks.data(), (int32_t) toks.size(), false, true);
        if (nTok < 0) { toks.resize((size_t) -nTok); nTok = -nTok; }
        toks.resize(nTok);

        printf("  token count used for decode: %d\n", nTok);
        printf("  last three: %d %d %d\n",
               nTok >= 3 ? (int) toks[nTok-3] : -1,
               nTok >= 2 ? (int) toks[nTok-2] : -1,
               nTok >= 1 ? (int) toks[nTok-1] : -1);
    }

    // Cross-check the "assistant" tokenization that TinyCoder produces:
    // TinyCoder reports 77091 for "assistant" while llama.cpp emits 74455.
    {
        const llama_token ids[] = { 74455, 77091, 77090, 77092 };
        for (int32_t t : ids) {
            std::string p1 = pieceToString(vocab, t, false);
            std::string p2 = pieceToString(vocab, t, true);
            printf("  [vocab] id=%d piece=\"%s\" special=\"%s\"\n", (int) t,
                   p1.c_str(), p2.c_str());
        }
    }

    dumpEmbeddingRows(modelPath, toks, n_embd, n_vocab);

    dumpWeightBlocks(modelPath);

    // ---- decode ----
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = 4096;
    cparams.n_batch   = nTok;
    cparams.n_ubatch  = nTok;
    cparams.n_threads = 8;
    cparams.embeddings = true;

    llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "context init failed\n");
        llama_free_model(model);
        llama_backend_free();
        return 3;
    }

    llama_batch batch = llama_batch_init(nTok, 0, 1);
    for (int i = 0; i < nTok; ++i) {
        batch.token[i] = toks[i];
        batch.pos[i]   = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == nTok - 1);
    }
    batch.n_tokens = nTok;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 4;
    }

    // final (post output_norm) embedding of the last token
    float *emb = llama_get_embeddings_ith(ctx, nTok - 1);
    if (emb) {
        printf("  [final] post-output_norm embedding (last token):\n");
        printf("    norm=%12.6f rms=%.6f first8=[% .7f % .7f % .7f % .7f % .7f % .7f % .7f % .7f] fnv=%016llx\n",
               std::sqrt(vecNorm2(emb, n_embd)), std::sqrt(vecNorm2(emb, n_embd) / n_embd),
               emb[0], emb[1], emb[2], emb[3], emb[4], emb[5], emb[6], emb[7],
               (unsigned long long) fnv1a64(emb, n_embd));
    } else {
        printf("  [final] embeddings unavailable (pooling/embeddings off?)\n");
    }

    float *logits = llama_get_logits_ith(ctx, nTok - 1);
    if (logits) {
        std::vector<std::pair<float, int32_t>> scored;
        scored.reserve(n_vocab);
        for (int64_t i = 0; i < n_vocab; ++i) scored.emplace_back(logits[i], (int32_t) i);
        std::partial_sort(scored.begin(), scored.begin() + 5, scored.end(),
                          [](const auto &a, const auto &b) { return a.first > b.first; });
        printf("  [final] logits argmax: id=%d logit=%.6f\n", scored[0].second, scored[0].first);
        for (int i = 0; i < 5; ++i) {
            const auto &p = scored[i];
            std::string txt = pieceToString(vocab, p.second, false);
            printf("    top%d: id=%d logit=%12.6f text=\"%s\"\n", i + 1, p.second,
                   p.first, txt.c_str());
        }
    } else {
        printf("  [final] logits unavailable\n");
    }

    // ---- Control: token-by-token (autoregressive) decode of the same ids ----
    // The batched decode above ran the CHUNKED fused GDN; a fresh context
    // decoding the same tokens one at a time exercises the AUTOREGRESSIVE
    // fused GDN. TinyCoder's tests run token-by-token, so this is the
    // apples-to-apples reference. Both llama paths must agree; if they do
    // not, the recurrence implementation differs between chunked/AR.
    {
        // Per-layer capture via ggml_backend_sched eval callback. Tensor names
        // are set by llama's graph callback: "l_out-<il>" for each layer's
        // post-FFN-residual hidden, "result_norm" for the final output_norm.
        struct LayerCtx {
            int64_t n_embd = 0;
            int64_t n_layers = 0;
            std::vector<std::vector<float>> perLayer; // [il][n_embd]
            std::vector<float> resultNorm;
            int captures = 0;
            // Layer-0 recurrent intermediate capture (single token):
            // keyed by llama graph cb() tensor name suffix; stores the LAST
            // evaluated value (n_tokens=1 so there is exactly one).
            std::map<std::string, std::vector<float>> intm;
            void capture(const char *name, const float *d, int64_t n) {
                intm[name] = std::vector<float>(d, d + n);
            }
        };
        static LayerCtx g_layerCtx; // reused across decodes
        g_layerCtx.perLayer.assign(n_layer, {});
        g_layerCtx.resultNorm.clear();
        g_layerCtx.captures = 0;
        g_layerCtx.n_embd = n_embd;
        g_layerCtx.n_layers = n_layer;

        auto cbEval = [](ggml_tensor * t, bool ask, void * user_data) -> bool {
            if (ask || !t || t->type != GGML_TYPE_F32) return true;
            LayerCtx &lc = *static_cast<LayerCtx *>(user_data);
            lc.captures++;
            // NOTE: no early-return cutoff here. The intm map is overwritten on
            // every eval so it always holds the LAST processed token's values,
            // which is what we fingerprint. A cutoff would skip the final
            // token's intermediates (cumulative evals exceed 20000 across the
            // whole AR decode).
            const char * nm = t->name;
            if (nm[0] == '\0') return true;
            // "l_out-<il>"
            if (strncmp(nm, "l_out-", 6) == 0) {
                int il = atoi(nm + 6);
                if (il >= 0 && il < (int) lc.n_layers &&
                    (int64_t) ggml_nbytes(t) == lc.n_embd * 4) {
                    const float *d = (const float *) t->data;
                    lc.perLayer[il].assign(d, d + lc.n_embd);
                }
            } else if (strcmp(nm, "result_norm") == 0 &&
                       (int64_t) ggml_nbytes(t) == lc.n_embd * 4) {
                const float *d = (const float *) t->data;
                lc.resultNorm.assign(d, d + lc.n_embd);
            }
            // Layer-0 recurrent intermediate bisection. The map is overwritten
            // on every eval, so it always holds the LAST token's values. Names
            // come from llama_model_qwen35::graph cb() calls.
            {
                const int64_t nb = (int64_t) ggml_nbytes(t);
                // Match the layer-0 recurrent intermediates by exact name:
                static const char *want[] = {
                    "attn_norm-0", "linear_attn_qkv_mixed-0", "z-0",
                    "beta_sigmoid-0", "a_softplus-0", "gate-0",
                    "conv_output_raw-0", "conv_output_silu-0",
                    "q_conv_predelta-0", "k_conv_predelta-0", "v_conv_predelta-0",
                    "attn_output-0", "final_output-0", "linear_attn_out-0",
                    "attn_residual-0", "attn_post_norm-0", "ffn_out-0",
                    "post_ffn-0",
                    // Layer-3 full-attention intermediates (qwen35.cpp
                    // build_layer_attn names). Ambiguity note: "Kcur-3"/"Kcur"
                    // is emitted twice (pre-norm 2D and post-rope 3D) so it is
                    // NOT captured -- the LAST eval of a name wins, and for Kcur
                    // the last is the 3D post-rope whereas the pre-norm 2D is
                    // what we need; "Kcur_normed-3" is unambiguous. "Qcur-3" is
                    // emitted once post-rope (cb), "Vcur-3" once (reshaped).
                    "attn_norm-3", "Qcur_full-3", "Qcur_normed-3",
                    "Kcur_normed-3", "Qcur-3", "Vcur-3", "gate_reshaped-3",
                    "attn_pregate-3", "gate_sigmoid-3", "attn_gated-3",
                    "attn_output-3", "attn_residual-3", "attn_post_norm-3",
                    "ffn_out-3", "post_ffn-3",
                    // Qwen2 layer-0/1 intermediates (llama-graph builds via
                    // build_qkv/build_ffn with cb() labels). For Kcur the last
                    // eval wins = post-RoPE 3D (the pre-RoPE Kcur is emitted by
                    // build_qkv before ggml_rope_ext, so Kcur-0 is post-RoPE).
                    "attn_norm-0", "Qcur-0", "Kcur-0", "Vcur-0",
                    "ffn_inp-0", "ffn_norm-0", "ffn_out-0",
                    "attn_norm-1", "Qcur-1", "Kcur-1", "Vcur-1",
                };
                for (const char *w : want) {
                    if (strcmp(nm, w) == 0 && nb > 0) {
                        const int64_t n = nb / 4;
                        const float *d = (const float *) t->data;
                        lc.capture(w, d, n);
                        break;
                    }
                }
            }
            return true;
        };
        llama_context_params cparams2 = llama_context_default_params();
        cparams2.n_ctx     = 4096;
        cparams2.n_batch   = 1;
        cparams2.n_ubatch  = 1;
        cparams2.n_threads = 8;
        cparams2.embeddings = true;
        cparams2.cb_eval = cbEval;
        cparams2.cb_eval_user_data = &g_layerCtx;
        llama_context *ctx2 = llama_init_from_model(model, cparams2);
        if (ctx2) {
            for (int i = 0; i < nTok; ++i) {
                llama_batch b1 = llama_batch_init(1, 0, 1);
                b1.token[0] = toks[i];
                b1.pos[0]   = i;
                b1.n_seq_id[0] = 1;
                b1.seq_id[0][0] = 0;
                b1.logits[0] = true;
                b1.n_tokens = 1;
                if (llama_decode(ctx2, b1) != 0) {
                    fprintf(stderr, "  [ar] llama_decode token %d failed\n", i);
                    llama_batch_free(b1);
                    break;
                }
                llama_batch_free(b1);
            }
            // Per-layer hidden norms (post-FFN-residual, pre-final-norm) for
            // the LAST decoded token. These are directly comparable to
            // TinyCoder's debugQwen35PerLayer() states.
            printf("  [ar] per-layer post-FFN hidden norms (last token):\n");
            for (int il = 0; il < (int) n_layer; ++il) {
                const auto &v = g_layerCtx.perLayer[il];
                if (v.empty() || (int64_t) v.size() != n_embd) {
                    printf("    layer %d: <no capture>\n", il);
                    continue;
                }
                double s = 0.0;
                for (int64_t j = 0; j < n_embd; ++j) s += (double) v[j] * v[j];
                printf("    layer %3d: norm=%12.6f fnv=%016llx first8=[% .6f % .6f % .6f % .6f % .6f % .6f % .6f % .6f]\n",
                       il, std::sqrt(s),
                       (unsigned long long) fnv1a64(v.data(), n_embd),
                       v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            }
            if (!g_layerCtx.resultNorm.empty()) {
                double s = 0.0;
                for (int64_t j = 0; j < n_embd; ++j) s += (double) g_layerCtx.resultNorm[j] * g_layerCtx.resultNorm[j];
                printf("  [ar] result_norm (post final RMSNorm) norm=%12.6f fnv=%016llx\n",
                       std::sqrt(s),
                       (unsigned long long) fnv1a64(g_layerCtx.resultNorm.data(), n_embd));
            }
            float *emb = llama_get_embeddings_ith(ctx2, 0);
            if (emb) {
                printf("  [ar] post-output_norm embedding (token-by-token):\n");
                printf("    norm=%12.6f rms=%.6f first8=[% .7f % .7f % .7f % .7f % .7f % .7f % .7f % .7f] fnv=%016llx\n",
                       std::sqrt(vecNorm2(emb, n_embd)), std::sqrt(vecNorm2(emb, n_embd) / n_embd),
                       emb[0], emb[1], emb[2], emb[3], emb[4], emb[5], emb[6], emb[7],
                       (unsigned long long) fnv1a64(emb, n_embd));
            } else {
                printf("  [ar] embeddings unavailable\n");
            }

            // Layer-0 recurrent intermediate fingerprints (single-token bisect).
            printf("  [ar] layer-0 recurrent intermediates (last token):\n");
            for (const auto &kv : g_layerCtx.intm) {
                const std::string &name = kv.first;
                const auto &v = kv.second;
                const int64_t n = (int64_t) v.size();
                if (n <= 0) continue;
                double s = 0.0;
                for (int64_t j = 0; j < n; ++j) s += (double) v[j] * v[j];
                printf("    %-26s n=%-6lld norm=%12.6f fnv=%016llx first4=[% .6f % .6f % .6f % .6f]\n",
                       name.c_str(), (long long) n, std::sqrt(s),
                       (unsigned long long) fnv1a64(v.data(), n),
                       v[0], n > 1 ? v[1] : 0.0f, n > 2 ? v[2] : 0.0f, n > 3 ? v[3] : 0.0f);
            }
            llama_free(ctx2);
        } else {
            fprintf(stderr, "  [ar] context init failed\n");
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    return 0;
}
