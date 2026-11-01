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

/**
 * TinyCoder EXL3 codec / loader unit tests.
 *
 * These validate the CPU reference codec against INDEPENDENT reimplementations
 * (a from-scratch bit-packer, a dense Sylvester-Hadamard matrix, a literal
 * transform-order transcription) rather than calling the code under test twice.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <vector>

#include "EXL3Codec.hpp"
#include "EXL3Loader.hpp"
#include "SafetensorsReader.hpp"
#include "Tokenizer.hpp"

using namespace tinycoder;

namespace {

    // ---- Independent K-bit tile packer (LSB-first, 16-bit words) -------------
    std::vector<uint16_t> packTileIndependent(const std::vector<uint16_t> &codes, uint32_t K) {
        std::vector<uint16_t> words(16 * K, 0);
        for (uint32_t j = 0; j < 256; ++j) {
            const uint32_t pos = j * K;
            for (uint32_t b = 0; b < K; ++b) {
                const uint32_t bit = (codes[j] >> b) & 1u;
                const uint32_t p = pos + b;
                words[p >> 4] |= static_cast<uint16_t>(bit << (p & 15));
            }
        }
        return words;
    }

    std::vector<uint16_t> packTileFracIndependent(const std::vector<uint16_t> &codes,
                                                  uint32_t KA) {
        std::vector<uint16_t> words(16 * KA + 8, 0);
        uint32_t pos = 0;
        for (uint32_t j = 0; j < 256; ++j) {
            const uint32_t width = (j % 2 == 0) ? (KA + 1) : KA;
            for (uint32_t b = 0; b < width; ++b) {
                const uint32_t bit = (codes[j] >> b) & 1u;
                const uint32_t p = pos + b;
                words[p >> 4] |= static_cast<uint16_t>(bit << (p & 15));
            }
            pos += width;
        }
        return words;
    }

    // ---- Dense normalized Sylvester-Hadamard (natural order) -----------------
    // H[i][j] = (-1)^popcount(i & j) / sqrt(128)
    void buildHadamard128(std::vector<float> &H) {
        const uint32_t n = 128;
        H.assign(n * n, 0.0f);
        const float scale = 1.0f / std::sqrt(static_cast<float>(n));
        for (uint32_t i = 0; i < n; ++i) {
            for (uint32_t j = 0; j < n; ++j) {
                const bool odd = (__builtin_popcount(i & j) & 1u) != 0;
                H[i * n + j] = odd ? -scale : scale;
            }
        }
    }

    std::vector<uint16_t> randomCodes(std::mt19937 &rng, uint32_t K) {
        std::vector<uint16_t> c(256);
        const uint32_t mask = (1u << K) - 1u;
        for (auto &v: c) {
            v = static_cast<uint16_t>(rng() & mask);
        }
        return c;
    }

    // Build the row-major [k][n] "rotated basis" weight from a packed trellis,
    // reusing decodeTile (the shared primitive under test in other cases).
    std::vector<float> decodedRotatedBasis(const std::vector<uint16_t> &packed, uint32_t K,
                                           int codebook, uint32_t k, uint32_t n) {
        std::vector<float> w(static_cast<size_t>(k) * n, 0.0f);
        const uint32_t tilesK = k / 16, tilesN = n / 16;
        for (uint32_t ki = 0; ki < tilesK; ++ki) {
            for (uint32_t ni = 0; ni < tilesN; ++ni) {
                const uint16_t *tile = packed.data() + (static_cast<size_t>(ki) * tilesN + ni) * (16 * K);
                float tileRM[256];
                EXL3Codec::decodeTileNative(tile, K, codebook, tileRM);
                for (uint32_t r = 0; r < 16; ++r) {
                    std::memcpy(&w[(static_cast<size_t>(ki) * 16 + r) * n + ni * 16],
                                tileRM + r * 16, 16 * sizeof(float));
                }
            }
        }
        return w;
    }

    // Reference transform, transcribed literally from LinearEXL3.get_weight_tensor:
    // left H128 -> *suh -> right H128 -> *svh.
    std::vector<float> referenceTransform(const std::vector<float> &wHat,
                                          const std::vector<float> &H,
                                          const std::vector<float> *suh,
                                          const std::vector<float> *svh,
                                          uint32_t k, uint32_t n) {
        std::vector<float> w = wHat;
        // Left Hadamard over 128-wide row blocks (per column). Uses a temp
        // column buffer so the transform reads only the pre-transform values
        // (the codec likewise stages into a temp before the matrix multiply).
        for (uint32_t kb = 0; kb < k; kb += 128) {
            for (uint32_t c = 0; c < n; ++c) {
                float col[128];
                for (uint32_t j = 0; j < 128; ++j) {
                    col[j] = w[(kb + j) * n + c];
                }
                for (uint32_t i = 0; i < 128; ++i) {
                    float acc = 0.0f;
                    for (uint32_t j = 0; j < 128; ++j) {
                        acc += H[i * 128 + j] * col[j];
                    }
                    w[(kb + i) * n + c] = acc;
                }
            }
        }
        if (suh) {
            for (uint32_t r = 0; r < k; ++r) {
                for (uint32_t c = 0; c < n; ++c) {
                    w[r * n + c] *= (*suh)[r];
                }
            }
        }
        // Right Hadamard over 128-wide column blocks (per row). Temp row buffer
        // for the same reason as the left transform above.
        for (uint32_t r = 0; r < k; ++r) {
            for (uint32_t nb = 0; nb < n; nb += 128) {
                float row[128];
                for (uint32_t j = 0; j < 128; ++j) {
                    row[j] = w[r * n + nb + j];
                }
                for (uint32_t i = 0; i < 128; ++i) {
                    float acc = 0.0f;
                    for (uint32_t j = 0; j < 128; ++j) {
                        acc += H[i * 128 + j] * row[j];
                    }
                    w[r * n + nb + i] = acc;
                }
            }
        }
        if (svh) {
            for (uint32_t r = 0; r < k; ++r) {
                for (uint32_t c = 0; c < n; ++c) {
                    w[r * n + c] *= (*svh)[c];
                }
            }
        }
        return w;
    }

    // ---- Minimal safetensors writer (for the reader test) --------------------
    void writeSafetensors(const std::string &path) {
        // Two tensors: "a.weight" F32 [2,3] (24 bytes), "b.weight" F16 [4] (8 bytes).
        std::string header =
                "{"
                "\"a.weight\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]},"
                "\"b.weight\":{\"dtype\":\"F16\",\"shape\":[4],\"data_offsets\":[24,32]},"
                "\"c.weight\":{\"dtype\":\"I16\",\"shape\":[3],\"data_offsets\":[32,38]}"
                "}";
        uint64_t headerLen = header.size();
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(&headerLen), 8);
        f.write(header.data(), static_cast<std::streamsize>(header.size()));
        const float a[6] = {1, 2, 3, 4, 5, 6};
        f.write(reinterpret_cast<const char *>(a), sizeof(a));
        const uint16_t b[4] = {0x3C00, 0x4000, 0xC000, 0x0000};// 1,2,-2,0
        f.write(reinterpret_cast<const char *>(b), sizeof(b));
        const uint16_t c[3] = {11, 22, 33};
        f.write(reinterpret_cast<const char *>(c), sizeof(c));
    }

}// namespace

// ---------------------------------------------------------------------------
// Tensor-core permutation: bijection and mutual inverse.
// ---------------------------------------------------------------------------
TEST(EXL3CodecTest, TensorCorePermIsBijection) {
    const uint8_t *perm = EXL3Codec::tensorCorePerm();
    const uint8_t *inv = EXL3Codec::tensorCorePermInv();
    std::array<bool, 256> seen{};
    for (int i = 0; i < 256; ++i) {
        const uint8_t v = perm[i];
        EXPECT_FALSE(seen[v]) << "duplicate perm target " << int(v);
        seen[v] = true;
        EXPECT_EQ(inv[v], i);
    }
}

// ---------------------------------------------------------------------------
// unpackTile matches an independent LSB-first bit-packer (integer K).
// ---------------------------------------------------------------------------
TEST(EXL3CodecTest, UnpackTileMatchesIndependentPacker) {
    std::mt19937 rng(1234);
    for (uint32_t K = 1; K <= 8; ++K) {
        const auto codes = randomCodes(rng, K);
        const auto words = packTileIndependent(codes, K);
        uint16_t out[256];
        EXL3Codec::unpackTile(words.data(), K, out);
        for (int j = 0; j < 256; ++j) {
            ASSERT_EQ(out[j], codes[j]) << "K=" << K << " j=" << j;
        }
    }
}

// ---------------------------------------------------------------------------
// unpackTileFrac matches an independent alternating-width packer (mul1 only).
// ---------------------------------------------------------------------------
TEST(EXL3CodecTest, UnpackTileFracMatchesIndependentPacker) {
    std::mt19937 rng(99);
    for (uint32_t KA = 1; KA <= 7; ++KA) {
        std::vector<uint16_t> codes(256);
        for (uint32_t j = 0; j < 256; ++j) {
            const uint32_t width = (j % 2 == 0) ? (KA + 1) : KA;
            codes[j] = static_cast<uint16_t>(rng() & ((1u << width) - 1u));
        }
        const auto words = packTileFracIndependent(codes, KA);
        uint16_t out[256];
        EXL3Codec::unpackTileFrac(words.data(), KA, out);
        for (int j = 0; j < 256; ++j) {
            ASSERT_EQ(out[j], codes[j]) << "KA=" << KA << " j=" << j;
        }
    }
}

// ---------------------------------------------------------------------------
// expandPackedSigns: bit i -> element i, set bit = -1.
// ---------------------------------------------------------------------------
TEST(EXL3CodecTest, ExpandPackedSignsPolarityAndOrder) {
    // group 0: bits 0000...0001 -> element 0 = -1, rest +1
    // group 1: bits 1000...0000 -> element 15 = -1, rest +1
    std::array<uint16_t, 2> packed{0x0001, 0x8000};
    std::array<float, 32> out{};
    EXL3Codec::expandPackedSigns(packed.data(), 2, out.data());
    EXPECT_FLOAT_EQ(out[0], -1.0f);
    for (int i = 1; i < 16; ++i) {
        EXPECT_FLOAT_EQ(out[i], 1.0f) << i;
    }
    for (int i = 16; i < 31; ++i) {
        EXPECT_FLOAT_EQ(out[i], 1.0f) << i;
    }
    EXPECT_FLOAT_EQ(out[31], -1.0f);
    EXPECT_EQ(EXL3Codec::packedSignWords(32), 2u);
    EXPECT_EQ(EXL3Codec::packedSignWords(17), 2u);
    EXPECT_EQ(EXL3Codec::packedSignWords(16), 1u);
}

// ---------------------------------------------------------------------------
// hadamard128 equals the dense normalized Sylvester-Hadamard matrix.
// ---------------------------------------------------------------------------
TEST(EXL3CodecTest, Hadamard128MatchesSylvesterMatrix) {
    std::vector<float> H;
    buildHadamard128(H);
    std::mt19937 rng(7);
    std::vector<float> v(128);
    for (auto &x: v) {
        x = static_cast<float>(rng() % 1000) / 100.0f - 5.0f;
    }
    std::vector<float> expect(128, 0.0f);
    for (int i = 0; i < 128; ++i) {
        for (int j = 0; j < 128; ++j) {
            expect[i] += H[i * 128 + j] * v[j];
        }
    }
    std::vector<float> got = v;
    EXL3Codec::hadamard128(got.data());
    for (int i = 0; i < 128; ++i) {
        EXPECT_NEAR(got[i], expect[i], 1e-4f) << i;
    }
    // Twice = identity (orthonormal).
    EXL3Codec::hadamard128(got.data());
    for (int i = 0; i < 128; ++i) {
        EXPECT_NEAR(got[i], v[i], 1e-4f) << i;
    }
}

// ---------------------------------------------------------------------------
// reconstructMatrix applies decode -> left H -> *suh -> right H -> *svh in the
// documented order, matching a literal transcription with an independent H.
// ---------------------------------------------------------------------------
TEST(EXL3CodecTest, ReconstructMatrixTransformOrder) {
    const uint32_t k = 256, n = 128, K = 3;
    std::mt19937 rng(2024);
    const auto codes = randomCodes(rng, K);
    const auto packed = packTileIndependent(codes, K);
    // Tile grid: [k/16][n/16][16*K] — replicate the single random tile.
    const uint32_t tilesK = k / 16, tilesN = n / 16;
    std::vector<uint16_t> trellis(static_cast<size_t>(tilesK) * tilesN * 16 * K);
    for (uint32_t t = 0; t < tilesK * tilesN; ++t) {
        std::memcpy(trellis.data() + static_cast<size_t>(t) * 16 * K, packed.data(),
                    16 * K * sizeof(uint16_t));
    }

    const auto wHat = decodedRotatedBasis(trellis, K, EXL3Codec::CB_3INST, k, n);

    std::vector<float> H;
    buildHadamard128(H);

    // Random sign vectors.
    std::vector<float> suh(k), svh(n);
    for (auto &s: suh) {
        s = (rng() & 1) ? 1.0f : -1.0f;
    }
    for (auto &s: svh) {
        s = (rng() & 1) ? 1.0f : -1.0f;
    }

    std::vector<float> expect = referenceTransform(wHat, H, &suh, &svh, k, n);

    std::vector<float> got(static_cast<size_t>(k) * n);
    EXL3Codec::reconstructMatrix(trellis.data(), K, EXL3Codec::CB_3INST, suh.data(),
                                 svh.data(), k, n, got.data());

    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], expect[i], 2e-4f) << i;
    }
}

// ---------------------------------------------------------------------------
// reconstructMatrix with identity signs is a pure two-sided Hadamard.
// ---------------------------------------------------------------------------
TEST(EXL3CodecTest, ReconstructMatrixIdentitySigns) {
    const uint32_t k = 128, n = 128, K = 2;
    std::mt19937 rng(55);
    const auto codes = randomCodes(rng, K);
    const auto packed = packTileIndependent(codes, K);
    std::vector<uint16_t> trellis(static_cast<size_t>(k / 16) * (n / 16) * 16 * K);
    for (uint32_t t = 0; t < (k / 16) * (n / 16); ++t) {
        std::memcpy(trellis.data() + static_cast<size_t>(t) * 16 * K, packed.data(),
                    16 * K * sizeof(uint16_t));
    }
    const auto wHat = decodedRotatedBasis(trellis, K, EXL3Codec::CB_3INST, k, n);
    std::vector<float> H;
    buildHadamard128(H);
    const auto expect = referenceTransform(wHat, H, nullptr, nullptr, k, n);
    std::vector<float> got(static_cast<size_t>(k) * n);
    EXL3Codec::reconstructMatrix(trellis.data(), K, EXL3Codec::CB_3INST, nullptr, nullptr, k,
                                 n, got.data());
    for (size_t i = 0; i < got.size(); ++i) {
        EXPECT_NEAR(got[i], expect[i], 2e-4f) << i;
    }
}

// ---------------------------------------------------------------------------
// JsonValue parser: basic document.
// ---------------------------------------------------------------------------
TEST(EXL3SafetensorsTest, JsonParserBasics) {
    JsonValue root;
    std::string err;
    ASSERT_TRUE(JsonValue::parse(
            "{\"name\":\"qwen\",\"n\":42,\"arr\":[1,2,3],\"ok\":true,\"nil\":null}", root, &err))
            << err;
    ASSERT_EQ(root.type, JsonValue::Type::Object);
    ASSERT_NE(root.find("name"), nullptr);
    EXPECT_EQ(root.find("name")->asString(), "qwen");
    ASSERT_NE(root.find("n"), nullptr);
    EXPECT_EQ(root.find("n")->asU64(), 42u);
    const JsonValue *arr = root.find("arr");
    ASSERT_NE(arr, nullptr);
    ASSERT_EQ(arr->type, JsonValue::Type::Array);
    ASSERT_EQ(arr->array.size(), 3u);
    EXPECT_EQ(arr->at(2)->asU64(), 3u);
    EXPECT_TRUE(root.find("ok")->boolean);
    EXPECT_EQ(root.find("nil")->type, JsonValue::Type::Null);
}

// ---------------------------------------------------------------------------
// SafetensorsReader: header parse + tensor payload access.
// ---------------------------------------------------------------------------
TEST(EXL3SafetensorsTest, ReadShard) {
    const std::string path = "/tmp/tinycoder_exl3_test.safetensors";
    writeSafetensors(path);

    SafetensorsReader reader;
    ASSERT_TRUE(reader.open(path));

    const auto *a = reader.find("a.weight");
    ASSERT_NE(a, nullptr);
    EXPECT_EQ(a->dtype, SafetensorsReader::DType::F32);
    ASSERT_EQ(a->shape.size(), 2u);
    EXPECT_EQ(a->shape[0], 2u);
    EXPECT_EQ(a->shape[1], 3u);
    EXPECT_EQ(SafetensorsReader::numel(*a), 6u);
    const float *adata = reinterpret_cast<const float *>(reader.data("a.weight"));
    ASSERT_NE(adata, nullptr);
    EXPECT_FLOAT_EQ(adata[0], 1.0f);
    EXPECT_FLOAT_EQ(adata[5], 6.0f);

    const auto *b = reader.find("b.weight");
    ASSERT_NE(b, nullptr);
    EXPECT_EQ(b->dtype, SafetensorsReader::DType::F16);

    const auto *c = reader.find("c.weight");
    ASSERT_NE(c, nullptr);
    EXPECT_EQ(c->dtype, SafetensorsReader::DType::I16);

    EXPECT_EQ(reader.find("missing"), nullptr);
    EXPECT_EQ(reader.data("missing"), nullptr);

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// EXL3Loader::isExl3Path sniffing.
// ---------------------------------------------------------------------------
TEST(EXL3LoaderTest, PathSniffing) {
    namespace fs = std::filesystem;
    const std::string dir = "/tmp/tinycoder_exl3_sniff";
    fs::remove_all(dir);
    fs::create_directories(dir);

    // Empty dir: no safetensors, no gguf -> not EXL3.
    EXPECT_FALSE(EXL3Loader::isExl3Path(dir));

    // Directory with a safetensors shard only -> EXL3.
    {
        std::ofstream(dir + "/model.safetensors").put('\0');
        EXPECT_TRUE(EXL3Loader::isExl3Path(dir));
    }
    // Add a .gguf next to it -> prefer GGUF.
    {
        std::ofstream(dir + "/model.gguf").put('\0');
        EXPECT_FALSE(EXL3Loader::isExl3Path(dir));
    }

    fs::remove_all(dir);

    // Explicit .safetensors file path -> EXL3.
    EXPECT_TRUE(EXL3Loader::isExl3Path("/some/where/model.safetensors"));
    EXPECT_FALSE(EXL3Loader::isExl3Path("/some/where/model.gguf"));
}

// ---------------------------------------------------------------------------
// HF tokenizer.json loading (the EXL3 checkpoint tokenizer).
// Self-contained: builds a tiny ByteLevel-BPE tokenizer.json on disk and checks
// vocab / merges / special-token resolution / encode-decode round-trip.
// ---------------------------------------------------------------------------
TEST(EXL3TokenizerTest, LoadsHuggingFaceTokenizerJson) {
    namespace fs = std::filesystem;
    const std::string dir = "/tmp/tinycoder_exl3_tok";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const std::string path = dir + "/tokenizer.json";

    // Miniature vocab: the 256 byte tokens via bytes_to_unicode, plus a couple of
    // merges and the Qwen-style special tokens at the end.
    std::string json = "{\"added_tokens\":["
                       "{\"id\":258,\"content\":\"<|endoftext|>\",\"special\":true},"
                       "{\"id\":259,\"content\":\"<|im_start|>\",\"special\":true},"
                       "{\"id\":260,\"content\":\"<|im_end|>\",\"special\":true}],"
                       "\"model\":{\"type\":\"BPE\",\"vocab\":{";
    for (int b = 0; b < 256; ++b) {
        // bytes_to_unicode()
        const bool good = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                          (b >= 174 && b <= 255);
        int cp = b;
        if (!good) {
            int n = 0;
            for (int x = 0; x < b; ++x) {
                const bool gx = (x >= 33 && x <= 126) || (x >= 161 && x <= 172) ||
                                (x >= 174 && x <= 255);
                if (!gx) ++n;
            }
            cp = 256 + n;
        }
        std::string u;
        if (cp < 0x80) {
            u += static_cast<char>(cp);
        } else if (cp < 0x800) {
            u += static_cast<char>(0xC0 | (cp >> 6));
            u += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            u += static_cast<char>(0xE0 | (cp >> 12));
            u += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            u += static_cast<char>(0x80 | (cp & 0x3F));
        }
        // JSON-escape the (ASCII/UTF-8) token string.
        std::string esc;
        for (unsigned char c: u) {
            if (c == '"' || c == '\\') esc += '\\';
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                esc += buf;
            } else {
                esc += static_cast<char>(c);
            }
        }
        if (b) json += ",";
        json += "\"" + esc + "\":" + std::to_string(b);
    }
    // "he" and "ll" merged tokens (ids 256/257) — byte tokens h(104),e(101),l(108).
    json += ",\"he\":256,\"ll\":257";
    json += "},\"merges\":[\"h e\",\"l l\"]}}";

    {
        std::ofstream f(path, std::ios::binary);
        f << json;
    }

    Tokenizer tok;
    tok.configureForArchitecture("qwen35");
    ASSERT_TRUE(tok.loadFromHFTokenizer(path, "qwen35"));
    EXPECT_GE(tok.vocabSize(), 256u);
    // Special tokens resolved from the real vocabulary.
    EXPECT_EQ(tok.eosTokenId(), 258);
    EXPECT_EQ(tok.imStartId(), 259);
    EXPECT_EQ(tok.imEndId(), 260);
    EXPECT_TRUE(tok.isEogToken(258));
    EXPECT_TRUE(tok.isEogToken(260));

    // Encode/decode round-trip on plain ASCII text.
    const std::string text = "hello";
    auto ids = tok.encode(text);
    ASSERT_FALSE(ids.empty());
    std::string decoded;
    for (int32_t t: ids) decoded += tok.decodeToken(t);
    EXPECT_EQ(decoded, text);

    // The <|im_end|> marker is emitted as its single id, not BPE-split.
    auto frame = tok.encode("hi<|im_end|>");
    ASSERT_FALSE(frame.empty());
    EXPECT_EQ(frame.back(), 260);

    fs::remove_all(dir);
}

// ---------------------------------------------------------------------------
// EXL3 checkpoint: full Model::load through the format-neutral path (only runs
// when a directory is resolvable, e.g. TINYCODER_MODEL_PATH points at a
// checkpoint; otherwise skipped).
// ---------------------------------------------------------------------------
TEST(EXL3LoaderTest, RealCheckpointIfAvailable) {
    namespace fs = std::filesystem;
    const char *env = std::getenv("TINYCODER_EXL3_CHECKPOINT");
    std::string dir = env ? env : "/data/models/qwen/qwen3.8-27b/qwen3.8-27B-exl3-1.60bpw";
    if (!fs::exists(dir) || !fs::exists(dir + "/config.json")) {
        GTEST_SKIP() << "no EXL3 checkpoint at " << dir;
    }

    // Metadata-only load is cheap and validates config.json mapping.
    // (hasTrellis() is only set by the full load(), which materializes tensors.)
    EXL3Loader loader;
    ASSERT_TRUE(loader.loadMetadata(dir)) << "loadMetadata failed for " << dir;
    EXPECT_GT(loader.config().numLayers, 0u);
    EXPECT_GT(loader.config().hiddenSize, 0u);
    EXPECT_GT(loader.config().vocabSize, 0u);
    EXPECT_FALSE(loader.detectedArchitecture().empty());

    // The checkpoint's real HF tokenizer.json loads and round-trips text — this
    // is the tokenizer ModelLoad now wires for EXL3 (no 20-minute weight load).
    ASSERT_TRUE(fs::exists(dir + "/tokenizer.json"));
    Tokenizer tok;
    ASSERT_TRUE(tok.loadFromHFTokenizer(dir + "/tokenizer.json",
                                        loader.detectedArchitecture()));
    EXPECT_GT(tok.vocabSize(), 200000u);
    // Qwen3.5 places the specials at the end of the vocabulary.
    EXPECT_EQ(tok.eosTokenId(), 248044);
    EXPECT_EQ(tok.imStartId(), 248045);
    EXPECT_EQ(tok.imEndId(), 248046);
    EXPECT_TRUE(tok.isEogToken(248046));

    const std::string text = "Hello world";
    auto ids = tok.encode(text);
    ASSERT_FALSE(ids.empty());
    std::string decoded;
    for (int32_t t: ids) decoded += tok.decodeToken(t);
    EXPECT_EQ(decoded, text);
    // A chat marker survives as a single special id.
    auto frame = tok.encode("hi<|im_end|>");
    ASSERT_FALSE(frame.empty());
    EXPECT_EQ(frame.back(), 248046);
}
