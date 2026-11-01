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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinycoder {

    /// @brief Read-only safetensors container reader (Hugging Face format).
    ///
    /// Layout: 8-byte little-endian header length N, N bytes of JSON header, then
    /// the packed tensor payload. Each header entry maps a tensor name to
    /// {"dtype": "...", "shape": [...], "data_offsets": [begin, end]} where the
    /// offsets are relative to the start of the payload section.
    ///
    /// Supports the dtypes EXL3 checkpoints use (F16/BF16/F32/I16 markers, and
    /// the full set for completeness). Multimodel checkpoints may be split over
    /// several .safetensors shards; open() takes a list and merges the headers.
    ///
    /// The payload is mmap()ed read-only on Linux (file-backed, reclaimable) and
    /// heap-copied elsewhere — the same strategy GGUFLoader uses for GGUF.
    class SafetensorsReader {
    public:
        SafetensorsReader() = default;
        ~SafetensorsReader();
        SafetensorsReader(const SafetensorsReader &) = delete;
        SafetensorsReader &operator=(const SafetensorsReader &) = delete;

        /// @brief Data type of a stored tensor.
        enum class DType : uint32_t {
            Unknown = 0,
            F64,
            F32,
            F16,
            BF16,
            I64,
            I32,
            I16,
            I8,
            U8,
            BOOL,
            U16,
            U32,
            U64,
            F8_E4M3,
            F8_E5M2,
        };

        /// @brief One tensor's location and layout within its shard.
        struct TensorDesc {
            std::string name;
            DType dtype = DType::Unknown;
            std::vector<uint64_t> shape;
            uint64_t begin = 0;// payload-relative start offset
            uint64_t end = 0;  // payload-relative end offset
            uint32_t shard = 0;// index into the shard list
        };

        /// @brief Open one or more safetensors shards and parse their headers.
        /// @param paths Shard paths (a single-file checkpoint passes one entry).
        /// @return true if every shard header parsed successfully.
        bool open(const std::vector<std::string> &paths);

        /// @brief Convenience overload for a single shard.
        bool open(const std::string &path);

        /// @brief True once at least one shard is open.
        bool isOpen() const { return !shards_.empty(); }

        /// @brief Look up a tensor descriptor by name (nullptr when absent).
        const TensorDesc *find(const std::string &name) const;

        /// @brief Every parsed descriptor (shard order).
        const std::vector<TensorDesc> &tensors() const { return tensors_; }

        /// @brief Raw pointer to a tensor's payload bytes (nullptr when absent).
        const uint8_t *data(const std::string &name) const;

        /// @brief Raw pointer to a descriptor's payload bytes.
        const uint8_t *data(const TensorDesc &desc) const;

        /// @brief Number of elements in a tensor (product of its shape).
        static uint64_t numel(const TensorDesc &desc);

        /// @brief Bytes-per-element for a dtype (Unknown → 0).
        static uint32_t dtypeSize(DType dt);

        /// @brief Human-readable dtype name.
        static const char *dtypeName(DType dt);

        /// @brief Payload base pointer for a shard (exposed for advanced callers).
        const uint8_t *shardPayload(uint32_t shard) const;

    private:
        struct Shard {
            std::string path;
            uint64_t payloadOffset = 0;// file offset where the payload begins
            const uint8_t *base = nullptr;// mapped file base
            uint64_t mapLen = 0;
            void *mmapPtr = nullptr;
            int fd = -1;
            std::vector<uint8_t> heap;// non-Linux fallback storage
            ~Shard() = default;
        };

        bool openShard(const std::string &path, uint32_t shardIndex);
        void closeAll();

        std::vector<std::unique_ptr<Shard>> shards_;
        std::vector<TensorDesc> tensors_;
        std::unordered_map<std::string, size_t> index_;
    };

    /// @brief Minimal JSON value parser (objects/arrays/strings/numbers/bools),
    /// sufficient for safetensors headers. Exposed so unit tests and the EXL3
    /// config.json reader can reuse it.
    class JsonValue {
    public:
        enum class Type { Null,
                          Bool,
                          Number,
                          String,
                          Array,
                          Object };

        Type type = Type::Null;
        bool boolean = false;
        double number = 0.0;
        std::string str;
        std::vector<JsonValue> array;
        // Object members preserve insertion order (safetensors headers are flat).
        std::vector<std::pair<std::string, JsonValue>> members;

        /// @brief Member lookup (nullptr when absent or not an object).
        const JsonValue *find(const std::string &key) const;

        /// @brief Array element access (nullptr when out of range).
        const JsonValue *at(size_t i) const;

        /// @brief Parse a JSON document. Returns false on syntax error.
        static bool parse(const std::string &text, JsonValue &out, std::string *err = nullptr);

        // Typed convenience accessors (0 / empty when the type does not match).
        double asNumber() const { return type == Type::Number ? number : 0.0; }
        uint64_t asU64() const { return type == Type::Number ? static_cast<uint64_t>(number) : 0; }
        const std::string &asString() const { return str; }
    };

}// namespace tinycoder
