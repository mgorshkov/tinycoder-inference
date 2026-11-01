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

#include "SafetensorsReader.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace tinycoder {

    // =======================================================================
    // JsonValue
    // =======================================================================

    const JsonValue *JsonValue::find(const std::string &key) const {
        if (type != Type::Object) {
            return nullptr;
        }
        for (const auto &kv: members) {
            if (kv.first == key) {
                return &kv.second;
            }
        }
        return nullptr;
    }

    const JsonValue *JsonValue::at(size_t i) const {
        if (type != Type::Array || i >= array.size()) {
            return nullptr;
        }
        return &array[i];
    }

    namespace {

        // Recursive-descent JSON parser over a null-terminated byte buffer.
        struct JsonParser {
            const char *p;
            const char *end;
            std::string err;

            bool fail(const std::string &msg) {
                if (err.empty()) {
                    err = msg;
                }
                return false;
            }

            void skipWs() {
                while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
                    ++p;
                }
            }

            bool parseValue(JsonValue &out) {
                skipWs();
                if (p >= end) {
                    return fail("unexpected end of JSON");
                }
                char c = *p;
                switch (c) {
                    case '{':
                        return parseObject(out);
                    case '[':
                        return parseArray(out);
                    case '"': {
                        out.type = JsonValue::Type::String;
                        return parseString(out.str);
                    }
                    case 't':
                    case 'f':
                        return parseBool(out);
                    case 'n':
                        return parseNull(out);
                    default:
                        if (c == '-' || (c >= '0' && c <= '9')) {
                            return parseNumber(out);
                        }
                        return fail("unexpected character in JSON");
                }
            }

            bool parseNull(JsonValue &out) {
                if (end - p >= 4 && std::strncmp(p, "null", 4) == 0) {
                    p += 4;
                    out.type = JsonValue::Type::Null;
                    return true;
                }
                return fail("invalid literal (expected null)");
            }

            bool parseBool(JsonValue &out) {
                if (end - p >= 4 && std::strncmp(p, "true", 4) == 0) {
                    p += 4;
                    out.type = JsonValue::Type::Bool;
                    out.boolean = true;
                    return true;
                }
                if (end - p >= 5 && std::strncmp(p, "false", 5) == 0) {
                    p += 5;
                    out.type = JsonValue::Type::Bool;
                    out.boolean = false;
                    return true;
                }
                return fail("invalid boolean literal");
            }

            bool parseNumber(JsonValue &out) {
                const char *start = p;
                if (p < end && *p == '-') {
                    ++p;
                }
                while (p < end && ((*p >= '0' && *p <= '9') || *p == '.' || *p == 'e' ||
                                   *p == 'E' || *p == '+' || *p == '-')) {
                    ++p;
                }
                std::string num(start, static_cast<size_t>(p - start));
                if (num.empty()) {
                    return fail("empty number");
                }
                out.type = JsonValue::Type::Number;
                out.number = std::strtod(num.c_str(), nullptr);
                return true;
            }

            bool parseString(std::string &out) {
                if (p >= end || *p != '"') {
                    return fail("expected string");
                }
                ++p;
                out.clear();
                while (p < end) {
                    char c = *p++;
                    if (c == '"') {
                        return true;
                    }
                    if (c == '\\') {
                        if (p >= end) {
                            return fail("unterminated escape");
                        }
                        char e = *p++;
                        switch (e) {
                            case '"': out.push_back('"'); break;
                            case '\\': out.push_back('\\'); break;
                            case '/': out.push_back('/'); break;
                            case 'b': out.push_back('\b'); break;
                            case 'f': out.push_back('\f'); break;
                            case 'n': out.push_back('\n'); break;
                            case 'r': out.push_back('\r'); break;
                            case 't': out.push_back('\t'); break;
                            case 'u': {
                                if (end - p < 4) {
                                    return fail("truncated \\u escape");
                                }
                                unsigned code = 0;
                                for (int i = 0; i < 4; ++i) {
                                    char h = *p++;
                                    code <<= 4;
                                    if (h >= '0' && h <= '9') {
                                        code |= static_cast<unsigned>(h - '0');
                                    } else if (h >= 'a' && h <= 'f') {
                                        code |= static_cast<unsigned>(h - 'a' + 10);
                                    } else if (h >= 'A' && h <= 'F') {
                                        code |= static_cast<unsigned>(h - 'A' + 10);
                                    } else {
                                        return fail("bad hex in \\u escape");
                                    }
                                }
                                // UTF-8 encode (BMP only; surrogates passed through).
                                if (code < 0x80) {
                                    out.push_back(static_cast<char>(code));
                                } else if (code < 0x800) {
                                    out.push_back(static_cast<char>(0xC0 | (code >> 6)));
                                    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                                } else {
                                    out.push_back(static_cast<char>(0xE0 | (code >> 12)));
                                    out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                                    out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                                }
                                break;
                            }
                            default:
                                return fail("unknown escape");
                        }
                    } else {
                        out.push_back(c);
                    }
                }
                return fail("unterminated string");
            }

            bool parseArray(JsonValue &out) {
                out.type = JsonValue::Type::Array;
                ++p;// consume '['
                skipWs();
                if (p < end && *p == ']') {
                    ++p;
                    return true;
                }
                while (true) {
                    JsonValue elem;
                    if (!parseValue(elem)) {
                        return false;
                    }
                    out.array.push_back(std::move(elem));
                    skipWs();
                    if (p >= end) {
                        return fail("unterminated array");
                    }
                    if (*p == ',') {
                        ++p;
                        continue;
                    }
                    if (*p == ']') {
                        ++p;
                        return true;
                    }
                    return fail("expected ',' or ']' in array");
                }
            }

            bool parseObject(JsonValue &out) {
                out.type = JsonValue::Type::Object;
                ++p;// consume '{'
                skipWs();
                if (p < end && *p == '}') {
                    ++p;
                    return true;
                }
                while (true) {
                    skipWs();
                    std::string key;
                    if (!parseString(key)) {
                        return false;
                    }
                    skipWs();
                    if (p >= end || *p != ':') {
                        return fail("expected ':' in object");
                    }
                    ++p;
                    JsonValue val;
                    if (!parseValue(val)) {
                        return false;
                    }
                    out.members.emplace_back(std::move(key), std::move(val));
                    skipWs();
                    if (p >= end) {
                        return fail("unterminated object");
                    }
                    if (*p == ',') {
                        ++p;
                        continue;
                    }
                    if (*p == '}') {
                        ++p;
                        return true;
                    }
                    return fail("expected ',' or '}' in object");
                }
            }
        };

    }// namespace

    bool JsonValue::parse(const std::string &text, JsonValue &out, std::string *err) {
        JsonParser parser{text.data(), text.data() + text.size(), {}};
        if (!parser.parseValue(out)) {
            if (err) {
                *err = parser.err;
            }
            return false;
        }
        return true;
    }

    // =======================================================================
    // SafetensorsReader
    // =======================================================================

    SafetensorsReader::~SafetensorsReader() {
        closeAll();
    }

    void SafetensorsReader::closeAll() {
        for (auto &shard: shards_) {
            if (!shard) {
                continue;
            }
#if defined(__linux__)
            if (shard->mmapPtr != nullptr) {
                ::munmap(shard->mmapPtr, shard->mapLen);
                shard->mmapPtr = nullptr;
            }
            if (shard->fd >= 0) {
                ::close(shard->fd);
                shard->fd = -1;
            }
#else
            shard->heap.clear();
            shard->heap.shrink_to_fit();
#endif
            shard->base = nullptr;
        }
        shards_.clear();
        tensors_.clear();
        index_.clear();
    }

    uint32_t SafetensorsReader::dtypeSize(DType dt) {
        switch (dt) {
            case DType::F64: return 8;
            case DType::F32: return 4;
            case DType::F16: return 2;
            case DType::BF16: return 2;
            case DType::I64: return 8;
            case DType::I32: return 4;
            case DType::I16: return 2;
            case DType::I8: return 1;
            case DType::U8: return 1;
            case DType::BOOL: return 1;
            case DType::U16: return 2;
            case DType::U32: return 4;
            case DType::U64: return 8;
            case DType::F8_E4M3: return 1;
            case DType::F8_E5M2: return 1;
            default: return 0;
        }
    }

    const char *SafetensorsReader::dtypeName(DType dt) {
        switch (dt) {
            case DType::F64: return "F64";
            case DType::F32: return "F32";
            case DType::F16: return "F16";
            case DType::BF16: return "BF16";
            case DType::I64: return "I64";
            case DType::I32: return "I32";
            case DType::I16: return "I16";
            case DType::I8: return "I8";
            case DType::U8: return "U8";
            case DType::BOOL: return "BOOL";
            case DType::U16: return "U16";
            case DType::U32: return "U32";
            case DType::U64: return "U64";
            case DType::F8_E4M3: return "F8_E4M3";
            case DType::F8_E5M2: return "F8_E5M2";
            default: return "Unknown";
        }
    }

    static SafetensorsReader::DType dtypeFromString(const std::string &s) {
        using D = SafetensorsReader::DType;
        if (s == "F64") return D::F64;
        if (s == "F32") return D::F32;
        if (s == "F16") return D::F16;
        if (s == "BF16") return D::BF16;
        if (s == "I64") return D::I64;
        if (s == "I32") return D::I32;
        if (s == "I16") return D::I16;
        if (s == "I8") return D::I8;
        if (s == "U8") return D::U8;
        if (s == "BOOL") return D::BOOL;
        if (s == "U16") return D::U16;
        if (s == "U32") return D::U32;
        if (s == "U64") return D::U64;
        if (s == "F8_E4M3") return D::F8_E4M3;
        if (s == "F8_E5M2") return D::F8_E5M2;
        return D::Unknown;
    }

    bool SafetensorsReader::open(const std::string &path) {
        std::vector<std::string> one{path};
        return open(one);
    }

    bool SafetensorsReader::open(const std::vector<std::string> &paths) {
        closeAll();
        for (uint32_t i = 0; i < paths.size(); ++i) {
            if (!openShard(paths[i], i)) {
                closeAll();
                return false;
            }
        }
        return !shards_.empty();
    }

    bool SafetensorsReader::openShard(const std::string &path, uint32_t shardIndex) {
        auto shard = std::make_unique<Shard>();
        shard->path = path;

        // Determine the file size first (mmap length / heap read size).
        uint64_t fileSize = 0;
        {
            std::ifstream sz(path, std::ios::binary | std::ios::ate);
            if (!sz.is_open()) {
                std::cerr << "[safetensors] cannot open: " << path << std::endl;
                return false;
            }
            fileSize = static_cast<uint64_t>(sz.tellg());
        }
        if (fileSize < 8) {
            std::cerr << "[safetensors] file too small: " << path << std::endl;
            return false;
        }

#if defined(__linux__)
        shard->fd = ::open(path.c_str(), O_RDONLY);
        if (shard->fd < 0) {
            std::cerr << "[safetensors] open() failed: " << std::strerror(errno) << std::endl;
            return false;
        }
        void *m = ::mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, shard->fd, 0);
        if (m == MAP_FAILED) {
            std::cerr << "[safetensors] mmap() failed: " << std::strerror(errno) << std::endl;
            ::close(shard->fd);
            shard->fd = -1;
            return false;
        }
        shard->mmapPtr = m;
        shard->mapLen = fileSize;
        shard->base = static_cast<const uint8_t *>(m);
#else
        shard->heap.resize(fileSize);
        {
            std::ifstream f(path, std::ios::binary);
            f.read(reinterpret_cast<char *>(shard->heap.data()),
                   static_cast<std::streamsize>(fileSize));
            if (!f) {
                std::cerr << "[safetensors] read failed: " << path << std::endl;
                return false;
            }
        }
        shard->base = shard->heap.data();
#endif

        // Parse the header length + JSON header.
        uint64_t headerLen = 0;
        std::memcpy(&headerLen, shard->base, 8);
        if (headerLen == 0 || 8 + headerLen > fileSize) {
            std::cerr << "[safetensors] invalid header length in " << path << std::endl;
            return false;
        }
        std::string headerJson(reinterpret_cast<const char *>(shard->base + 8),
                               static_cast<size_t>(headerLen));

        JsonValue root;
        std::string err;
        if (!JsonValue::parse(headerJson, root, &err) ||
            root.type != JsonValue::Type::Object) {
            std::cerr << "[safetensors] header JSON parse failed (" << path << "): " << err
                      << std::endl;
            return false;
        }

        shard->payloadOffset = 8 + headerLen;
        const uint64_t payloadLen = fileSize - shard->payloadOffset;

        for (const auto &kv: root.members) {
            const std::string &name = kv.first;
            if (name == "__metadata__") {
                continue;// format-level metadata blob, not a tensor
            }
            const JsonValue &entry = kv.second;
            if (entry.type != JsonValue::Type::Object) {
                continue;
            }
            const JsonValue *dtypeV = entry.find("dtype");
            const JsonValue *shapeV = entry.find("shape");
            const JsonValue *offsV = entry.find("data_offsets");
            if (!dtypeV || !shapeV || !offsV) {
                continue;
            }

            TensorDesc desc;
            desc.name = name;
            desc.dtype = dtypeFromString(dtypeV->asString());
            desc.shard = shardIndex;
            if (shapeV->type == JsonValue::Type::Array) {
                desc.shape.reserve(shapeV->array.size());
                for (const auto &d: shapeV->array) {
                    desc.shape.push_back(d.asU64());
                }
            }
            if (offsV->type == JsonValue::Type::Array && offsV->array.size() >= 2) {
                desc.begin = offsV->array[0].asU64();
                desc.end = offsV->array[1].asU64();
            }
            if (desc.begin > desc.end || desc.end > payloadLen) {
                std::cerr << "[safetensors] tensor \"" << name
                          << "\" offsets out of range" << std::endl;
                return false;
            }
            // Sanity: the declared dtype width must match the shape span.
            uint32_t esz = dtypeSize(desc.dtype);
            uint64_t expect = numel(desc) * esz;
            if (esz > 0 && expect != (desc.end - desc.begin)) {
                std::cerr << "[safetensors] tensor \"" << name << "\" size mismatch: shape "
                          << "declares " << expect << " bytes but offsets span "
                          << (desc.end - desc.begin) << std::endl;
                return false;
            }

            if (index_.count(name)) {
                std::cerr << "[safetensors] duplicate tensor name: " << name << std::endl;
                return false;
            }
            index_[name] = tensors_.size();
            tensors_.push_back(std::move(desc));
        }

        shards_.push_back(std::move(shard));
        return true;
    }

    const uint8_t *SafetensorsReader::shardPayload(uint32_t shard) const {
        if (shard >= shards_.size()) {
            return nullptr;
        }
        return shards_[shard]->base + shards_[shard]->payloadOffset;
    }

    const SafetensorsReader::TensorDesc *SafetensorsReader::find(const std::string &name) const {
        auto it = index_.find(name);
        if (it == index_.end()) {
            return nullptr;
        }
        return &tensors_[it->second];
    }

    const uint8_t *SafetensorsReader::data(const std::string &name) const {
        const TensorDesc *d = find(name);
        return d ? data(*d) : nullptr;
    }

    const uint8_t *SafetensorsReader::data(const TensorDesc &desc) const {
        const uint8_t *base = shardPayload(desc.shard);
        return base ? base + desc.begin : nullptr;
    }

    uint64_t SafetensorsReader::numel(const TensorDesc &desc) {
        uint64_t n = 1;
        for (uint64_t d: desc.shape) {
            n *= d;
        }
        return n;
    }

}// namespace tinycoder
