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
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "IModelLoader.hpp"
#include "ModelConfig.hpp"

namespace tinycoder {

    /// @brief GGUF v3 file format reader (implements IModelLoader).
    ///
    /// The GGML type enum, PrismMetadata, TensorInfo and the
    /// ggmlBlockSize()/ggmlTypeSize() helpers now live in IModelLoader.hpp so
    /// format-agnostic code (the EXL3 loader, its dequant kernels, Model) can
    /// use them without pulling in the GGUF reader. This header re-exports them
    /// to existing consumers (GGMLDequantize.hpp, tools, unit tests).
    class GGUFLoader : public IModelLoader {
    public:
        GGUFLoader() = default;
        ~GGUFLoader();

        /// @brief Load a GGUF model file and extract tensor data.
        /// @param path Path to the .gguf file
        /// @return true if loading succeeded
        bool load(const std::string &path) override;

        /// @brief Load only metadata from a GGUF file (no tensor data).
        /// Useful for estimating memory requirements before a full load.
        /// @param path Path to the .gguf file
        /// @return true if metadata was read successfully
        bool loadMetadata(const std::string &path) override;

        /// @brief Get the model configuration parsed from GGUF metadata.
        const ModelConfig &config() const override { return config_; }

        /// @brief Prism Hadamard-fold metadata (empty/`present == false` for
        /// every non-Prism model).
        const PrismMetadata &prism() const override { return prism_; }

        /// @brief Get raw tensor data by name.
        /// @param name Tensor name (e.g., "blk.0.attn_q.weight")
        /// @return Pointer to tensor data, or nullptr if not found
        const uint8_t *getTensor(const std::string &name) const override;

        /// @brief Get tensor info (offset, size, shape). TensorInfo is the
        /// format-neutral record declared in IModelLoader.hpp.
        const TensorInfo *getTensorInfo(const std::string &name) const override;

        /// @brief Read-only access to the enumerated tensor records (name,
        /// offset, shape, type) parsed from the GGUF header. Available after
        /// load() or loadMetadata(). Returns a lazily-built public view over
        /// the private records.
        const std::vector<TensorInfo> &tensorRecords() const override;

        /// @brief GGUFLoader is the native GGUF container reader.
        bool isGguf() const override { return true; }

    private:
        // GGUF header structures
        struct GGUFHeader {
            uint32_t magic;          // "GGUF" magic bytes
            uint32_t version;        // Format version (3)
            uint64_t tensorCount;    // Number of tensors
            uint64_t metadataKVCount;// Number of metadata key-value pairs
        };

        struct GGUFTensorInfo {
            std::string name;
            uint32_t type;              // GGML_TYPE_* enum
            uint64_t offset;            // Offset from start of tensor data
            std::vector<uint32_t> shape;// Dimensions (numpy-style, reversed)
        };

        bool readHeader();
        bool readMetadata();
        bool readTensorInfos();
#if defined(__linux__)
        /// @brief Map the tensor data section into memory (file-backed, no heap
        /// copy of the weights). Requires the loader to outlive the consumer.
        bool mapTensorData();
#else
        /// @brief Read the tensor data section into a heap buffer (fallback).
        bool readTensorData();
#endif
        /// @brief Release the mapped/heap tensor data (called by the destructor).
        void unmapTensorData();

        std::ifstream file_;
        std::string filePath_;
        ModelConfig config_;
        GGUFHeader header_{};

        // Metadata storage
        std::unordered_map<std::string, std::string> metadata_;

        // Prism Hadamard-fold metadata (parsed from the prism.hadamard.* keys
        // during readMetadata; captured here because sign_values / weight_names
        // are large arrays that the scalar metadata_ map does not retain).
        PrismMetadata prism_;

        // Tensor storage
        std::vector<GGUFTensorInfo> tensorInfos_;
        std::unordered_map<std::string, size_t> tensorNameIndex_;

        // Lazily-built public view of tensor records (tensorRecords()).
        mutable std::vector<TensorInfo> tensorRecordCache_;
        mutable bool tensorRecordCacheValid_ = false;
#if defined(__linux__)
        // File-backed mmap of the GGUF tensor data section. The loader itself
        // holds NO heap copy of the weights: pages are faulted in from the page
        // cache on demand and stay file-backed (reclaimable), which is what lets
        // a 27B (16.4 GB) model load into ~25 GB of RAM instead of the ~41 GB
        // peak of the previous all-heap approach (16.4 GB heap read + 16.4 GB of
        // per-layer weight copies + ~8.5 GB of pre-dequantized embeddings).
        const uint8_t *tensorData_ = nullptr;// Tensor section base (may be offset
                                             // within the mapped range).
        void *mmapPtr_ = nullptr;            // Raw mmap() result (for munmap).
        uint64_t mmapLen_ = 0;               // Mapped length (for munmap).
        int mmapFd_ = -1;                    // Read-only fd used for the mapping.
#else
        std::vector<uint8_t> tensorData_;// All tensor data concatenated
#endif

        uint64_t tensorDataOffset_ = 0;
    };

}// namespace tinycoder
