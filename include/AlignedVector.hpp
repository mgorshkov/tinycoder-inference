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

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace tinycoder {

    /// @brief A std::vector-like contiguous buffer whose data is aligned to a
    /// cache-line boundary (default 64 bytes).
    ///
    /// Weight matrices are read by the SIMD mat-vec kernels with `_mm256_load_ps`
    /// / `_mm256_loadu_si128` etc. 64-byte alignment guarantees each row block
    /// starts on a cache-line boundary, which reduces L1/L2 cache-miss penalties
    /// and lets the compiler use aligned (non-`loadu`) loads where possible.
    ///
    /// The API intentionally mirrors the subset of std::vector used by the model
    /// loader and kernels, so call sites change only by type.
    ///
    /// Besides the owning (heap) mode, the buffer supports a NON-OWNING
    /// external mode via setExternal(): the pointer simply aliases memory owned
    /// elsewhere (e.g. a file-backed mmap kept alive by the Model's persistent
    /// GGUFLoader). data()/size()/empty() are transparent, so every kernel call
    /// site works unchanged, while zero bytes are copied at load time. Copying
    /// an external buffer shares the pointer (the real owner outlives the
    /// Model); copying an owned buffer deep-copies exactly as before.
    template<typename T, size_t ALIGN = 64>
    class AlignedVector {
    public:
        static_assert(ALIGN >= sizeof(void *), "alignment must be at least a pointer");
        static_assert((ALIGN & (ALIGN - 1)) == 0, "alignment must be a power of two");

        AlignedVector() = default;
        ~AlignedVector() { releaseStorage(); }

        AlignedVector(const AlignedVector &other) { refFrom(other); }
        AlignedVector(AlignedVector &&other) noexcept { moveFrom(other); }

        AlignedVector &operator=(const AlignedVector &other) {
            if (this != &other) {
                refFrom(other);
            }
            return *this;
        }
        AlignedVector &operator=(AlignedVector &&other) noexcept {
            if (this != &other) {
                releaseStorage();
                moveFrom(other);
            }
            return *this;
        }

        /// @brief Copy-assign from a std::vector (e.g. helper functions that return
        /// std::vector accumulate into the aligned buffer). Copies the contents and
        /// reallocates the aligned backing store.
        template<typename U, typename Alloc>
        AlignedVector &operator=(const std::vector<U, Alloc> &src) {
            assign(src.data(), src.data() + src.size());
            return *this;
        }

        /// @brief Adopt an EXTERNAL (non-owning) buffer.
        ///
        /// The caller guarantees the memory at [ptr, ptr + count) outlives this
        /// AlignedVector. For the model loader this is the Model's persistent
        /// GGUFLoader mmap: the quantized weight section stays file-backed and
        /// zero bytes are copied at load time (the previous heap-copy path
        /// doubled the 36.9 GB model and OOM-killed the process). This
        /// AlignedVector never frees this memory; any previous owned storage is
        /// released, and any previous external reference is dropped.
        void setExternal(const T *ptr, size_t count) {
            releaseStorage();
            if (ptr == nullptr || count == 0) {
                data_ = nullptr;
                n_ = 0;
                return;
            }
            data_ = const_cast<T *>(ptr);
            n_ = count;
            external_ = true;
        }

        /// @brief true when this buffer aliases memory it does not own.
        bool external() const noexcept { return external_; }

        /// @brief Resize the buffer and leave contents uninitialized.
        void resize(size_t n) {
            if (n == n_) {
                return;
            }
            allocate(n);
        }

        /// @brief Fill the buffer from a contiguous range [first, last).
        /// Deep-copies into freshly allocated aligned (owned) storage.
        void assign(const T *first, const T *last) {
            if (first == nullptr || last == nullptr || first == last) {
                releaseStorage();
                return;
            }
            const size_t count = static_cast<size_t>(last - first);
            allocate(count);
            if (count > 0) {
                std::memcpy(data_, first, count * sizeof(T));
            }
        }

        /// @brief Fill from a std::vector source (convenience for existing callers).
        template<typename U, typename Alloc>
        void assign(const std::vector<U, Alloc> &src) {
            assign(src.data(), src.data() + src.size());
        }

        T *data() noexcept { return data_; }
        const T *data() const noexcept { return data_; }

        size_t size() const noexcept { return n_; }
        bool empty() const noexcept { return n_ == 0; }

        T &operator[](size_t i) noexcept { return data_[i]; }
        const T &operator[](size_t i) const noexcept { return data_[i]; }

        void clear() {
            releaseStorage();
        }

    private:
        /// @brief Adopt the contents of another buffer without transferring
        /// ownership: shares the pointer of an external (non-owning) source,
        /// deep-copies an owned source.
        void refFrom(const AlignedVector &other) {
            if (other.external_) {
                releaseStorage();
                data_ = other.data_;
                n_ = other.n_;
                external_ = true;
                return;
            }
            if (other.n_ == 0 || other.data_ == nullptr) {
                releaseStorage();
                return;
            }
            assign(other.data_, other.data_ + other.n_);
        }

        void allocate(size_t n) {
            releaseStorage();
            n_ = n;
            if (n_ == 0) {
                data_ = nullptr;
                return;
            }
            size_t bytes = n_ * sizeof(T);
            if (bytes == 0) {
                data_ = nullptr;
                return;
            }
            void *ptr = nullptr;
            // posix_memalign requires the size to be a multiple of the alignment.
            size_t allocSize = ((bytes + ALIGN - 1) / ALIGN) * ALIGN;
            if (allocSize < bytes) {
                // overflow guard; fall back to null (caller treats as OOM)
                data_ = nullptr;
                n_ = 0;
                return;
            }
            if (posix_memalign(&ptr, ALIGN, allocSize) != 0 || ptr == nullptr) {
                data_ = nullptr;
                n_ = 0;
                return;
            }
            data_ = static_cast<T *>(ptr);
        }

        void deallocate() {
            if (data_ != nullptr) {
                std::free(data_);
                data_ = nullptr;
            }
        }

        /// @brief Release whatever storage this buffer refers to: free owned
        /// (heap) memory, or drop an external (borrowed) reference without
        /// touching the memory itself. Never frees a borrowed pointer.
        void releaseStorage() {
            if (external_) {
                external_ = false;
                data_ = nullptr;
                n_ = 0;
            } else {
                deallocate();
            }
        }

        void moveFrom(AlignedVector &other) noexcept {
            data_ = other.data_;
            n_ = other.n_;
            external_ = other.external_;
            other.data_ = nullptr;
            other.n_ = 0;
            other.external_ = false;
        }

        T *data_ = nullptr;
        size_t n_ = 0;
        bool external_ = false;
    };
}// namespace tinycoder