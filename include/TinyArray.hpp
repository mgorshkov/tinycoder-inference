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

// TinyArray.hpp - owning row-major array and shape used by the engine.
//
// Architecture-neutral by design: no SIMD headers, no OpenMP pragmas, no CUDA
// and no expression templates.  Allocation is `new T[n]` (uninitialised), so a
// large buffer such as the KV cache costs no page-touching at allocation time.
// Safe to include from host code in a .cu translation unit.

#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace tinycoder {

    /// @brief Extent list of an Array, constructed as Shape{rows, cols, ...}.
    ///
    /// Only construction and numel() are needed by the engine; the accessors
    /// exist so a caller can inspect a shape without re-deriving it.
    class Shape {
    public:
        Shape() noexcept = default;

        /// Variadic integral extents. SFINAE keeps Shape copy/move construction
        /// from being captured by this template.
        template<typename... Dims,
                 typename = std::enable_if_t<
                         (sizeof...(Dims) > 0) &&
                         (std::is_integral_v<std::decay_t<Dims>> && ...)>>
        Shape(Dims... dims) : dims_{static_cast<std::size_t>(dims)...} {}

        /// Number of dimensions (rank), not the element count.
        std::size_t size() const noexcept { return dims_.size(); }

        std::size_t operator[](std::size_t i) const noexcept { return dims_[i]; }

        /// Product of the extents: the number of elements of this shape.
        std::size_t numel() const noexcept {
            if (dims_.empty()) return 0;
            std::size_t n = 1;
            for (std::size_t d: dims_) n *= d;
            return n;
        }

        const std::vector<std::size_t> &dims() const noexcept { return dims_; }

        bool operator==(const Shape &other) const noexcept {
            return dims_ == other.dims_;
        }

    private:
        std::vector<std::size_t> dims_;
    };

    /// @brief Owning, row-major array of T with an optional Shape.
    ///
    /// Storage is uninitialised on construction.
    /// Copies are deep; moves transfer ownership.
    template<typename T>
    class Array {
    public:
        using value_type = T;
        using iterator = T *;
        using const_iterator = const T *;

        Array() noexcept = default;

        /// Allocate shape.numel() elements, uninitialised.
        explicit Array(const Shape &shape)
            : shape_(shape), size_(shape.numel()) {
            if (size_ != 0) {
                data_.reset(new T[size_]);
            }
        }

        explicit Array(std::size_t size) : Array(Shape{size}) {}

        /// Adopt (copy) a contiguous vector. The vector's length wins over the
        /// shape if they disagree, so size() always matches the data.
        Array(const std::vector<T> &values, const Shape &shape = Shape{})
            : shape_(shape.numel() == values.size() ? shape : Shape{values.size()}),
              size_(values.size()) {
            if (size_ != 0) {
                data_.reset(new T[size_]);
                std::memcpy(data_.get(), values.data(), size_ * sizeof(T));
            }
        }

        /// Adopt a fixed-size array (dequantize scratch).
        template<std::size_t N>
        explicit Array(const std::array<T, N> &values)
            : Array(values.data(), N) {}

        Array(const T *values, std::size_t count)
            : shape_(Shape{count}), size_(count) {
            if (size_ != 0) {
                data_.reset(new T[size_]);
                std::memcpy(data_.get(), values, size_ * sizeof(T));
            }
        }

        Array(const Array &other)
            : shape_(other.shape_), size_(other.size_) {
            if (size_ != 0) {
                data_.reset(new T[size_]);
                std::memcpy(data_.get(), other.data_.get(), size_ * sizeof(T));
            }
        }

        Array(Array &&other) noexcept = default;

        Array &operator=(const Array &other) {
            if (this != &other) {
                Array copy(other);
                swap(copy);
            }
            return *this;
        }

        Array &operator=(Array &&other) noexcept = default;

        T *data() noexcept { return data_.get(); }
        const T *data() const noexcept { return data_.get(); }

        std::size_t size() const noexcept { return size_; }
        bool empty() const noexcept { return size_ == 0; }

        const Shape &shape() const noexcept { return shape_; }

        T &operator[](std::size_t i) noexcept { return data_[i]; }
        const T &operator[](std::size_t i) const noexcept { return data_[i]; }

        T get(std::size_t i) const noexcept { return data_[i]; }
        void set(std::size_t i, T value) noexcept { data_[i] = value; }

        T *begin() noexcept { return data_.get(); }
        T *end() noexcept { return data_.get() + size_; }
        const T *begin() const noexcept { return data_.get(); }
        const T *end() const noexcept { return data_.get() + size_; }

        void swap(Array &other) noexcept {
            std::swap(shape_, other.shape_);
            std::swap(size_, other.size_);
            data_.swap(other.data_);
        }

    private:
        Shape shape_;
        std::size_t size_ = 0;
        std::unique_ptr<T[]> data_;
    };

    template<typename T>
    void swap(Array<T> &a, Array<T> &b) noexcept {
        a.swap(b);
    }

}// namespace tinycoder
