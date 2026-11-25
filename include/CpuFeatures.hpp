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

// CpuFeatures.hpp - runtime SIMD-level detection, architecture-neutral.
//
// The engine's SIMD kernels are x86-only (AVX2/AVX-512/AMX translation units compiled with the
// matching -m flags); on aarch64 there is no x86 instruction set to dispatch
// to, so maxSimdLevel() returns SCALAR and every caller takes its scalar path.
// That is a supported configuration, not a degraded one: on Jetson the compute
// that matters runs on the GPU.
//
// The x86 detection is CPUID-based and capped by the compile-time ENABLE_*
// macros, so a build selects the same kernels it was compiled for.
// Note the OS-state checks (XCR0): a CPU that advertises AVX2 while the kernel
// has not enabled YMM state must not be given an AVX2 kernel, or the first
// instruction is SIGILL.

#pragma once

#include <cstdint>
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define TINYCODER_ARCH_X86 1
#else
#define TINYCODER_ARCH_X86 0
#endif

#if TINYCODER_ARCH_X86 && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace tinycoder::cpu {

    /// SIMD instruction set levels in ascending order.
    enum class SimdLevel : int {
        SCALAR = 0,///< No SIMD (scalar fallback)
        SSE2 = 1,  ///< SSE2
        SSE3 = 2,  ///< SSE3
        AVX = 3,   ///< AVX (128-bit)
        AVX2 = 4,  ///< AVX2 (256-bit)
        AVX512 = 5,///< AVX-512
        AMX = 6    ///< AMX (Intel Advanced Matrix Extensions)
    };

    /// True when $TINYCODER_FORCE_SCALAR asks for the scalar reference path.
    /// Checked once per process. The dispatchers below consult it, so the
    /// switch really means "no SIMD anywhere" and the scalar kernels stay
    /// reachable (and testable) on an AVX2 host.
    inline bool forceScalar() noexcept {
        static const bool forced = [] {
            const char *v = std::getenv("TINYCODER_FORCE_SCALAR");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        return forced;
    }

#if TINYCODER_ARCH_X86

    namespace detail {

#if defined(_MSC_VER)

        struct CpuidResult {
            int eax, ebx, ecx, edx;
        };

        inline CpuidResult cpuid(int leaf, int subleaf) noexcept {
            int regs[4] = {0, 0, 0, 0};
            __cpuidex(regs, leaf, subleaf);
            return {regs[0], regs[1], regs[2], regs[3]};
        }

        inline std::uint64_t xcr0() noexcept {
            return _xgetbv(0);
        }

#else

        struct CpuidResult {
            unsigned int eax, ebx, ecx, edx;
        };

        inline CpuidResult cpuid(unsigned int leaf, unsigned int subleaf) noexcept {
            unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
            __asm__ volatile("cpuid"
                             : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                             : "a"(leaf), "c"(subleaf));
            return {eax, ebx, ecx, edx};
        }

        inline std::uint64_t xcr0() noexcept {
            unsigned int eax = 0, edx = 0;
            __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
            return (static_cast<std::uint64_t>(edx) << 32) | eax;
        }

#endif

        inline bool supportsSse2() noexcept {
            // CPUID.1:EDX[26]
            return (cpuid(1, 0).edx >> 26) & 1u;
        }

        inline bool supportsSse3() noexcept {
            // CPUID.1:ECX[0]
            return cpuid(1, 0).ecx & 1u;
        }

        inline bool supportsAvx() noexcept {
            // CPUID.1:ECX[28] AVX, [27] OSXSAVE; XCR0[2:1] = 11b (XMM+YMM)
            const CpuidResult r = cpuid(1, 0);
            const bool hasAvx = (r.ecx >> 28) & 1u;
            const bool hasOsxsave = (r.ecx >> 27) & 1u;
            if (!hasAvx || !hasOsxsave) return false;
            return (xcr0() & 0x6u) == 0x6u;
        }

        inline bool supportsAvx2() noexcept {
            if (!supportsAvx()) return false;
            // CPUID.(7,0):EBX[5]
            return (cpuid(7, 0).ebx >> 5) & 1u;
        }

        inline bool supportsAvx512() noexcept {
            if (!supportsAvx()) return false;
            // CPUID.(7,0):EBX[16] AVX512F; XCR0[7:5] and [2:1] must be set
            if (!((cpuid(7, 0).ebx >> 16) & 1u)) return false;
            return (xcr0() & 0xE6u) == 0xE6u;
        }

        inline bool supportsAmx() noexcept {
            // CPUID.(7,0):EDX[24] AMX-TILE
            return (cpuid(7, 0).edx >> 24) & 1u;
        }

    }// namespace detail

    /// Maximum SIMD level of this CPU, capped by the compile-time ENABLE_*
    /// macros (set by CMake for this target).
    inline SimdLevel maxSimdLevel() noexcept {
        if (forceScalar()) return SimdLevel::SCALAR;
#ifdef ENABLE_AMX
        if (detail::supportsAmx()) return SimdLevel::AMX;
#endif
#ifdef ENABLE_AVX512
        if (detail::supportsAvx512()) return SimdLevel::AVX512;
#endif
#ifdef ENABLE_AVX2
        if (detail::supportsAvx2()) return SimdLevel::AVX2;
#endif
#ifdef ENABLE_AVX
        if (detail::supportsAvx()) return SimdLevel::AVX;
#endif
#ifdef ENABLE_SSE3
        if (detail::supportsSse3()) return SimdLevel::SSE3;
#endif
#ifdef ENABLE_SSE2
        if (detail::supportsSse2()) return SimdLevel::SSE2;
#endif
        return SimdLevel::SCALAR;
    }

#else// !TINYCODER_ARCH_X86

    /// No x86 SIMD on this architecture (e.g. aarch64/Jetson): the scalar
    /// kernels are the whole CPU path.
    inline SimdLevel maxSimdLevel() noexcept {
        return SimdLevel::SCALAR;
    }

#endif// TINYCODER_ARCH_X86

    /// True when the CPU can run code at `level` or above. Cached: the level
    /// cannot change during a process lifetime.
    inline bool simdAtLeast(SimdLevel level) noexcept {
        static const SimdLevel cached = maxSimdLevel();
        return cached >= level;
    }

}// namespace tinycoder::cpu
