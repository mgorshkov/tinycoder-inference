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

// Per-model thread pools: a Model owns its pool, binds it on the calling thread
// for the duration of forward(), and each worker resolves current() to its own
// owner. These tests cover the binding contract that the 60+ kernel call sites
// depend on (they call ThreadPool::current() instead of instance()).

#include <atomic>

#include <gtest/gtest.h>

#include "ThreadPool.hpp"

namespace {

    using tinycoder::ThreadPool;
    using tinycoder::ThreadPoolBinding;

    TEST(ThreadPoolOwnership, BindingIsScopedAndRestored) {
        ThreadPool pool;
        pool.initialize(2);

        // Nothing bound: current() degrades to the process-wide instance.
        EXPECT_EQ(ThreadPool::boundCurrent(), nullptr);
        EXPECT_EQ(&ThreadPool::current(), &ThreadPool::instance());

        {
            ThreadPoolBinding binding(&pool);
            EXPECT_EQ(&ThreadPool::current(), &pool);

            // A null binding (a Model that has no pool yet) must not clobber an
            // outer binding.
            {
                ThreadPoolBinding inner(nullptr);
                EXPECT_EQ(&ThreadPool::current(), &pool);
            }
            EXPECT_EQ(&ThreadPool::current(), &pool);
        }

        EXPECT_EQ(ThreadPool::boundCurrent(), nullptr);
        EXPECT_EQ(&ThreadPool::current(), &ThreadPool::instance());
    }

    TEST(ThreadPoolOwnership, WorkersResolveToTheirOwnPool) {
        ThreadPool poolA;
        ThreadPool poolB;
        poolA.initialize(2);
        poolB.initialize(2);

        std::atomic<int> wrongA{0};
        std::atomic<int> ranA{0};
        {
            ThreadPoolBinding binding(&poolA);
            poolA.parallelFor(0, 64, [&](uint32_t) {
                if (&ThreadPool::current() != &poolA) {
                    wrongA.fetch_add(1, std::memory_order_relaxed);
                }
                ranA.fetch_add(1, std::memory_order_relaxed);
            });
        }

        std::atomic<int> wrongB{0};
        std::atomic<int> ranB{0};
        {
            ThreadPoolBinding binding(&poolB);
            poolB.parallelFor(0, 64, [&](uint32_t) {
                if (&ThreadPool::current() != &poolB) {
                    wrongB.fetch_add(1, std::memory_order_relaxed);
                }
                ranB.fetch_add(1, std::memory_order_relaxed);
            });
        }

        EXPECT_EQ(ranA.load(), 64);
        EXPECT_EQ(ranB.load(), 64);
        EXPECT_EQ(wrongA.load(), 0) << "a worker used the wrong pool for A";
        EXPECT_EQ(wrongB.load(), 0) << "a worker used the wrong pool for B";
    }

}// namespace
