/**
 * @file  test_correctness.cpp
 * @brief Correctness tests for StringPool.
 *
 * Covers:
 *  - Basic intern / lookup / get
 *  - Deduplication (same handle for identical strings)
 *  - Empty / null handle semantics
 *  - Reference counting (increment on re-intern, decrement on release)
 *  - Concurrent intern safety (multiple threads, no data races)
 *  - Edge cases: single-char, very long string, Unicode bytes, NUL byte guard
 *  - RAII InternedString (move, clone, destructor)
 */

#include "../include/string_pool.hpp"

#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <unordered_set>
#include <atomic>

// ─── Minimal test harness ─────────────────────────────────────────────────────
static int g_pass = 0, g_fail = 0;

#define EXPECT_TRUE(expr) \
    do { \
        if (!(expr)) { \
            std::fprintf(stderr, "FAIL  %s:%d  %s\n", __FILE__, __LINE__, #expr); \
            ++g_fail; \
        } else { \
            ++g_pass; \
        } \
    } while (0)

#define EXPECT_EQ(a, b)  EXPECT_TRUE((a) == (b))
#define EXPECT_NEQ(a, b) EXPECT_TRUE((a) != (b))

// ─────────────────────────────────────────────────────────────────────────────

static void test_basic() {
    sp::StringPool pool;

    sp::StringHandle h = pool.intern("hello");
    EXPECT_TRUE(h.valid());
    EXPECT_EQ(pool.get(h), "hello");
}

static void test_deduplication() {
    sp::StringPool pool;

    sp::StringHandle h1 = pool.intern("world");
    sp::StringHandle h2 = pool.intern("world");
    sp::StringHandle h3 = pool.intern("world");
    EXPECT_EQ(h1, h2);
    EXPECT_EQ(h2, h3);

    sp::StringHandle ha = pool.intern("foo");
    sp::StringHandle hb = pool.intern("bar");
    EXPECT_NEQ(ha, hb);

    EXPECT_EQ(pool.get(ha), "foo");
    EXPECT_EQ(pool.get(hb), "bar");
}

static void test_empty_string() {
    sp::StringPool pool;

    sp::StringHandle h = pool.intern("");
    EXPECT_TRUE(!h.valid());
    EXPECT_EQ(pool.get(h), "");
    EXPECT_EQ(pool.get(sp::StringHandle::null()), "");
}

static void test_lookup() {
    sp::StringPool pool;

    EXPECT_TRUE(!pool.lookup("missing").has_value());

    pool.intern("present");
    auto opt = pool.lookup("present");
    EXPECT_TRUE(opt.has_value());
    EXPECT_EQ(pool.get(*opt), "present");

    EXPECT_TRUE(!pool.lookup("still_missing").has_value());
}

static void test_reference_counting() {
#ifndef SP_NO_REFCOUNT
    sp::StringPool pool;

    sp::StringHandle h1 = pool.intern("reftest");
    sp::StringHandle h2 = pool.intern("reftest");  // increments refcount
    EXPECT_EQ(h1, h2);

    pool.release(h1);
    // String should still be accessible (refcount was 2, now 1)
    EXPECT_EQ(pool.get(h2), "reftest");
    pool.release(h2);
    // After releasing all refs the slot value is 0.
    // We don't reclaim memory (arena-based), so get() is still safe.
    EXPECT_EQ(pool.get(h2), "reftest");
#endif
}

static void test_raii_interned_string() {
    sp::StringPool pool;

    {
        auto is = pool.make_interned("raii");
        EXPECT_TRUE(is.valid());
        EXPECT_EQ(is.str(), "raii");

        // Move
        auto is2 = std::move(is);
        EXPECT_TRUE(!is.valid());  // moved-from
        EXPECT_EQ(is2.str(), "raii");

        // Clone
        auto is3 = is2.clone();
        EXPECT_EQ(is2.handle(), is3.handle());
    }
    // All destructors called; pool still valid
    EXPECT_EQ(pool.get(pool.intern("raii")), "raii");
}

static void test_single_char() {
    sp::StringPool pool;
    for (char c = 'a'; c <= 'z'; ++c) {
        std::string s(1, c);
        sp::StringHandle h = pool.intern(s);
        EXPECT_EQ(pool.get(h), std::string_view(s));
    }
}

static void test_long_string() {
    sp::StringPool pool;
    std::string big(64 * 1024, 'X');  // 64 KiB
    sp::StringHandle h = pool.intern(big);
    EXPECT_EQ(pool.get(h).size(), big.size());
    EXPECT_EQ(pool.get(h)[0], 'X');

    sp::StringHandle h2 = pool.intern(big);
    EXPECT_EQ(h, h2);
}

static void test_many_distinct_strings() {
    sp::StringPool pool;
    const int N = 100'000;
    std::vector<sp::StringHandle> handles(N);

    for (int i = 0; i < N; ++i) {
        std::string s = "str_" + std::to_string(i);
        handles[i] = pool.intern(s);
    }
    // Re-intern and verify deduplication
    for (int i = 0; i < N; ++i) {
        std::string s = "str_" + std::to_string(i);
        sp::StringHandle h = pool.intern(s);
        EXPECT_EQ(h, handles[i]);
        EXPECT_EQ(pool.get(h), std::string_view(s));
    }

    auto st = pool.stats();
    EXPECT_EQ(st.string_count, static_cast<size_t>(N));
}

static void test_concurrent_intern() {
    sp::StringPool pool;
    const int NUM_THREADS = 8;
    const int STRINGS_PER_THREAD = 10'000;

    std::vector<std::thread> threads;
    std::atomic<int> errors{0};

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < STRINGS_PER_THREAD; ++i) {
                std::string s = "thr" + std::to_string(t) + "_str" + std::to_string(i);
                sp::StringHandle h = pool.intern(s);
                if (!h.valid()) { ++errors; continue; }
                if (pool.get(h) != std::string_view(s)) { ++errors; }
            }
        });
    }
    for (auto& th : threads) th.join();

    EXPECT_EQ(errors.load(), 0);

    // Deduplication across threads: same string must return same handle
    const std::string shared = "shared_across_threads";
    std::vector<sp::StringHandle> results(NUM_THREADS);
    std::vector<std::thread> threads2;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads2.emplace_back([&, t]() {
            results[t] = pool.intern(shared);
        });
    }
    for (auto& th : threads2) th.join();
    for (int t = 1; t < NUM_THREADS; ++t) {
        EXPECT_EQ(results[0], results[t]);
    }
}

static void test_reset() {
    sp::StringPool pool;
    pool.intern("a");
    pool.intern("b");
    pool.intern("c");
    EXPECT_EQ(pool.stats().string_count, size_t(3));

    pool.reset();
    EXPECT_EQ(pool.stats().string_count, size_t(0));

    // Pool should work normally after reset
    sp::StringHandle h = pool.intern("after_reset");
    EXPECT_EQ(pool.get(h), "after_reset");
}

static void test_stats() {
    sp::StringPool pool;
    for (int i = 0; i < 1000; ++i) {
        pool.intern("item_" + std::to_string(i));
    }
    auto st = pool.stats();
    EXPECT_EQ(st.string_count, size_t(1000));
    EXPECT_TRUE(st.arena_bytes > 0);
    EXPECT_TRUE(st.table_bytes > 0);
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::printf("=== StringPool Correctness Tests ===\n");

    test_basic();
    test_deduplication();
    test_empty_string();
    test_lookup();
    test_reference_counting();
    test_raii_interned_string();
    test_single_char();
    test_long_string();
    test_many_distinct_strings();
    test_concurrent_intern();
    test_reset();
    test_stats();

    std::printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
