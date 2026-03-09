/**
 * @file  bench_main.cpp
 * @brief Comprehensive benchmark for StringPool vs std::unordered_map.
 *
 * Test scenarios
 * ──────────────
 *  1. intern  – all unique strings (10 M ops, simulates first-time insertion)
 *  2. intern  – all duplicate strings (10 M ops, hot-path lookup)
 *  3. lookup  – existing strings (10 M ops)
 *  4. get     – handle → string_view (10 M ops)
 *  5. mixed   – 50% new + 50% duplicate (10 M ops)
 *  6. multi-thread intern (8 threads × 1.25 M ops = 10 M total)
 *
 *  Reference: same scenarios with std::unordered_map<std::string, uint32_t>
 *             protected by a single std::shared_mutex (mirrors car_string_pool.h)
 *
 *  Memory report: arena bytes + table bytes vs unordered_map heap estimate.
 */

#include "../include/string_pool.hpp"
#include "bench_utils.hpp"

#include <cstdio>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════════
// Reference: naive string pool (mirrors car_string_pool.h)
// ═══════════════════════════════════════════════════════════════════════════════

class NaiveStringPool {
public:
    using StringId = uint32_t;

    NaiveStringPool() { m_strings.emplace_back(""); }

    StringId intern(std::string_view sv) {
        if (sv.empty()) return 0;
        {
            std::shared_lock lock(m_mutex);
            auto it = m_map.find(std::string(sv));
            if (it != m_map.end()) return it->second;
        }
        std::unique_lock lock(m_mutex);
        auto key = std::string(sv);
        auto it  = m_map.find(key);
        if (it != m_map.end()) return it->second;
        StringId id = static_cast<StringId>(m_strings.size());
        m_strings.emplace_back(std::move(key));
        m_map[m_strings.back()] = id;
        return id;
    }

    std::string_view get(StringId id) const {
        if (id == 0 || id >= m_strings.size()) return "";
        return m_strings[id];
    }

    size_t mem_estimate() const {
        size_t sz = 0;
        for (auto& s : m_strings) sz += s.capacity();
        // rough estimate for unordered_map buckets
        sz += m_map.bucket_count() * sizeof(void*);
        return sz;
    }

private:
    mutable std::shared_mutex                     m_mutex;
    std::vector<std::string>                      m_strings;
    std::unordered_map<std::string, StringId>     m_map;
};

// ═══════════════════════════════════════════════════════════════════════════════
// Test-data generator
// ═══════════════════════════════════════════════════════════════════════════════

static std::vector<std::string> gen_strings(size_t n, size_t min_len = 4, size_t max_len = 24) {
    std::vector<std::string> v;
    v.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        size_t len = min_len + (i % (max_len - min_len + 1));
        std::string s(len, 'a');
        // encode index into the string to make each unique
        size_t tmp = i;
        for (size_t j = 0; j < len && tmp; ++j, tmp >>= 5) {
            s[j] = static_cast<char>('a' + (tmp & 31) % 26);
        }
        v.push_back(std::move(s));
    }
    return v;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark cases
// ═══════════════════════════════════════════════════════════════════════════════

static constexpr size_t OPS = 10'000'000;

// ── 1. intern unique ──────────────────────────────────────────────────────────

static bench::Result bench_sp_intern_unique(const std::vector<std::string>& strs) {
    sp::StringPool pool(512);
    bench::Timer t;
    t.start();
    for (size_t i = 0; i < OPS; ++i) {
        auto h = pool.intern(strs[i % strs.size()]);
        (void)h;
    }
    double s = t.elapsed_s();
    auto st = pool.stats();
    return {"StringPool: intern (unique)",  OPS, s,
            st.arena_bytes + st.table_bytes};
}

static bench::Result bench_naive_intern_unique(const std::vector<std::string>& strs) {
    NaiveStringPool pool;
    bench::Timer t;
    t.start();
    for (size_t i = 0; i < OPS; ++i) {
        auto id = pool.intern(strs[i % strs.size()]);
        (void)id;
    }
    return {"NaivePool:  intern (unique)",  OPS, t.elapsed_s(), pool.mem_estimate()};
}

// ── 2. intern duplicate (hot-path) ───────────────────────────────────────────

static bench::Result bench_sp_intern_dup(const std::vector<std::string>& strs) {
    sp::StringPool pool(512);
    // Pre-populate
    for (auto& s : strs) pool.intern(s);

    bench::Timer t;
    t.start();
    for (size_t i = 0; i < OPS; ++i) {
        auto h = pool.intern(strs[i % strs.size()]);
        (void)h;
    }
    double s = t.elapsed_s();
    auto st = pool.stats();
    return {"StringPool: intern (dup/hot)", OPS, s,
            st.arena_bytes + st.table_bytes};
}

static bench::Result bench_naive_intern_dup(const std::vector<std::string>& strs) {
    NaiveStringPool pool;
    for (auto& s : strs) pool.intern(s);

    bench::Timer t;
    t.start();
    for (size_t i = 0; i < OPS; ++i) {
        auto id = pool.intern(strs[i % strs.size()]);
        (void)id;
    }
    return {"NaivePool:  intern (dup/hot)", OPS, t.elapsed_s(), pool.mem_estimate()};
}

// ── 3. lookup ─────────────────────────────────────────────────────────────────

static bench::Result bench_sp_lookup(const std::vector<std::string>& strs) {
    sp::StringPool pool(512);
    for (auto& s : strs) pool.intern(s);

    bench::Timer t;
    t.start();
    size_t found = 0;
    for (size_t i = 0; i < OPS; ++i) {
        if (pool.lookup(strs[i % strs.size()])) ++found;
    }
    double s = t.elapsed_s();
    auto st = pool.stats();
    (void)found;
    return {"StringPool: lookup",           OPS, s,
            st.arena_bytes + st.table_bytes};
}

static bench::Result bench_naive_lookup(const std::vector<std::string>& strs) {
    NaiveStringPool pool;
    for (auto& s : strs) pool.intern(s);

    bench::Timer t;
    t.start();
    size_t found = 0;
    for (size_t i = 0; i < OPS; ++i) {
        auto sv = pool.get(pool.intern(strs[i % strs.size()]));
        if (!sv.empty()) ++found;
    }
    (void)found;
    return {"NaivePool:  lookup",           OPS, t.elapsed_s(), pool.mem_estimate()};
}

// ── 4. get (handle → string_view) ────────────────────────────────────────────

static bench::Result bench_sp_get(const std::vector<std::string>& strs) {
    sp::StringPool pool(512);
    std::vector<sp::StringHandle> handles;
    handles.reserve(strs.size());
    for (auto& s : strs) handles.push_back(pool.intern(s));

    bench::Timer t;
    t.start();
    size_t total_len = 0;
    for (size_t i = 0; i < OPS; ++i) {
        total_len += pool.get(handles[i % handles.size()]).size();
    }
    (void)total_len;
    auto st = pool.stats();
    return {"StringPool: get (handle→sv)",  OPS, t.elapsed_s(),
            st.arena_bytes + st.table_bytes};
}

// ── 5. mixed (50% new + 50% dup) ─────────────────────────────────────────────

static bench::Result bench_sp_mixed(const std::vector<std::string>& strs,
                                    const std::vector<std::string>& dups) {
    sp::StringPool pool(512);
    bench::Timer t;
    t.start();
    for (size_t i = 0; i < OPS; ++i) {
        auto h = (i & 1) ? pool.intern(strs[i % strs.size()])
                         : pool.intern(dups[i % dups.size()]);
        (void)h;
    }
    double s = t.elapsed_s();
    auto st = pool.stats();
    return {"StringPool: mixed (50/50)",    OPS, s,
            st.arena_bytes + st.table_bytes};
}

static bench::Result bench_naive_mixed(const std::vector<std::string>& strs,
                                       const std::vector<std::string>& dups) {
    NaiveStringPool pool;
    bench::Timer t;
    t.start();
    for (size_t i = 0; i < OPS; ++i) {
        auto id = (i & 1) ? pool.intern(strs[i % strs.size()])
                          : pool.intern(dups[i % dups.size()]);
        (void)id;
    }
    return {"NaivePool:  mixed (50/50)",    OPS, t.elapsed_s(), pool.mem_estimate()};
}

// ── 6. multi-thread intern ────────────────────────────────────────────────────

static bench::Result bench_sp_mt_intern(const std::vector<std::string>& strs) {
    const int NUM_THREADS = 8;
    const size_t OPS_PER  = OPS / NUM_THREADS;

    sp::StringPool pool(512);
    std::vector<std::thread> threads;

    bench::Timer t;
    t.start();
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&, i]() {
            for (size_t j = 0; j < OPS_PER; ++j) {
                auto h = pool.intern(strs[(i * OPS_PER + j) % strs.size()]);
                (void)h;
            }
        });
    }
    for (auto& th : threads) th.join();
    double s = t.elapsed_s();
    auto st = pool.stats();
    return {"StringPool: MT intern (8T)",   OPS, s,
            st.arena_bytes + st.table_bytes};
}

static bench::Result bench_naive_mt_intern(const std::vector<std::string>& strs) {
    const int NUM_THREADS = 8;
    const size_t OPS_PER  = OPS / NUM_THREADS;

    NaiveStringPool pool;
    std::vector<std::thread> threads;

    bench::Timer t;
    t.start();
    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.emplace_back([&, i]() {
            for (size_t j = 0; j < OPS_PER; ++j) {
                auto id = pool.intern(strs[(i * OPS_PER + j) % strs.size()]);
                (void)id;
            }
        });
    }
    for (auto& th : threads) th.join();
    return {"NaivePool:  MT intern (8T)",   OPS, t.elapsed_s(), pool.mem_estimate()};
}

// ═══════════════════════════════════════════════════════════════════════════════
// main
// ═══════════════════════════════════════════════════════════════════════════════

int main() {
    std::printf("╔══════════════════════════════════════════════════════════╗\n");
    std::printf("║          StringPool  –  Performance Benchmark           ║\n");
    std::printf("║          %10zu operations per scenario              ║\n", OPS);
    std::printf("╚══════════════════════════════════════════════════════════╝\n");

    // Generate test data
    const size_t UNIQUE = 1'000'000;       // 1 M unique strings
    const size_t DUP    = 1'000;           // 1 K strings for dedup scenario
    auto unique_strs = gen_strings(UNIQUE, 4, 32);
    auto dup_strs    = gen_strings(DUP,    4, 16);

    bench::print_header();

    // ── intern unique
    auto r1a = bench_sp_intern_unique(unique_strs);
    auto r1b = bench_naive_intern_unique(unique_strs);
    bench::print_row(r1a);
    bench::print_row(r1b);
    bench::print_separator();

    // ── intern dup
    auto r2a = bench_sp_intern_dup(dup_strs);
    auto r2b = bench_naive_intern_dup(dup_strs);
    bench::print_row(r2a);
    bench::print_row(r2b);
    bench::print_separator();

    // ── lookup
    auto r3a = bench_sp_lookup(dup_strs);
    auto r3b = bench_naive_lookup(dup_strs);
    bench::print_row(r3a);
    bench::print_row(r3b);
    bench::print_separator();

    // ── get
    auto r4 = bench_sp_get(dup_strs);
    bench::print_row(r4);
    bench::print_separator();

    // ── mixed
    auto r5a = bench_sp_mixed(unique_strs, dup_strs);
    auto r5b = bench_naive_mixed(unique_strs, dup_strs);
    bench::print_row(r5a);
    bench::print_row(r5b);
    bench::print_separator();

    // ── multi-thread
    auto r6a = bench_sp_mt_intern(dup_strs);
    auto r6b = bench_naive_mt_intern(dup_strs);
    bench::print_row(r6a);
    bench::print_row(r6b);
    bench::print_separator();

    // ── Summary: speedup ratios
    std::printf("\n%-40s  %12s  %12s  %8s\n",
                "Scenario", "StringPool", "NaivePool", "Speedup");
    std::printf("%s\n", std::string(80, '-').c_str());

    auto ratio = [](const bench::Result& a, const bench::Result& b) {
        return b.ops_per_sec() > 0 ? a.ops_per_sec() / b.ops_per_sec() : 0.0;
    };

    auto fmt_ops = [](double v) {
        return bench::si(v) + "ops/s";
    };

    struct Pair { std::string name; bench::Result a, b; };
    std::vector<Pair> pairs = {
        {"intern unique", r1a, r1b},
        {"intern dup",    r2a, r2b},
        {"lookup",        r3a, r3b},
        {"mixed",         r5a, r5b},
        {"MT intern 8T",  r6a, r6b},
    };
    for (auto& p : pairs) {
        std::printf("%-40s  %12s  %12s  %7.2fx\n",
                    p.name.c_str(),
                    fmt_ops(p.a.ops_per_sec()).c_str(),
                    fmt_ops(p.b.ops_per_sec()).c_str(),
                    ratio(p.a, p.b));
    }

    // ── Memory comparison
    std::printf("\n%-40s  %12s  %12s\n", "Memory usage", "StringPool", "NaivePool");
    std::printf("%s\n", std::string(80, '-').c_str());
    std::printf("%-40s  %12s  %12s\n",
                "intern unique (1M strings)",
                bench::fmt_bytes(r1a.mem_bytes).c_str(),
                bench::fmt_bytes(r1b.mem_bytes).c_str());

    return 0;
}
