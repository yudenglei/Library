/**
 * @file  bench_utils.hpp
 * @brief Lightweight benchmark utilities: timer, throughput, stats reporter.
 */
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace bench {

// ─── High-resolution wall-clock timer ────────────────────────────────────────

class Timer {
public:
    void start() noexcept {
        t0_ = std::chrono::high_resolution_clock::now();
    }
    /** Returns elapsed seconds since last start(). */
    double elapsed_s() const noexcept {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(t1 - t0_).count();
    }
    double elapsed_ms() const noexcept { return elapsed_s() * 1e3; }
    double elapsed_us() const noexcept { return elapsed_s() * 1e6; }

private:
    std::chrono::high_resolution_clock::time_point t0_;
};

// ─── Throughput helpers ───────────────────────────────────────────────────────

/** Format a large number with SI suffix: 12345678 → "12.35 M" */
inline std::string si(double n) {
    char buf[64];
    if      (n >= 1e9) std::snprintf(buf, sizeof(buf), "%.2f G", n / 1e9);
    else if (n >= 1e6) std::snprintf(buf, sizeof(buf), "%.2f M", n / 1e6);
    else if (n >= 1e3) std::snprintf(buf, sizeof(buf), "%.2f K", n / 1e3);
    else               std::snprintf(buf, sizeof(buf), "%.2f  ", n);
    return buf;
}

/** Format bytes: 1048576 → "1.00 MiB" */
inline std::string fmt_bytes(size_t b) {
    char buf[64];
    if      (b >= (1u << 30)) std::snprintf(buf, sizeof(buf), "%.2f GiB", b / (double)(1u << 30));
    else if (b >= (1u << 20)) std::snprintf(buf, sizeof(buf), "%.2f MiB", b / (double)(1u << 20));
    else if (b >= (1u << 10)) std::snprintf(buf, sizeof(buf), "%.2f KiB", b / (double)(1u << 10));
    else                      std::snprintf(buf, sizeof(buf), "%zu  B",   b);
    return buf;
}

// ─── Simple result structure ──────────────────────────────────────────────────

struct Result {
    std::string label;
    size_t      ops;
    double      seconds;
    size_t      mem_bytes{0};   // optional memory usage

    double ops_per_sec() const noexcept {
        return seconds > 0 ? static_cast<double>(ops) / seconds : 0;
    }
};

// ─── Table printer ────────────────────────────────────────────────────────────

inline void print_header() {
    std::printf("\n%-40s  %12s  %10s  %12s\n",
                "Benchmark", "ops/s", "time(ms)", "memory");
    std::printf("%s\n", std::string(80, '-').c_str());
}

inline void print_row(const Result& r) {
    std::printf("%-40s  %12s  %10.1f  %12s\n",
                r.label.c_str(),
                si(r.ops_per_sec()).c_str(),
                r.seconds * 1e3,
                r.mem_bytes ? fmt_bytes(r.mem_bytes).c_str() : "-");
}

inline void print_separator() {
    std::printf("%s\n", std::string(80, '-').c_str());
}

}  // namespace bench
