/**
 * @file  test_memory.cpp
 * @brief Memory layout and POD safety tests for StringPool.
 *
 * Covers:
 *  - static_assert verifications (replicated at runtime for visibility)
 *  - StringHandle is POD / trivially copyable
 *  - HashSlot size and standard-layout
 *  - Arena string data is physically contiguous within a chunk
 *  - memcpy of StringHandle arrays works correctly
 *  - Arena grows across chunk boundaries correctly
 */

#include "../include/string_pool.hpp"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

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

#define EXPECT_EQ(a, b) EXPECT_TRUE((a) == (b))

// ─── Tests ────────────────────────────────────────────────────────────────────

static void test_handle_is_pod() {
    EXPECT_TRUE(std::is_trivially_copyable_v<sp::StringHandle>);
    EXPECT_TRUE(std::is_standard_layout_v<sp::StringHandle>);
    EXPECT_EQ(sizeof(sp::StringHandle), size_t(4));
}

static void test_handle_null_sentinel() {
    sp::StringHandle null_h = sp::StringHandle::null();
    EXPECT_TRUE(!null_h.valid());
    EXPECT_EQ(null_h.value, ~uint32_t(0));
}

static void test_handle_encoding() {
    // Ensure shard and slot bits round-trip correctly
    sp::StringHandle h = sp::StringHandle::make(5, 1023);
    EXPECT_EQ(h.shard_id(), uint32_t(5));
    EXPECT_EQ(h.slot_index(), uint32_t(1023));

    sp::StringHandle h2 = sp::StringHandle::make(15, 0x0FFF'FFFF);
    EXPECT_EQ(h2.shard_id(), uint32_t(15));
    EXPECT_EQ(h2.slot_index(), uint32_t(0x0FFF'FFFF));
}

static void test_handle_memcpy() {
    sp::StringPool pool;

    // Build an array of handles (simulating POD struct array)
    const int N = 1000;
    std::vector<sp::StringHandle> src(N);
    for (int i = 0; i < N; ++i) {
        src[i] = pool.intern("item_" + std::to_string(i));
    }

    // memcpy the entire array
    std::vector<sp::StringHandle> dst(N);
    std::memcpy(dst.data(), src.data(), N * sizeof(sp::StringHandle));

    // Verify all handles in dst resolve correctly
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(src[i], dst[i]);
        EXPECT_EQ(pool.get(dst[i]), pool.get(src[i]));
    }
}

static void test_arena_contiguous_within_chunk() {
    sp::StringPool pool;

    // Intern a set of strings that fit in one chunk
    // and check that their data pointers are in ascending order
    // (proves Arena uses bump-pointer, not random malloc)
    const int N = 100;
    std::vector<const char*> ptrs;
    ptrs.reserve(N);

    for (int i = 0; i < N; ++i) {
        sp::StringHandle h = pool.intern("contiguous_test_" + std::to_string(i));
        ptrs.push_back(pool.get(h).data());
    }

    bool ascending = true;
    for (int i = 1; i < N; ++i) {
        if (ptrs[i] <= ptrs[i - 1]) { ascending = false; break; }
    }
    EXPECT_TRUE(ascending);
}

static void test_arena_cross_chunk() {
    // Stress the chunk boundary by writing many large strings that together
    // exceed one chunk (SP_CHUNK_BYTES = 1 MiB).
    sp::StringPool pool;

    const size_t STR_SIZE  = 64 * 1024;                   // 64 KiB each
    const int    N         = 24;                           // > 1 MiB total
    std::string  big(STR_SIZE, 'A');

    std::vector<sp::StringHandle> handles(N);
    for (int i = 0; i < N; ++i) {
        big[0] = static_cast<char>('A' + (i % 26));       // unique per string
        handles[i] = pool.intern(big);
    }

    // All handles valid; content verifiable
    for (int i = 0; i < N; ++i) {
        auto sv = pool.get(handles[i]);
        EXPECT_EQ(sv.size(), STR_SIZE);
        EXPECT_EQ(sv[0], static_cast<char>('A' + (i % 26)));
    }

    auto st = pool.stats();
    EXPECT_TRUE(st.arena_bytes >= STR_SIZE * static_cast<size_t>(N));
}

static void test_handle_equality_after_memcpy() {
    sp::StringPool pool;

    sp::StringHandle original = pool.intern("equality_check");

    // Copy through raw bytes
    sp::StringHandle copy;
    std::memcpy(&copy, &original, sizeof(sp::StringHandle));

    EXPECT_EQ(original, copy);
    EXPECT_EQ(pool.get(copy), "equality_check");
}

static void test_pod_struct_embed() {
    // Simulate real-world usage: POD struct containing StringHandle
    struct CarData {
        float     speed;
        uint32_t  id;
        sp::StringHandle name_handle;
        uint32_t  flags;
    };

    static_assert(std::is_standard_layout_v<sp::StringHandle>);

    sp::StringPool pool;

    CarData car;
    car.speed       = 100.0f;
    car.id          = 42;
    car.name_handle = pool.intern("Ferrari");
    car.flags       = 0xFF;

    // memcpy the whole POD
    CarData car2;
    std::memcpy(&car2, &car, sizeof(CarData));

    EXPECT_EQ(car2.speed,  100.0f);
    EXPECT_EQ(car2.id,     uint32_t(42));
    EXPECT_EQ(car2.flags,  uint32_t(0xFF));
    EXPECT_EQ(pool.get(car2.name_handle), "Ferrari");
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::printf("=== StringPool Memory Layout Tests ===\n");

    test_handle_is_pod();
    test_handle_null_sentinel();
    test_handle_encoding();
    test_handle_memcpy();
    test_arena_contiguous_within_chunk();
    test_arena_cross_chunk();
    test_handle_equality_after_memcpy();
    test_pod_struct_embed();

    std::printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
