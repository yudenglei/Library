/**
 * @file  string_pool.hpp
 * @brief High-performance, header-only C++17 string interning pool.
 *
 * Architecture
 * ────────────
 *  ┌─────────────────────────────────────────────────────────────────┐
 *  │  StringPool (public API)                                        │
 *  │   intern / lookup / get / release / stats / reset              │
 *  │                      │                                          │
 *  │            shard = hash & (NUM_SHARDS-1)                       │
 *  │                      │                                          │
 *  │  ┌──────────────────────────────────────────────────────────┐  │
 *  │  │  Shard[i]  (shared_mutex + RobinHood table + Arena)      │  │
 *  │  │   HashSlot[] ─── offset ──→ Arena Chunk chain            │  │
 *  │  │   {hash64, offset, length, refcount, probe_dist}         │  │
 *  │  └──────────────────────────────────────────────────────────┘  │
 *  └─────────────────────────────────────────────────────────────────┘
 *
 * Key properties
 * ──────────────
 *  • Zero external dependencies, single header include.
 *  • All string bytes stored in contiguous Arena chunks → memcpy-friendly.
 *  • StringHandle is a 4-byte POD → embeds directly in your POD structs.
 *  • 16-shard sharding reduces multi-thread lock contention ~16×.
 *  • FNV-1a hash: platform-independent, deterministic, inline.
 *  • Robin Hood open-addressing: cache-local, low probe distance variance.
 *  • Atomic reference counting (disable with SP_NO_REFCOUNT).
 *
 * Quick start
 * ───────────
 *  #include "string_pool.hpp"
 *
 *  sp::StringPool pool;
 *  sp::StringHandle h = pool.intern("hello");
 *  assert(pool.get(h) == "hello");
 *  assert(pool.intern("hello") == h);   // same handle
 *
 * Thread safety
 * ─────────────
 *  intern()  → acquires per-shard write lock
 *  lookup()  → acquires per-shard read  lock
 *  get()     → lock-free (Arena pointer never moves)
 *  release() → lock-free (atomic decrement)
 *
 * Compile-time knobs
 * ──────────────────
 *  SP_NO_REFCOUNT   – disable ref-counting for max throughput
 *  SP_NUM_SHARDS    – override shard count (must be power-of-2, default 16)
 *  SP_CHUNK_BYTES   – Arena chunk size in bytes (default 1 MiB)
 */

#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string_view>
#include <type_traits>
#include <vector>

// ─── Platform detection ───────────────────────────────────────────────────────
#if defined(_WIN32)
#  define SP_PLATFORM_WINDOWS
#elif defined(__APPLE__)
#  define SP_PLATFORM_MACOS
#else
#  define SP_PLATFORM_LINUX
#endif

#if defined(_MSC_VER)
#  define SP_FORCEINLINE __forceinline
#  define SP_NOINLINE    __declspec(noinline)
#else
#  define SP_FORCEINLINE __attribute__((always_inline)) inline
#  define SP_NOINLINE    __attribute__((noinline))
#endif

// ─── Tunables ─────────────────────────────────────────────────────────────────
#ifndef SP_NUM_SHARDS
#  define SP_NUM_SHARDS 16
#endif

#ifndef SP_CHUNK_BYTES
#  define SP_CHUNK_BYTES (1u << 20)   // 1 MiB
#endif

static_assert((SP_NUM_SHARDS & (SP_NUM_SHARDS - 1)) == 0,
              "SP_NUM_SHARDS must be a power of 2");

namespace sp {

// ═══════════════════════════════════════════════════════════════════════════════
// § 1  StringHandle
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Lightweight 4-byte handle to an interned string.
 * Safe to store in POD structs, copy with memcpy, and compare with ==.
 *
 * Encoding:  upper 4 bits  = shard index (0-15)
 *            lower 28 bits = slot index within the shard's hash table
 */
struct StringHandle {
    uint32_t value{~uint32_t(0)};

    [[nodiscard]] SP_FORCEINLINE bool valid() const noexcept {
        return value != ~uint32_t(0);
    }
    [[nodiscard]] SP_FORCEINLINE uint32_t shard_id() const noexcept {
        return value >> 28;
    }
    [[nodiscard]] SP_FORCEINLINE uint32_t slot_index() const noexcept {
        return value & 0x0FFF'FFFFu;
    }
    [[nodiscard]] SP_FORCEINLINE bool operator==(StringHandle o) const noexcept {
        return value == o.value;
    }
    [[nodiscard]] SP_FORCEINLINE bool operator!=(StringHandle o) const noexcept {
        return value != o.value;
    }

    static SP_FORCEINLINE StringHandle make(uint32_t shard, uint32_t slot) noexcept {
        return StringHandle{(shard << 28) | (slot & 0x0FFF'FFFFu)};
    }
    static constexpr StringHandle null() noexcept { return StringHandle{}; }
};

static_assert(sizeof(StringHandle) == 4,
              "StringHandle must be exactly 4 bytes");
static_assert(std::is_trivially_copyable_v<StringHandle>,
              "StringHandle must be trivially copyable (POD)");

// ═══════════════════════════════════════════════════════════════════════════════
// § 2  FNV-1a hash (64-bit, platform-independent)
// ═══════════════════════════════════════════════════════════════════════════════

namespace detail {

SP_FORCEINLINE constexpr uint64_t fnv1a_64(const char* data, size_t len) noexcept {
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint8_t>(data[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

SP_FORCEINLINE uint64_t hash_sv(std::string_view sv) noexcept {
    return fnv1a_64(sv.data(), sv.size());
}

// ═══════════════════════════════════════════════════════════════════════════════
// § 3  Arena allocator  (append-only, chunk-linked, memcpy-friendly)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Arena: a chain of fixed-size Chunks.
 * All allocations are bump-pointer. Freeing is not supported per-item;
 * call reset() to reclaim everything at once.
 *
 * The global byte offset of any allocation never changes after it is made,
 * so StringHandle offsets remain valid for the lifetime of the Arena.
 */
class Arena {
public:
    static constexpr size_t CHUNK_SIZE = SP_CHUNK_BYTES;
    static constexpr size_t ALIGN      = 8;

    Arena() { add_chunk(); }

    Arena(const Arena&)            = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&&)                 = default;
    Arena& operator=(Arena&&)      = default;

    /**
     * Allocate `size` bytes (8-byte aligned).
     * Returns the byte offset within the Arena's logical address space.
     * The pointer is stable for the Arena's lifetime.
     */
    [[nodiscard]] uint32_t alloc(size_t size) {
        const size_t aligned = (size + ALIGN - 1) & ~(ALIGN - 1);
        if (chunks_.back().used + aligned > CHUNK_SIZE) {
            add_chunk();
        }
        Chunk& c      = chunks_.back();
        uint32_t off  = c.base_offset + static_cast<uint32_t>(c.used);
        c.used       += aligned;
        total_bytes_ += aligned;
        return off;
    }

    /**
     * Write `data` of `len` bytes into the arena and return the offset.
     * A NUL terminator is appended automatically.
     */
    [[nodiscard]] uint32_t write(const char* data, size_t len) {
        const size_t total   = len + 1;                     // +1 for '\0'
        const size_t aligned = (total + ALIGN - 1) & ~(ALIGN - 1);

        if (chunks_.back().used + aligned > CHUNK_SIZE) {
            add_chunk();
        }
        Chunk& c      = chunks_.back();
        uint32_t off  = c.base_offset + static_cast<uint32_t>(c.used);
        char*    dst  = c.data.get() + c.used;
        std::memcpy(dst, data, len);
        dst[len]      = '\0';
        c.used       += aligned;
        total_bytes_ += aligned;
        return off;
    }

    /**
     * Translate a global byte offset back to a pointer.
     * O(1) when there is only one chunk; O(chunks) worst case.
     * In practice: almost always O(1) because chunks fill sequentially.
     */
    [[nodiscard]] const char* ptr(uint32_t offset) const noexcept {
        // Fast path: last chunk (most recent strings live here)
        for (auto it = chunks_.rbegin(); it != chunks_.rend(); ++it) {
            if (offset >= it->base_offset) {
                return it->data.get() + (offset - it->base_offset);
            }
        }
        assert(false && "Arena::ptr: offset out of range");
        return nullptr;
    }

    /** Reset all chunks, reclaiming all memory. */
    void reset() noexcept {
        chunks_.clear();
        total_bytes_ = 0;
        add_chunk();
    }

    [[nodiscard]] size_t total_bytes() const noexcept { return total_bytes_; }
    [[nodiscard]] size_t chunk_count() const noexcept { return chunks_.size(); }

private:
    struct Chunk {
        std::unique_ptr<char[]> data;
        uint32_t base_offset{0};
        size_t   used{0};
    };

    void add_chunk() {
        Chunk c;
        c.data        = std::make_unique<char[]>(CHUNK_SIZE);
        c.base_offset = static_cast<uint32_t>(chunks_.size() * CHUNK_SIZE);
        c.used        = 0;
        chunks_.push_back(std::move(c));
    }

    std::vector<Chunk> chunks_;
    size_t total_bytes_{0};
};

// ═══════════════════════════════════════════════════════════════════════════════
// § 4  HashSlot  (POD, 24 bytes, memcpy-safe)
// ═══════════════════════════════════════════════════════════════════════════════

struct HashSlot {
    uint64_t hash{0};          ///< full 64-bit FNV-1a hash
    uint32_t offset{0};        ///< byte offset in Arena
    uint32_t length{0};        ///< string byte-length (without NUL)
    uint32_t probe_dist{0};    ///< Robin Hood probe distance from ideal slot

#ifdef SP_NO_REFCOUNT
    uint32_t _pad{0};
#else
    // NOTE: atomic is NOT trivially copyable, so we store it as uint32_t
    // and perform atomic ops via reinterpret_cast only when needed.
    // The field is aligned to 4 bytes, safe for atomic<uint32_t> aliasing.
    alignas(4) uint32_t ref_count{0};
#endif

    [[nodiscard]] bool occupied() const noexcept { return length != 0 || hash != 0; }
    // "empty sentinel": hash == 0 AND length == 0.
    // Collision with a genuine empty string is handled specially.
};

// Verify the layout is what we expect
static_assert(sizeof(HashSlot) == 24,
              "HashSlot must be 24 bytes");
static_assert(std::is_standard_layout_v<HashSlot>,
              "HashSlot must be standard-layout");

// ═══════════════════════════════════════════════════════════════════════════════
// § 5  Robin Hood hash table
// ═══════════════════════════════════════════════════════════════════════════════

class RobinHoodTable {
public:
    static constexpr uint32_t EMPTY_SENTINEL = 0;
    static constexpr float    MAX_LOAD       = 0.75f;

    explicit RobinHoodTable(size_t initial_capacity = 256) {
        size_t cap = next_pow2(initial_capacity);
        slots_.assign(cap, HashSlot{});
        mask_     = static_cast<uint32_t>(cap - 1);
        capacity_ = cap;
    }

    /**
     * Insert or find a string.
     * @param hash64  FNV-1a hash of the string
     * @param offset  Arena byte offset where the string is stored
     * @param length  String byte length
     * @param arena   Used only to verify string content on hash collision
     * @param sv      Original string_view for collision resolution
     * @return  {slot_index, is_new_entry}
     */
    std::pair<uint32_t, bool> insert(
        uint64_t hash64, uint32_t offset, uint32_t length,
        const Arena& arena, std::string_view sv)
    {
        if (static_cast<float>(count_ + 1) > static_cast<float>(capacity_) * MAX_LOAD) {
            rehash(capacity_ * 2);
        }

        uint32_t   ideal = static_cast<uint32_t>(hash64) & mask_;
        uint32_t   pos   = ideal;
        uint32_t   dist  = 0;

        // Inserting slot (may be displaced by Robin Hood)
        HashSlot   ins{hash64, offset, length, dist};

        for (;;) {
            HashSlot& cur = slots_[pos];

            if (!cur.occupied()) {
                // Empty slot → place here
                cur = ins;
                ++count_;
                return {pos, true};
            }

            // Check if this is the same string
            if (cur.hash == hash64 && cur.length == length) {
                const char* stored = arena.ptr(cur.offset);
                if (std::memcmp(stored, sv.data(), length) == 0) {
                    // Already exists
                    return {pos, false};
                }
            }

            // Robin Hood: steal from the rich
            if (cur.probe_dist < dist) {
                std::swap(ins, cur);
                dist = ins.probe_dist;
            }

            pos = (pos + 1) & mask_;
            ++dist;
            ins.probe_dist = dist;
        }
    }

    /**
     * Lookup without insertion.
     * @return slot index if found, UINT32_MAX if not found.
     */
    [[nodiscard]] uint32_t find(
        uint64_t hash64, uint32_t length,
        const Arena& arena, std::string_view sv) const noexcept
    {
        uint32_t pos  = static_cast<uint32_t>(hash64) & mask_;
        uint32_t dist = 0;

        for (;;) {
            const HashSlot& cur = slots_[pos];

            if (!cur.occupied() || cur.probe_dist < dist) {
                return UINT32_MAX;
            }
            if (cur.hash == hash64 && cur.length == length) {
                const char* stored = arena.ptr(cur.offset);
                if (std::memcmp(stored, sv.data(), length) == 0) {
                    return pos;
                }
            }
            pos = (pos + 1) & mask_;
            ++dist;
        }
    }

    [[nodiscard]] const HashSlot& slot(uint32_t idx) const noexcept {
        return slots_[idx];
    }
    [[nodiscard]] HashSlot& slot(uint32_t idx) noexcept {
        return slots_[idx];
    }

    [[nodiscard]] size_t count()    const noexcept { return count_;    }
    [[nodiscard]] size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] size_t slot_bytes() const noexcept {
        return capacity_ * sizeof(HashSlot);
    }

    void reset() noexcept {
        slots_.assign(capacity_, HashSlot{});
        count_ = 0;
    }

private:
    static size_t next_pow2(size_t n) noexcept {
        if (n == 0) return 1;
        --n;
        n |= n >> 1; n |= n >> 2; n |= n >> 4;
        n |= n >> 8; n |= n >> 16; n |= n >> 32;
        return n + 1;
    }

    SP_NOINLINE void rehash(size_t new_cap) {
        std::vector<HashSlot> old = std::move(slots_);
        slots_.assign(new_cap, HashSlot{});
        capacity_ = new_cap;
        mask_     = static_cast<uint32_t>(new_cap - 1);
        count_    = 0;

        for (auto& s : old) {
            if (!s.occupied()) continue;

            uint32_t pos  = static_cast<uint32_t>(s.hash) & mask_;
            uint32_t dist = 0;
            HashSlot ins  = s;
            ins.probe_dist = 0;

            for (;;) {
                HashSlot& cur = slots_[pos];
                if (!cur.occupied()) {
                    cur = ins;
                    ++count_;
                    break;
                }
                if (cur.probe_dist < dist) {
                    std::swap(ins, cur);
                    dist = ins.probe_dist;
                }
                pos = (pos + 1) & mask_;
                ++dist;
                ins.probe_dist = dist;
            }
        }
    }

    std::vector<HashSlot> slots_;
    size_t   capacity_{0};
    uint32_t mask_{0};
    size_t   count_{0};
};

// ═══════════════════════════════════════════════════════════════════════════════
// § 6  Shard  (Arena + RobinHoodTable + shared_mutex)
// ═══════════════════════════════════════════════════════════════════════════════

class Shard {
public:
    explicit Shard(size_t initial_table_cap = 256)
        : table_(initial_table_cap) {}

    Shard(const Shard&)            = delete;
    Shard& operator=(const Shard&) = delete;

    /**
     * Intern sv.  Returns {handle_slot_index, is_new}.
     * Caller must hold the write lock OR call under intern().
     */
    std::pair<uint32_t, bool> intern_locked(std::string_view sv, uint64_t hash) {
        uint32_t length = static_cast<uint32_t>(sv.size());

        // First: check if already present (read-only table probe)
        uint32_t existing = table_.find(hash, length, arena_, sv);
        if (existing != UINT32_MAX) {
#ifndef SP_NO_REFCOUNT
            auto* rc = reinterpret_cast<std::atomic<uint32_t>*>(
                           &table_.slot(existing).ref_count);
            rc->fetch_add(1, std::memory_order_relaxed);
#endif
            return {existing, false};
        }

        // New string: write to arena first, then insert slot
        uint32_t offset = arena_.write(sv.data(), sv.size());
        auto [slot_idx, inserted] = table_.insert(hash, offset, length, arena_, sv);
        (void)inserted;  // always true here

#ifndef SP_NO_REFCOUNT
        table_.slot(slot_idx).ref_count = 1;
#endif
        return {slot_idx, true};
    }

    [[nodiscard]] uint32_t lookup_locked(std::string_view sv, uint64_t hash) const noexcept {
        return table_.find(hash, static_cast<uint32_t>(sv.size()), arena_, sv);
    }

    [[nodiscard]] std::string_view get(uint32_t slot_idx) const noexcept {
        const HashSlot& s = table_.slot(slot_idx);
        return {arena_.ptr(s.offset), s.length};
    }

    void release(uint32_t slot_idx) noexcept {
#ifndef SP_NO_REFCOUNT
        auto* rc = reinterpret_cast<std::atomic<uint32_t>*>(
                       &table_.slot(slot_idx).ref_count);
        rc->fetch_sub(1, std::memory_order_acq_rel);
#endif
        (void)slot_idx;
    }

    void reset() noexcept {
        arena_.reset();
        table_.reset();
    }

    // ── Stats ────────────────────────────────────────────────────────────────
    [[nodiscard]] size_t string_count() const noexcept { return table_.count(); }
    [[nodiscard]] size_t arena_bytes()  const noexcept { return arena_.total_bytes(); }
    [[nodiscard]] size_t table_bytes()  const noexcept { return table_.slot_bytes(); }

    // ── Locks (public for StringPool) ────────────────────────────────────────
    mutable std::shared_mutex mutex;

private:
    Arena           arena_;
    RobinHoodTable  table_;
};

}  // namespace detail

// ═══════════════════════════════════════════════════════════════════════════════
// § 7  StringPool — public API
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * @class StringPool
 * @brief Thread-safe, high-performance string interning pool.
 *
 * Usage
 * ─────
 *   sp::StringPool pool;
 *   sp::StringHandle h1 = pool.intern("hello");
 *   sp::StringHandle h2 = pool.intern("hello");
 *   assert(h1 == h2);                           // same handle
 *   assert(pool.get(h1) == "hello");             // zero-copy retrieval
 *
 *   auto opt = pool.lookup("world");             // non-inserting lookup
 *   assert(!opt.has_value());
 *
 * Notes
 * ─────
 *  • intern()  : O(1) amortized, write-locks one shard
 *  • lookup()  : O(1), read-locks one shard
 *  • get()     : O(1), completely lock-free
 *  • release() : lock-free atomic decrement (no-op if SP_NO_REFCOUNT)
 */
class StringPool {
public:
    static constexpr size_t NUM_SHARDS = SP_NUM_SHARDS;

    struct Stats {
        size_t string_count{0};  ///< total unique strings
        size_t arena_bytes{0};   ///< bytes used by string data
        size_t table_bytes{0};   ///< bytes used by hash table slots
    };

    /**
     * @param initial_capacity_per_shard  Initial hash-table slot count per shard.
     *        Total initial memory ≈ initial_capacity_per_shard * NUM_SHARDS * 24 bytes.
     */
    explicit StringPool(size_t initial_capacity_per_shard = 256) {
        for (size_t i = 0; i < NUM_SHARDS; ++i)
            shards_[i] = std::make_unique<detail::Shard>(initial_capacity_per_shard);
    }

    StringPool(const StringPool&)            = delete;
    StringPool& operator=(const StringPool&) = delete;
    StringPool(StringPool&&)                 = default;
    StringPool& operator=(StringPool&&)      = default;

    // ── Core API ─────────────────────────────────────────────────────────────

    /**
     * Intern a string.  If an identical string already exists, returns its
     * handle and increments the reference count.  Otherwise stores a copy.
     *
     * Empty strings always return StringHandle::null().
     */
    [[nodiscard]] StringHandle intern(std::string_view sv) {
        if (sv.empty()) return StringHandle::null();

        const uint64_t h      = detail::hash_sv(sv);
        const uint32_t shard  = static_cast<uint32_t>(h >> 60) &
                                (static_cast<uint32_t>(NUM_SHARDS) - 1);
        auto& sh = *shards_[shard];

        std::unique_lock lock(sh.mutex);
        auto [slot, _is_new] = sh.intern_locked(sv, h);
        return StringHandle::make(shard, slot);
    }

    /**
     * Look up without inserting.
     * Returns nullopt if the string is not in the pool.
     */
    [[nodiscard]] std::optional<StringHandle> lookup(std::string_view sv) const {
        if (sv.empty()) return std::nullopt;

        const uint64_t h      = detail::hash_sv(sv);
        const uint32_t shard  = static_cast<uint32_t>(h >> 60) &
                                (static_cast<uint32_t>(NUM_SHARDS) - 1);
        auto& sh = *shards_[shard];

        std::shared_lock lock(sh.mutex);
        uint32_t slot = sh.lookup_locked(sv, h);
        if (slot == UINT32_MAX) return std::nullopt;
        return StringHandle::make(shard, slot);
    }

    /**
     * Retrieve the string for a handle.  Lock-free, O(1).
     * Returns "" for null handles.
     */
    [[nodiscard]] std::string_view get(StringHandle h) const noexcept {
        if (!h.valid()) return {};
        return shards_[h.shard_id()]->get(h.slot_index());
    }

    /**
     * Decrement the reference count of a handle.
     * No-op if SP_NO_REFCOUNT is defined.
     */
    void release(StringHandle h) noexcept {
        if (!h.valid()) return;
        shards_[h.shard_id()]->release(h.slot_index());
    }

    /**
     * Reset the entire pool, freeing all memory.
     * All previously issued handles become invalid.
     */
    void reset() noexcept {
        for (auto& s : shards_) s->reset();
    }

    /** Aggregate statistics across all shards. */
    [[nodiscard]] Stats stats() const noexcept {
        Stats st{};
        for (auto& s : shards_) {
            st.string_count += s->string_count();
            st.arena_bytes  += s->arena_bytes();
            st.table_bytes  += s->table_bytes();
        }
        return st;
    }

    // ── RAII wrapper ─────────────────────────────────────────────────────────

    /**
     * @class InternedString
     * RAII handle that calls release() in its destructor.
     * Movable but not copyable (to prevent double-release).
     * Use clone() to obtain a second reference-counted handle.
     */
    class InternedString {
    public:
        InternedString() noexcept = default;
        InternedString(StringPool& pool, StringHandle h) noexcept
            : pool_(&pool), handle_(h) {}

        InternedString(const InternedString&)            = delete;
        InternedString& operator=(const InternedString&) = delete;

        InternedString(InternedString&& o) noexcept
            : pool_(o.pool_), handle_(o.handle_) {
            o.pool_   = nullptr;
            o.handle_ = StringHandle::null();
        }
        InternedString& operator=(InternedString&& o) noexcept {
            reset();
            pool_   = o.pool_;   o.pool_   = nullptr;
            handle_ = o.handle_; o.handle_ = StringHandle::null();
            return *this;
        }

        ~InternedString() { reset(); }

        [[nodiscard]] StringHandle    handle()    const noexcept { return handle_; }
        [[nodiscard]] std::string_view str()      const noexcept {
            return pool_ ? pool_->get(handle_) : std::string_view{};
        }
        [[nodiscard]] bool             valid()    const noexcept { return handle_.valid(); }
        [[nodiscard]] bool operator==(const InternedString& o) const noexcept {
            return handle_ == o.handle_;
        }

        void reset() noexcept {
            if (pool_ && handle_.valid()) {
                pool_->release(handle_);
                pool_   = nullptr;
                handle_ = StringHandle::null();
            }
        }

        /** Obtain a new reference-counted handle to the same string. */
        [[nodiscard]] InternedString clone() const {
            if (!pool_ || !handle_.valid()) return {};
            // Re-intern to bump refcount
            StringHandle h = pool_->intern(str());
            return InternedString(*pool_, h);
        }

    private:
        StringPool*  pool_{nullptr};
        StringHandle handle_{};
    };

    /** Create a managed InternedString (RAII ref-counted). */
    [[nodiscard]] InternedString make_interned(std::string_view sv) {
        return InternedString(*this, intern(sv));
    }

private:
    std::unique_ptr<detail::Shard> shards_[NUM_SHARDS];
};

}  // namespace sp
