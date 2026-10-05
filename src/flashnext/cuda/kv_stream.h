// Where attention K/V rows live: the store targets of attn_prep, and KV streaming for the QSA layers.
//
// KV streaming (docs/maintainer/flash-next-engine.md, "Long context: KV streaming"). A streamed attention
// layer keeps its authoritative fp16 K/V in pinned, device-mapped host memory, in the layout a fully
// resident cache has ([max_ctx][2 kv heads][256]). VRAM holds a page cache of n_slots pages of 4 cells (one
// QSA block, K and V: 8 KiB) and a page table block -> slot (or -1):
//   - writers (attn_prep) store every row in the host copy, and in its slot when the block is resident, so
//     slots never go stale: nothing is invalidated after a rejected draft, a rollback or a restore;
//   - after the selection and before the attention, kv_resolve makes every block the selection names
//     resident (a CLOCK sweep that never evicts a block this call uses), rewrites the selected cells into
//     rows of the page pool, and kv_copy_misses reads the missed blocks from the host copy over PCIe;
//   - attn_sparse then reads the pool exactly as it reads a resident cache: the same fp16 values in the
//     same order, so attention is bitwise identical to the resident engine's.
// All of it reads positions from device memory and has fixed addresses, so it is captured in the decode
// graphs. Prompt chunks that reach beyond the page cache stage the layer's rows into a full-context pool
// instead (the engine does that with plain DMA).
//
// The resolve and copy kernels are adapted from Strata (github.com/Niko1221/Strata,
// src/kernels/cuda/kv_stream.cu), MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors.
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ninfer::flashnext::cuda {

// Where attn_prep stores the K/V rows ([2][256] fp16 per position) of a step's tokens.
struct KvStore {
    half * k = nullptr;  // row (pos % ring) of k and v; ring 0: row pos
    half * v = nullptr;
    std::int64_t ring = 0;
    half * k2 = nullptr;  // optional second copy, row pos (the host copy beside a staging pool)
    half * v2 = nullptr;
    const std::int32_t * page_table = nullptr;  // optional page cache: row slot * 4 + pos % 4 of kp / vp when
    half * kp = nullptr;                        // slot = page_table[pos / 4] >= 0
    half * vp = nullptr;
};

constexpr int kKvPage = 4;                // cells per page: one QSA block
constexpr int kKvCtlInts = 16;            // [0] epoch [1] clock hand [2] misses of the last call [3] overflow; u64 at [4] misses,
                                          // [6] lookups, [8] calls
constexpr int kKvResolveThreads = 1024;   // the resolve kernel's one CUDA block; also the fewest slots a map may have

// The page table and clock state of one streamed layer, all device memory at fixed addresses.
struct KvStreamMap {
    std::int32_t * page_table = nullptr;  // [n_blocks] block -> slot, -1 (or -2 while a resolve claims it)
    std::int32_t * slot_block = nullptr;  // [n_slots] slot -> block or -1
    std::int32_t * slot_stamp = nullptr;  // [n_slots] the epoch of the last resolve that used the slot
    std::int32_t * slot_ref = nullptr;    // [n_slots] clock reference bit
    std::int32_t * miss_block = nullptr;  // [n_slots] this call's misses and their slots
    std::int32_t * miss_slot = nullptr;
    std::int32_t * ctl = nullptr;         // [kKvCtlInts]
    std::int64_t n_blocks = 0, n_slots = 0;
};
// device bytes of a map's arrays (page table, five per-slot arrays, ctl)
std::size_t kv_map_bytes(std::int64_t n_blocks, std::int64_t n_slots);
// points m's arrays into mem (kv_map_bytes bytes, 16-byte aligned)
KvStreamMap kv_map_layout(void * mem, std::int64_t n_blocks, std::int64_t n_slots);
// every block evicted, counters zeroed
void kv_map_reset(const KvStreamMap & m, cudaStream_t s);

// Makes resident every block that the selections of T queries name (cells [T][width], n_cells [T]: sorted
// cells, as qsa_select writes them). With translate, rewrites those cells in place into rows of the page pool
// (slot * 4 + cell % 4); otherwise kv_translate does it. Capturable. A call must name at most n_slots
// distinct blocks; one that names more sets the overflow flag (ctl[3]) and leaves the excess unmapped.
void kv_resolve(const KvStreamMap & m, std::int32_t * cells, const std::int32_t * n_cells, int T, int width, bool translate,
                cudaStream_t s);
// The same for every block of positions [0, pos0 + T) (pos0 in device memory): a prompt chunk while the whole
// context fits in the page cache. Needs (pos0 + T + 3) / 4 <= n_slots.
void kv_resolve_prefix(const KvStreamMap & m, const std::int64_t * pos0, int T, cudaStream_t s);
// cells -> page-pool rows for T queries (after kv_resolve without translate, or kv_resolve_prefix)
void kv_translate(const KvStreamMap & m, std::int32_t * cells, const std::int32_t * n_cells, int T, int width, cudaStream_t s);
// Copies the blocks the last resolve missed from the host copy (device-mapped pointers, identity layout) into
// their slots of the pool. Capturable.
void kv_copy_misses(const KvStreamMap & m, const half * host_k, const half * host_v, half * pool_k, half * pool_v, cudaStream_t s);

struct KvStreamCounters {
    std::uint64_t misses = 0, lookups = 0, calls = 0;
    bool overflow = false;
};
KvStreamCounters kv_counters(const KvStreamMap & m);  // synchronous

}  // namespace ninfer::flashnext::cuda
