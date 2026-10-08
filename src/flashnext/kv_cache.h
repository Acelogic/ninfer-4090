// KV streaming for the engine's QSA attention layers (the kernels and the scheme: cuda/kv_stream.h).
//
// Per streamed layer: the authoritative fp16 K/V of every position in pinned, device-mapped host memory
// ([max_ctx][2][256] each, 1 KiB per position and tensor), a VRAM page cache of resident_cells cells
// (pages of 4) with its page table, and the clock state. The engine calls, per step and layer:
//   begin_step() once per step, before the first layer;
//   store()      where attn_prep writes the step's K/V rows;
//   attend()     instead of attn_sparse: the attention of the step's queries over their selected cells.
// Steps come in four kinds:
//   - decode steps (T <= kMaxTokens, captured in graphs): resolve the selected blocks into the page cache;
//   - prompt chunks that end within the page cache (pos0 + T <= resident_cells): make every block of
//     [0, pos0 + T) resident (it fits), then read the page cache;
//   - short prompt chunks beyond it (T <= max(group_tokens, pos0 / kGroupDepth)): the queries in groups
//     small enough that a group's selected blocks always fit the page cache, each resolved like a decode
//     step (staging costs a fixed DMA of the whole depth that a short chunk cannot hide);
//   - longer prompt chunks beyond it: stage the layer's rows [0, pos0) from the host copy into a
//     full-context pool (one layer at a time, by DMA on a side stream overlapped with the previous layers),
//     while attn_prep writes the chunk's rows into the stage, the host copy and any resident page.
// The attention reads the same fp16 values in the same order as from a fully resident cache (groups use the
// whole chunk's split count), so streamed and resident engines agree bit for bit.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flashnext/cuda/device.h"
#include "flashnext/cuda/kv_stream.h"
#include "flashnext/engine.h"

namespace ninfer::flashnext {

class KvStreamCache {
public:
    // resident_cells: the page cache per layer (a multiple of 4, at least kMinResident); stream: the engine's.
    // group_tokens: prompt chunks beyond the page cache up to this long (or pos0 / kGroupDepth if longer) run in
    // groups instead of staging; 0: never.
    KvStreamCache(std::int64_t max_ctx, std::int64_t resident_cells, int group_tokens, cudaStream_t stream);
    ~KvStreamCache();
    KvStreamCache(const KvStreamCache &) = delete;
    KvStreamCache & operator=(const KvStreamCache &) = delete;

    // Fewest resident cells: the distinct blocks of a decode step of kMaxTokens queries (4 x 514 pages) and at
    // least kKvResolveThreads pages.
    static constexpr std::int64_t kMinResident = 8448;
    // Prompt chunks of up to pos0 / kGroupDepth tokens (and at least group_tokens) attend in groups. Measured at
    // 250K depth: staging adds about 0.2 s per chunk (12 layers x 500 MB of DMA that a short chunk cannot
    // hide), groups about 0.25 ms per token over one attention call; they break even near 500 tokens.
    static constexpr std::int64_t kGroupDepth = 512;

    int add_layer();  // the next streamed layer's index
    std::int64_t resident_cells() const { return resident_; }
    // Bytes of the staging pool a prompt chunk ending at position pos_end - 1 needs (0: none).
    std::size_t stage_bytes(std::int64_t pos_end) const;

    // A step of T tokens at positions p0 .. p0+T-1 (p0 is used on the host for prompt chunks only; decode steps
    // are captured in graphs and must not depend on it). stage: a device region of stage_bytes(p0 + T) bytes,
    // needed by staged prompt chunks only.
    void begin_step(std::int64_t p0, int T, void * stage);
    cuda::KvStore store(int li) const;
    // attn_sparse for the step's T queries of layer li (q, gate, out: [T][24][256]; cells [T][kQsaWidth] and
    // n_cells as qsa_select wrote them, rewritten here; d_pos0: the step's first position in device memory).
    void attend(int li, const float * q, const float * gate, std::int32_t * cells, const std::int32_t * n_cells, const std::int64_t * d_pos0,
                float scale, float * work, float * out);
    // Synchronous: counters and the overflow flag (an overflow means a step named more blocks than the page
    // cache holds, which a legal resident size rules out; the engine throws on it).
    KvStreamStats stats() const;

    // Bytes of one position of K (or V) in a layer's host copy.
    static std::size_t row_bytes();
    // Positions [0, end) of layer li's host copy, K and V (Engine::park / unpark). The engine's stream must be idle:
    // steps write the host copy through device-mapped memory. After write_rows the page cache may hold other tokens
    // at those positions: call reset_pages() before the next step.
    void read_rows(int li, std::int64_t end, std::uint8_t * k, std::uint8_t * v) const;
    void write_rows(int li, std::int64_t end, const std::uint8_t * k, const std::uint8_t * v);
    // Empties every layer's page cache (later steps load the pages they need from the host copy again); this also
    // zeroes its counters.
    void reset_pages();

private:
    struct Layer;
    void issue_stage(int li);
    std::int64_t max_ctx_, resident_, n_slots_;
    int group_tokens_, group_;  // group_: queries per group, so that a group's blocks fit the page cache
    cudaStream_t stream_, copy_ = nullptr;
    cudaEvent_t ev_free_ = nullptr, ev_staged_ = nullptr;
    std::vector<std::unique_ptr<Layer>> layers_;
    // the current step
    std::int64_t p0_ = 0;
    int T_ = 0;
    enum class Mode { Decode, Prefix, Groups, Stage } mode_ = Mode::Decode;
    half * stage_k_ = nullptr;
    half * stage_v_ = nullptr;
    int staged_ = -1;  // the layer whose rows the staging pool holds (or is being filled with)
    std::uint64_t staged_chunks_ = 0, grouped_chunks_ = 0;
    double staged_bytes_ = 0;
};

}  // namespace ninfer::flashnext
