// Qwen3.8-Flash-Next inference engine for one RTX 4090 and a 16-core AVX-512 CPU.
//
// The dense part of every layer (hyper-connections, Gated DeltaNet or attention, router, shared
// expert) and the output head run on the GPU with FP32 activations; the routed experts run on the
// CPU (CpuExperts). Steps of up to four tokens (decoding, MTP verification) run as one CUDA graph
// each; longer inputs are processed in batched chunks with FP32 GEMMs.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "flashnext/gguf.h"

namespace ninfer::flashnext {

struct EngineOptions {
    std::int64_t max_ctx = 32768;         // KV cache length
    int cpu_threads = 16;                 // expert threads
    bool precise_cpu_experts = true;      // 16-bit activations for the CPU experts (5e-5 error instead of 1.3%)
    bool pin_cpu_threads = false;         // pin the expert threads to physical cores (faster big prompt chunks, 3-6% slower decode)
    std::int64_t expert_cache_mib = -1;   // VRAM for routed experts; -1: all that is free but the reserve
    std::int64_t vram_reserve_mib = 1536; // left free for the desktop and other programs
    std::string routing_stats;            // per-layer expert counts that choose the cached experts ("" = none)
    bool cuda_graphs = true;              // replay each step as one CUDA graph (off: launch kernels one by one)
    // Prompts (more than 4 tokens) run in chunks of prefill_chunk tokens; 0 = automatic: per prompt, the
    // largest chunk (256-token grid, up to prefill_chunk_max) whose buffers fit in the VRAM the expert cache
    // can lend. Chunks above 512 tokens borrow their buffers from the expert cache for the prompt's duration
    // (prefill_lend); with prefill_lend off, buffers for prefill_chunk tokens stay allocated (the old way).
    int prefill_chunk = 0;
    int prefill_chunk_max = 8192;
    bool prefill_lend = true;
    // Chunks of at least prefill_stream_min tokens compute every routed expert on the GPU, copying the ones
    // that are not cached from the CPU's pinned copy as the layers run; smaller ones split the experts
    // between the GPU (cached) and the CPU (the rest).
    bool prefill_stream = true;
    int prefill_stream_min = 1024;
    // Streamed chunks of at most this many tokens leave the experts predicted to get the fewest tokens to the
    // CPU (they cost the CPU less RAM time than their copy costs PCIe time), concurrently with the GPU. 0: never
    // (the default: measured, it helped only 2K chunks at a 262K window, +14%, and slowed 3-4K chunks and 64K
    // windows by 6-11%). Those pairs use the CPU's numerics (16-bit activations, 5e-5) and the split depends on
    // the routing seen so far, so such chunks are not bitwise reproducible across histories.
    int prefill_cpu_share_max = 0;
    // Prompt chunks multiply the Q8_0 dense weights on the tensor cores (exact weights, activations as two fp16
    // terms, FP32 accumulation of the main terms) instead of dequantizing them for FP32 cuBLAS SGEMM: 1.6x
    // faster and, measured against double precision on the model's matrices, more accurate (1-4e-7 relative
    // against SGEMM's 3e-7 to 1.3e-6; test_gemm_tc).
    bool prefill_dense_tc = true;
    bool profile = false;                 // per-stage timings of prompt chunks on stderr (adds event records)
    // A pinned copy of every expert in the GPU's layout (about 55 GB of RAM, only if free): cache swaps
    // copy from it, and the GPU can read a share of each step's cache misses from it over PCIe. Off by
    // default: those PCIe reads come out of the same DRAM bandwidth the CPU experts need (measured: a
    // 30% share made decode 2.7x slower), and pinning 55 GB more puts the machine under memory pressure.
    bool host_expert_images = false;
    int gpu_miss_permille = 0;            // share of each step's cache misses the GPU reads from host memory
    std::string mtp_path;                 // the MTP head's GGUF (shared-Q8_0 variant) for draft tokens; "" = none
    // The MTP layer's 512 routed experts (Q8_0, 2.7 GB): false = computed by the CPU (CpuExperts, 16-bit activations, as
    // the main model's uncached experts) and the VRAM goes to the expert cache; true = all of them in VRAM.
    bool mtp_experts_vram = false;
    // Re-rank the expert cache after every decode step from the last steps' routing, swapping experts in the background
    // (needs the pinned CPU copy, i.e. prefill_lend and prefill_stream). Off: only after prompts and every 256 tokens.
    bool decode_adapt = true;
    // KV streaming (kv_cache.h): the attention layers' K/V live in pinned host memory and VRAM keeps a page
    // cache of kv_resident cells per layer, so a long window costs little VRAM. -1: on when max_ctx exceeds
    // kv_resident; 0: off (all of it in VRAM); 1: on.
    int kv_stream = -1;
    std::int64_t kv_resident = 32768;     // cells per attention layer kept in VRAM when streaming (>= 8448)
    int kv_group_tokens = 64;             // prompt chunks beyond the page cache up to max(this, depth / 512) tokens attend in groups
                                          // instead of staging (0: never)
    // Cells of the dedicated staging pool for prompt chunks beyond the page cache (-1: max_ctx; 0: none, such
    // chunks then fail). Used only while the engine cannot borrow that VRAM for the prompt instead.
    std::int64_t kv_stage_cells = -1;
};

// Named intermediate activations, row-major [n_tokens][width], with the same names and layout as
// ReferenceModel's hook so that the two can be compared directly.
using EngineHook = std::function<void(const std::string & name, int layer, std::int64_t pos0, std::int64_t n_tokens,
                                      std::int64_t width, const float * data)>;

struct EngineStats {
    std::int64_t steps = 0, tokens = 0;
    double step_ms = 0;         // wall time inside forward()
    double cpu_experts_ms = 0;  // of which the CPU expert calls
    double gpu_wait_ms = 0;     // decode steps: the CPU waiting for the GPU to reach each layer's experts
    double mtp_cpu_ms = 0;      // the MTP layer's routed experts on the CPU (drafts)
    double mtp_wait_ms = 0, mtp_tail_ms = 0, mtp_catchup_ms = 0;  // drafts: waiting for the GPU before and after them; catch-up
    double mtp_launch_ms = 0;   // drafts: launching the pass's graph
    double swap_issue_ms = 0;   // queueing the background swaps (host time)
    double adapt_ms = 0;        // choosing them after each decode step (host time)
    std::int64_t expert_pairs = 0, expert_hits = 0;  // selected experts, and those computed from the VRAM cache
    std::int64_t cpu_expert_reads = 0;               // decode steps: distinct experts the CPU computed (summed over layers)
    std::int64_t graph_steps[5] = {};                // decode steps by token count (1..4)
    std::int64_t expert_host_reads = 0;              // misses the GPU computed from pinned host memory
    std::int64_t cached_experts = 0;
    double cache_gib = 0;
    std::int64_t cache_swaps = 0;  // experts replaced in VRAM as the routing of recent tokens changed
    std::int64_t decode_swaps = 0; // of which swapped in the background between decode steps
    std::int64_t invalid_drafts = 0;  // MTP drafts dropped because the head's output was not a valid token
    double cache_swap_ms = 0;      // time spent replacing them
    // prompts
    std::int64_t prompt_chunks = 0, streamed_chunks = 0;
    int last_chunk = 0;                // tokens per chunk of the last prompt
    double lent_gib = 0;               // expert-cache VRAM the last prompt borrowed
    std::int64_t streamed_experts = 0; // experts copied from host memory for prompt chunks
    double streamed_gib = 0;
    double refill_ms = 0;              // putting experts back into the lent VRAM after prompts
    std::int64_t cpu_share_experts = 0; // experts of streamed chunks computed by the CPU instead
    int pinned_layers = 0;             // layers whose CPU copy is pinned (streamable)
};

// KV streaming (kv_cache.h): its configuration and counters.
struct KvStreamStats {
    bool enabled = false;
    std::int64_t resident_cells = 0;  // per layer
    std::int64_t layers = 0;
    double vram_gib = 0, host_gib = 0;
    std::uint64_t misses = 0, lookups = 0, resolves = 0;  // pages, over all layers since load
    std::uint64_t staged_chunks = 0, grouped_chunks = 0;  // prompt chunks beyond the page cache: staged, or in groups
    double staged_gib = 0;  // DMA'd into the staging pool
    bool overflow = false;
};

// Inputs of a forward() call beyond its token ids, for prompts with images.
struct ForwardInputs {
    // Rope positions of the call's n tokens, axis-major [3][n]: temporal, height, width (the model's interleaved
    // multi-axis rope; Qwen3-VL's layout: an image's tokens share the temporal position p and count rows from p in
    // height and columns from p in width; text has the same position in all three). Null: text positions that
    // continue after the largest position so far (next_position()).
    const std::int32_t * positions = nullptr;
    // Per token, the input embedding ([2560] floats: a vision encoder's output row) that replaces the token's own
    // embedding, or null. Such a token's id must be Engine::image_token_id(), which the PLE n-gram hash reads there.
    // A null array: every token embeds its id.
    const float * const * embeddings = nullptr;
};

// The recurrent state after a sequence of tokens: DeltaNet recurrent and conv states, the PLE conv
// history and the raw indexer keys of the incomplete QSA blocks, plus the tokens themselves. Attention keys and values (and indexer keys) stay in the
// engine's caches, by position, so a snapshot is valid while those positions still hold its tokens.
struct EngineSnapshot {
    std::vector<std::int32_t> tokens;
    std::vector<std::uint8_t> state;
};

class Engine {
public:
    Engine(const GgufModel & model, EngineOptions options = {});
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    // Consumes `tokens` at positions n_past() .. and returns the logits of the last one ([n_vocab]),
    // or of every new token ([size][n_vocab]) when all_logits is set.
    std::vector<float> forward(const std::vector<std::int32_t> & tokens, bool all_logits = false);
    // The same with rope positions and input embeddings (images). Without them, forward() continues text positions.
    std::vector<float> forward(const std::vector<std::int32_t> & tokens, bool all_logits, const ForwardInputs & inputs);
    // The rope position of the next text token: one past the largest position so far (n_past() until an image).
    std::int64_t next_position() const;
    // The token id that image positions carry (qwen4exp.ple.image_token_id), -1 if the model names none.
    std::int32_t image_token_id() const;
    // Lends `bytes` of expert-cache VRAM to work outside the model (the vision encoder): the experts there leave the
    // cache until the next forward() or draft() takes the memory back (a long prompt binds its buffers over it). The
    // engine's stream is idle when this returns. Returns its device address.
    void * lend_vram(std::size_t bytes);

    void reset();
    std::int64_t n_past() const;
    // The n_past() tokens processed so far.
    std::vector<std::int32_t> tokens() const;

    // About 116 MiB of state; a few milliseconds.
    EngineSnapshot snapshot() const;
    // Returns to a snapshot taken earlier. Valid when the caches still hold the snapshot's tokens at
    // their positions, i.e. nothing different was processed at those positions since (checked, with the
    // positions' rope positions and input embedding rows; throws otherwise). Typical use: snapshot at the
    // end of each prompt, restore it when the next request extends that prompt.
    void restore(const EngineSnapshot & snapshot);
    int n_vocab() const;
    void set_activation_hook(EngineHook hook);  // slow: synchronizes and copies every activation
    const EngineStats & stats() const;
    // Changes the prompt chunking for later forward() calls (EngineOptions::prefill_chunk, prefill_stream).
    void set_prefill(int chunk, bool stream);
    // Writes the routing counts (the loaded ones plus everything seen since) for the next start.
    void save_routing_stats(const std::string & path) const;

    // Speculative decoding with the model's MTP head (EngineOptions::mtp_path). draft() proposes k
    // tokens to follow `next`, the token the caller feeds next; verify them with
    // forward({next, drafts...}, true) and keep the accepted prefix with rollback(n), where n counts
    // `next` itself. The MTP layer's own cache is kept in step with the main model automatically.
    bool has_mtp() const;
    // KV streaming state and counters (synchronous; enabled = false when every layer's K/V is in VRAM)
    KvStreamStats kv_stream_stats() const;
    std::vector<std::int32_t> draft(std::int32_t next, int k);
    // Undoes all but the first n_keep tokens of the last forward() call, which must have had at most
    // 4 tokens (1 <= n_keep <= that count). Needs MTP to be enabled (it keeps the per-token states).
    void rollback(int n_keep);

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace ninfer::flashnext
