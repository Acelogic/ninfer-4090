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
    std::int64_t expert_cache_mib = -1;   // VRAM for routed experts; -1: all that is free but the reserve
    std::int64_t vram_reserve_mib = 1536; // left free for the desktop and other programs
    std::string routing_stats;            // per-layer expert counts that choose the cached experts ("" = none)
    bool cuda_graphs = true;              // replay each step as one CUDA graph (off: launch kernels one by one)
    int prefill_chunk = 512;              // prompt tokens per batched pass (above 4 tokens)
    // A pinned copy of every expert in the GPU's layout (about 55 GB of RAM, only if free): cache swaps
    // copy from it, and the GPU can read a share of each step's cache misses from it over PCIe. Off by
    // default: those PCIe reads come out of the same DRAM bandwidth the CPU experts need (measured: a
    // 30% share made decode 2.7x slower), and pinning 55 GB more puts the machine under memory pressure.
    bool host_expert_images = false;
    int gpu_miss_permille = 0;            // share of each step's cache misses the GPU reads from host memory
};

// Named intermediate activations, row-major [n_tokens][width], with the same names and layout as
// ReferenceModel's hook so that the two can be compared directly.
using EngineHook = std::function<void(const std::string & name, int layer, std::int64_t pos0, std::int64_t n_tokens,
                                      std::int64_t width, const float * data)>;

struct EngineStats {
    std::int64_t steps = 0, tokens = 0;
    double step_ms = 0;         // wall time inside forward()
    double cpu_experts_ms = 0;  // of which the CPU expert calls
    std::int64_t expert_pairs = 0, expert_hits = 0;  // selected experts, and those computed from the VRAM cache
    std::int64_t expert_host_reads = 0;              // misses the GPU computed from pinned host memory
    std::int64_t cached_experts = 0;
    double cache_gib = 0;
    std::int64_t cache_swaps = 0;  // experts replaced in VRAM as the routing of recent tokens changed
    double cache_swap_ms = 0;      // time spent replacing them
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

    void reset();
    std::int64_t n_past() const;
    // The n_past() tokens processed so far.
    std::vector<std::int32_t> tokens() const;

    // About 116 MiB of state; a few milliseconds.
    EngineSnapshot snapshot() const;
    // Returns to a snapshot taken earlier. Valid when the caches still hold the snapshot's tokens at
    // their positions, i.e. nothing different was processed at those positions since (checked; throws
    // otherwise). Typical use: snapshot at the end of each prompt, restore it when the next request
    // extends that prompt.
    void restore(const EngineSnapshot & snapshot);
    int n_vocab() const;
    void set_activation_hook(EngineHook hook);  // slow: synchronizes and copies every activation
    const EngineStats & stats() const;
    // Writes the routing counts (the loaded ones plus everything seen since) for the next start.
    void save_routing_stats(const std::string & path) const;

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace ninfer::flashnext
