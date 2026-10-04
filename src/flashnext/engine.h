// Qwen3.8-Flash-Next inference engine for one RTX 4090 and a 16-core AVX-512 CPU.
//
// The dense part of every layer (hyper-connections, Gated DeltaNet or attention, router, shared
// expert) and the output head run on the GPU with FP32 activations; the routed experts run on the
// CPU (CpuExperts). Up to four tokens are processed per step, which serves decoding and MTP
// verification; longer inputs are fed four tokens at a time.
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
    std::int64_t cached_experts = 0;
    double cache_gib = 0;
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
