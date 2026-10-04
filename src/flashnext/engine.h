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
    std::int64_t max_ctx = 32768;  // KV cache length
    int cpu_threads = 16;          // expert threads
};

// Named intermediate activations, row-major [n_tokens][width], with the same names and layout as
// ReferenceModel's hook so that the two can be compared directly.
using EngineHook = std::function<void(const std::string & name, int layer, std::int64_t pos0, std::int64_t n_tokens,
                                      std::int64_t width, const float * data)>;

struct EngineStats {
    std::int64_t steps = 0, tokens = 0;
    double step_ms = 0;         // wall time inside forward()
    double cpu_experts_ms = 0;  // of which the CPU expert calls
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

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace ninfer::flashnext
