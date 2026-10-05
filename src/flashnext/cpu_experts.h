// Routed experts of Qwen3.8-Flash-Next on the CPU (Zen 4: AVX-512 BW + VNNI).
//
// At load, every expert is copied from the GGUF into RAM in a layout built for 64-weight vector loads.
// All conversions are lossless (they reorder bits, not values):
//   gate/up IQ3_S  -> Q4L  (4-bit codes, one byte shuffle decodes 64 weights; see quants.h)
//   gate/up IQ4_XS -> Q4X  (same nibble layout, 8-bit block scales)
//   down    IQ4_NL -> IQ4L (same nibbles and scales, chunked for 64-weight loads)
//   down    Q8_0   -> kept as is
// A call computes, for each token, the weighted sum of the selected experts that the caller assigns
// to the CPU. Tokens that share an expert share one pass over its weights.
//
// run() serves decode (up to 16 tokens, bound by RAM bandwidth). run_batch() serves prefill chunks:
// with hundreds of tokens per expert the work is bound by integer throughput, so it decodes each group
// of 16 weight rows once into an L1/L2 tile and runs a register-blocked VNNI kernel over all of the
// expert's tokens. Both quantize activations the same way and give the same results up to float
// summation order (8-bit activations; with 16-bit ones up to about 1e-5, see test_cpu_experts).
// Neither is reentrant: one call at a time per CpuExperts. export_expert() rebuilds an expert's
// original GGUF bytes from the resident copy.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "flashnext/gguf.h"
#include "flashnext/thread_pool.h"

namespace ninfer::flashnext {

struct CpuExpertsConfig {
    int threads = 16;
    std::vector<int> layers;  // empty = all
    // Pin pool thread t to one logical processor of physical core t (Windows; t = 0, the calling thread, only during
    // run_batch). Left to the scheduler, 16 busy threads often share SMT pairs while other cores idle: run_batch() on
    // 4096-token chunks measured 20% slower; decode (run) speed is unaffected.
    bool pin_threads = true;
    // 16-bit instead of 8-bit activations (x per 256, h per 32, as before), multiplied as int16 x int16 with exact int32
    // sums per block: the error against exact math drops from about 1.2% to 5e-5. Decode speed is unchanged (within
    // 1%), as is run_batch where RAM bounds it (up to about 1024 tokens with a GPU cache); compute-bound chunks are
    // slower, 4096 tokens 1.4x (GPU cache) to 1.6x (all on the CPU).
    bool precise_activations = true;
};

class CpuExperts {
public:
    static constexpr int kEmbd = 2560;
    static constexpr int kFF = 640;
    static constexpr int kExperts = 512;
    static constexpr int kUsed = 10;
    static constexpr int kMaxTokens = 16;
    static constexpr int kMaxBatchTokens = 8192;

    CpuExperts(const GgufModel & model, const CpuExpertsConfig & config);
    ~CpuExperts();

    // x: [n_tokens][2560]. ids, weights: [n_tokens][10]. on_cpu: [n_tokens][10], nonzero for the
    // (token, slot) pairs to compute here (nullptr = all). out: [n_tokens][2560], overwritten with
    // the weighted sum of those experts' outputs (zero where a token has none).
    void run(int layer, int n_tokens, const float * x, const std::int32_t * ids, const float * weights,
             const std::uint8_t * on_cpu, float * out);

    // Same contract as run(), for 1..kMaxBatchTokens tokens (prefill). Every expert with at least
    // one pair is read from RAM once per call. Scratch is allocated on the first call and grows to
    // the largest n_tokens seen: about 11 KB per token (19 KB with precise activations), plus 240 KB
    // (410 KB) per pool thread.
    void run_batch(int layer, int n_tokens, const float * x, const std::int32_t * ids, const float * weights,
                   const std::uint8_t * on_cpu, float * out);

    // Expert e of `layer` in the GGUF's own encoding, rebuilt bit for bit from the resident copy (every repack is
    // lossless): gate and up as kFF rows of row_bytes(type, kEmbd), down as kEmbd rows of row_bytes(type, kFF), i.e.
    // exactly the expert's slice of the GGUF tensors. A null pointer skips that matrix. Thread-safe; it only reads the
    // resident weights, so it may also run while run() or run_batch() runs.
    void export_expert(int layer, int e, std::uint8_t * gate, std::uint8_t * up, std::uint8_t * down) const;
    struct ExportSizes {
        std::size_t gate_up, down;  // bytes export_expert writes to gate (and to up), and to down
    };
    ExportSizes export_sizes(int layer) const;

    // A layer's resident copy as raw bytes, for streaming experts to the GPU: expert e is the range
    // [base + e * stride, + stride) holding gate (gate_bytes), up (gate_bytes) and down (down_bytes) in this class's
    // lossless repacks (gate/up Q4L or Q4X, down IQ4L or Q8_0; quants.h). base is page-aligned; the layer spans
    // `bytes` (kExperts * stride rounded up to a page). Valid for the lifetime of this object.
    struct HostLayer {
        const std::uint8_t * base = nullptr;
        std::size_t stride = 0, gate_bytes = 0, down_bytes = 0, bytes = 0;
        bool gate_q4x = false;  // gate/up: Q4X (from IQ4_XS), else Q4L (from IQ3_S)
        bool down_q8 = false;   // down: Q8_0 as in the GGUF, else IQ4L (from IQ4_NL)
    };
    HostLayer host_layer(int layer) const;

    bool precise_activations() const { return precise_; }
    bool has_layer(int layer) const;
    std::size_t resident_bytes() const { return resident_bytes_; }
    // Bytes of weights one expert of this layer streams per call (gate + up + down).
    std::size_t expert_bytes(int layer) const;
    const char * format_name(int layer) const;

    struct Layer;

private:
    template <bool Precise>
    void run_batch_impl(int layer, int n_tokens, const float * x, const std::int32_t * ids, const float * weights,
                        const std::uint8_t * on_cpu, float * out);

    std::vector<std::unique_ptr<Layer>> layers_;
    SpinPool pool_;
    bool precise_ = false;
    std::size_t resident_bytes_ = 0;
    std::uint64_t caller_affinity_ = 0;  // logical processor for the calling thread during run_batch (0 = not pinned)
    struct Scratch;
    std::unique_ptr<Scratch> scratch_;
    struct BatchScratch;
    std::unique_ptr<BatchScratch> batch_;
};

}  // namespace ninfer::flashnext
