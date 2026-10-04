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
};

class CpuExperts {
public:
    static constexpr int kEmbd = 2560;
    static constexpr int kFF = 640;
    static constexpr int kExperts = 512;
    static constexpr int kUsed = 10;
    static constexpr int kMaxTokens = 16;

    CpuExperts(const GgufModel & model, const CpuExpertsConfig & config);
    ~CpuExperts();

    // x: [n_tokens][2560]. ids, weights: [n_tokens][10]. on_cpu: [n_tokens][10], nonzero for the
    // (token, slot) pairs to compute here (nullptr = all). out: [n_tokens][2560], overwritten with
    // the weighted sum of those experts' outputs (zero where a token has none).
    void run(int layer, int n_tokens, const float * x, const std::int32_t * ids, const float * weights,
             const std::uint8_t * on_cpu, float * out);

    bool has_layer(int layer) const;
    std::size_t resident_bytes() const { return resident_bytes_; }
    // Bytes of weights one expert of this layer streams per call (gate + up + down).
    std::size_t expert_bytes(int layer) const;
    const char * format_name(int layer) const;

    struct Layer;

private:
    std::vector<std::unique_ptr<Layer>> layers_;
    SpinPool pool_;
    std::size_t resident_bytes_ = 0;
    struct Scratch;
    std::unique_ptr<Scratch> scratch_;
};

}  // namespace ninfer::flashnext
