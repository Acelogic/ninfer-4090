#pragma once

#include "ninfer/types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::runtime {

// Host-side twin of ops::sample() for backends whose logits arrive in host memory as FP32. It
// implements the same contract (include/ninfer/ops/sampling.h): presence and frequency penalties
// over the request's committed generated tokens, greedy minimum-id argmax at temperature <= 0,
// otherwise the top-20 (or top_k) candidates ordered by adjusted logit and then id, min-p and top-p
// truncation against the pre-truncation weight, and one draw from the counter-based RNG keyed by
// (seed, logical position, purpose). Only ids in [0,token_domain) participate.
class HostSampler {
public:
    // The same subkeys as ops::SamplePurpose: the first token after a prompt uses Prefill at the
    // prompt length, later tokens use Decode at the position of the token just executed.
    enum class Purpose : std::int32_t {
        Prefill = 0,
        Decode  = 1,
    };

    HostSampler(const ResolvedSamplingParameters& parameters, std::uint32_t token_domain);

    [[nodiscard]] TokenId sample(std::span<const float> logits, std::int32_t position,
                                 Purpose purpose) const;
    // Adds one committed generated token (sampled or forced) to the penalty counts.
    void commit(TokenId token);

    [[nodiscard]] static float uniform(std::uint64_t seed, std::int32_t position,
                                       std::int32_t purpose) noexcept;

private:
    [[nodiscard]] float adjusted(std::span<const float> logits, TokenId token) const noexcept;

    ResolvedSamplingParameters parameters_;
    std::uint32_t token_domain_ = 0;
    bool penalties_             = false;
    std::vector<std::uint32_t> counts_;
};

} // namespace ninfer::runtime
