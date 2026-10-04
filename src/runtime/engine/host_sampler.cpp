#include "runtime/engine/host_sampler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace ninfer::runtime {
namespace {

// The exact candidate domain of ops::sample(); resolved top_k values lie in [1,20].
constexpr int kCandidateCap = 20;

constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31U);
}

// Candidate order: higher adjusted logit first, lower token id on ties.
bool better(float value, TokenId id, float other_value, TokenId other_id) noexcept {
    return value > other_value || (value == other_value && id < other_id);
}

} // namespace

HostSampler::HostSampler(const ResolvedSamplingParameters& parameters, std::uint32_t token_domain)
    : parameters_(parameters), token_domain_(token_domain),
      penalties_(parameters.presence_penalty != 0.0F || parameters.frequency_penalty != 0.0F) {
    if (token_domain == 0) { throw std::invalid_argument("sampling token domain must be nonzero"); }
    if (penalties_) { counts_.assign(token_domain, 0); }
}

void HostSampler::commit(TokenId token) {
    if (token < 0 || static_cast<std::uint32_t>(token) >= token_domain_) {
        throw std::out_of_range("committed token is outside the sampling domain");
    }
    if (penalties_) { ++counts_[static_cast<std::size_t>(token)]; }
}

float HostSampler::uniform(std::uint64_t seed, std::int32_t position,
                           std::int32_t purpose) noexcept {
    std::uint64_t key = seed;
    key = splitmix64(key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(position)) *
                            0xD1B54A32D192ED03ULL));
    key =
        splitmix64(key ^ (static_cast<std::uint64_t>(static_cast<std::uint32_t>(purpose)) << 21U));
    const auto bits = static_cast<std::uint32_t>(key >> 40U); // 24 bits
    return static_cast<float>(bits) * (1.0F / 16777216.0F);
}

float HostSampler::adjusted(std::span<const float> logits, TokenId token) const noexcept {
    float value = logits[static_cast<std::size_t>(token)];
    if (!penalties_) { return value; }
    const std::uint32_t count = counts_[static_cast<std::size_t>(token)];
    if (count > 0) { value -= parameters_.presence_penalty; }
    if (parameters_.frequency_penalty != 0.0F) {
        value -= parameters_.frequency_penalty * static_cast<float>(count);
    }
    return value;
}

TokenId HostSampler::sample(std::span<const float> logits, std::int32_t position,
                            Purpose purpose) const {
    if (logits.size() < token_domain_) {
        throw std::invalid_argument("logits do not cover the sampling token domain");
    }
    const auto domain = static_cast<TokenId>(token_domain_);
    if (!(parameters_.temperature > 0.0F)) {
        TokenId best     = 0;
        float best_value = adjusted(logits, 0);
        for (TokenId token = 1; token < domain; ++token) {
            const float value = adjusted(logits, token);
            if (value > best_value) {
                best       = token;
                best_value = value;
            }
        }
        return best;
    }

    int cap = kCandidateCap;
    if (parameters_.top_k > 0 && parameters_.top_k < cap) { cap = parameters_.top_k; }
    cap = std::min(cap, static_cast<int>(std::min<std::uint32_t>(token_domain_, kCandidateCap)));
    std::array<float, kCandidateCap> values{};
    std::array<TokenId, kCandidateCap> ids{};
    int size = 0;
    for (TokenId token = 0; token < domain; ++token) {
        const float value = adjusted(logits, token);
        int position_in_list;
        if (size < cap) {
            position_in_list = size++;
        } else if (better(value, token, values[cap - 1], ids[cap - 1])) {
            position_in_list = cap - 1;
        } else {
            continue;
        }
        while (position_in_list > 0 &&
               better(value, token, values[position_in_list - 1], ids[position_in_list - 1])) {
            values[position_in_list] = values[position_in_list - 1];
            ids[position_in_list]    = ids[position_in_list - 1];
            --position_in_list;
        }
        values[position_in_list] = value;
        ids[position_in_list]    = token;
    }

    // Weights relative to the best candidate; min-p and top-p truncate a prefix of the ordered
    // candidates against the pre-truncation weight, keeping at least the best one.
    const float inverse_temperature = 1.0F / parameters_.temperature;
    const float maximum             = values[0] * inverse_temperature;
    std::array<float, kCandidateCap> weights{};
    float total = 0.0F;
    for (int j = 0; j < size; ++j) {
        weights[j] = std::exp(values[j] * inverse_temperature - maximum);
        total += weights[j];
    }
    const float min_p_threshold = parameters_.min_p > 0.0F ? parameters_.min_p * weights[0] : -1.0F;
    const bool top_p_active     = parameters_.top_p < 1.0F;
    const float top_p_target    = parameters_.top_p * total;
    float cumulative            = 0.0F;
    int support                 = 0;
    for (int j = 0; j < size; ++j) {
        if (min_p_threshold >= 0.0F && weights[j] < min_p_threshold) { break; }
        cumulative += weights[j];
        support = j + 1;
        if (top_p_active && cumulative >= top_p_target) { break; }
    }
    support             = std::max(support, 1);
    float support_total = 0.0F;
    for (int j = 0; j < support; ++j) { support_total += weights[j]; }
    const float inverse_support = 1.0F / support_total;
    const float u     = uniform(parameters_.seed, position, static_cast<std::int32_t>(purpose));
    float accumulated = 0.0F;
    for (int j = 0; j < support; ++j) {
        accumulated += weights[j] * inverse_support;
        if (u < accumulated) { return ids[j]; }
    }
    return ids[support - 1];
}

} // namespace ninfer::runtime
