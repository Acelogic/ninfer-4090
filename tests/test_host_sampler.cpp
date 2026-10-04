// Contract qualification for the host sampler (runtime/engine/host_sampler.h), which mirrors
// ops::sample() for FP32 host logits.
//
// Greedy rows are checked exactly against an independent argmax. Stochastic rows are checked
// against one FP64 distribution oracle over many independent RNG keys (positions); the test does
// not reproduce the sampler's arithmetic.
#include "runtime/engine/host_sampler.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using ninfer::ResolvedSamplingParameters;
using ninfer::TokenId;
using ninfer::runtime::HostSampler;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

struct Distribution {
    std::vector<int> tokens;
    std::vector<double> probabilities;
};

double adjusted(const std::vector<float>& logits, const std::vector<int>& counts,
                const ResolvedSamplingParameters& parameters, int token) {
    const int count = counts[static_cast<std::size_t>(token)];
    double value    = logits[static_cast<std::size_t>(token)];
    if (count > 0) { value -= parameters.presence_penalty; }
    return value - static_cast<double>(parameters.frequency_penalty) * count;
}

int greedy_oracle(const std::vector<float>& logits, int token_domain,
                  const ResolvedSamplingParameters& parameters, const std::vector<int>& counts) {
    int best = 0;
    for (int token = 1; token < token_domain; ++token) {
        if (adjusted(logits, counts, parameters, token) >
            adjusted(logits, counts, parameters, best)) {
            best = token;
        }
    }
    return best;
}

Distribution distribution_oracle(const std::vector<float>& logits, int token_domain,
                                 const ResolvedSamplingParameters& parameters,
                                 const std::vector<int>& counts) {
    std::vector<std::pair<double, int>> candidates;
    for (int token = 0; token < token_domain; ++token) {
        candidates.emplace_back(adjusted(logits, counts, parameters, token), token);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    int cap = 20;
    if (parameters.top_k > 0 && parameters.top_k < 20) { cap = parameters.top_k; }
    cap = std::min(cap, token_domain);
    std::vector<double> weights(static_cast<std::size_t>(cap));
    double total = 0.0;
    for (int rank = 0; rank < cap; ++rank) {
        weights[static_cast<std::size_t>(rank)] =
            std::exp((candidates[static_cast<std::size_t>(rank)].first - candidates[0].first) /
                     parameters.temperature);
        total += weights[static_cast<std::size_t>(rank)];
    }
    double cumulative = 0.0;
    int support       = 0;
    for (int rank = 0; rank < cap; ++rank) {
        if (parameters.min_p > 0.0F &&
            weights[static_cast<std::size_t>(rank)] < parameters.min_p * weights[0]) {
            break;
        }
        cumulative += weights[static_cast<std::size_t>(rank)];
        support = rank + 1;
        if (parameters.top_p < 1.0F && cumulative >= parameters.top_p * total) { break; }
    }
    support     = std::max(support, 1);
    double kept = 0.0;
    for (int rank = 0; rank < support; ++rank) { kept += weights[static_cast<std::size_t>(rank)]; }
    Distribution out;
    for (int rank = 0; rank < support; ++rank) {
        out.tokens.push_back(candidates[static_cast<std::size_t>(rank)].second);
        out.probabilities.push_back(weights[static_cast<std::size_t>(rank)] / kept);
    }
    return out;
}

int verify_distribution(const char* label, const std::vector<int>& samples,
                        const Distribution& expected) {
    std::vector<int> observed(expected.tokens.size(), 0);
    for (const int token : samples) {
        const auto it = std::find(expected.tokens.begin(), expected.tokens.end(), token);
        if (it == expected.tokens.end()) {
            std::cerr << label << ": sampled token " << token << " outside the oracle support\n";
            return 1;
        }
        ++observed[static_cast<std::size_t>(it - expected.tokens.begin())];
    }
    const double n = static_cast<double>(samples.size());
    for (std::size_t i = 0; i < expected.tokens.size(); ++i) {
        const double probability = expected.probabilities[i];
        const double frequency   = static_cast<double>(observed[i]) / n;
        const double limit       = 7.0 * std::sqrt(probability * (1.0 - probability) / n) + 2.0 / n;
        if (std::abs(frequency - probability) > limit) {
            std::cerr << label << ": token " << expected.tokens[i] << " frequency " << frequency
                      << " oracle " << probability << '\n';
            return 1;
        }
    }
    return 0;
}

// Ties at the maximum resolve to the lowest id, rows past the token domain never participate,
// penalties apply to committed tokens, and filters and the RNG are ignored.
int greedy_contract() {
    constexpr int kPhysicalRows = 248320;
    constexpr int kTokenDomain  = 248077;
    std::vector<float> logits(kPhysicalRows, -9.0F);
    logits[1000]              = 16.0F;
    logits[2000]              = 16.0F;
    logits[17]                = 30.0F;
    logits[kTokenDomain]      = 100.0F;
    logits[kPhysicalRows - 1] = 200.0F;
    ResolvedSamplingParameters parameters{.temperature       = 0.0F,
                                          .top_k             = 1,
                                          .top_p             = 0.01F,
                                          .min_p             = 0.99F,
                                          .presence_penalty  = 10.0F,
                                          .frequency_penalty = 2.0F,
                                          .seed              = 12345};
    HostSampler sampler(parameters, kTokenDomain);
    std::vector<int> counts(kTokenDomain, 0);
    int failures = 0;
    failures += check(sampler.sample(logits, 77, HostSampler::Purpose::Decode) == 17,
                      "greedy did not select the maximum");
    sampler.commit(17); // 30 - 10 - 2 = 18 still wins
    ++counts[17];
    failures += check(sampler.sample(logits, 78, HostSampler::Purpose::Decode) ==
                          greedy_oracle(logits, kTokenDomain, parameters, counts),
                      "greedy penalty result differs from the oracle");
    sampler.commit(17); // 30 - 10 - 4 = 16 ties with 1000 and 2000: the lowest id wins
    ++counts[17];
    const TokenId tied = sampler.sample(logits, 79, HostSampler::Purpose::Decode);
    failures += check(tied == 17 && tied == greedy_oracle(logits, kTokenDomain, parameters, counts),
                      "greedy tie did not resolve to the lowest id");
    sampler.commit(17);
    ++counts[17];
    failures += check(sampler.sample(logits, 80, HostSampler::Purpose::Decode) == 1000,
                      "greedy did not move past the penalized token");
    return failures;
}

// top_k=1 at positive temperature is the adjusted argmax regardless of the draw.
int top1_contract() {
    std::vector<float> logits = {5.0F, 4.5F, 4.0F, 3.0F, -1.0F};
    ResolvedSamplingParameters parameters{
        .temperature = 0.8F, .top_k = 1, .presence_penalty = 0.5F, .seed = 9981};
    HostSampler sampler(parameters, 5);
    int failures = 0;
    failures +=
        check(sampler.sample(logits, 11, HostSampler::Purpose::Decode) == 0, "top-1 result");
    sampler.commit(0); // 4.5 ties with token 1
    failures += check(sampler.sample(logits, 12, HostSampler::Purpose::Decode) == 0,
                      "top-1 adjusted tie did not resolve to the lowest id");
    return failures;
}

int distribution_contract() {
    std::vector<float> logits(64, -20.0F);
    const float values[] = {3.0F,  2.6F,  2.5F,  2.2F,  2.0F,  1.7F,  1.5F,  1.0F,
                            0.5F,  0.0F,  -0.5F, -1.0F, -1.5F, -2.0F, -2.5F, -3.0F,
                            -3.5F, -4.0F, -4.5F, -5.0F, -5.5F, -6.0F};
    for (std::size_t i = 0; i < std::size(values); ++i) { logits[3 * i + 1] = values[i]; }

    struct Case {
        const char* label;
        ResolvedSamplingParameters parameters;
    };

    const Case cases[] = {
        {"top-20", {.temperature = 1.0F, .top_k = 20, .seed = 7}},
        {"top-5 temperature 0.6", {.temperature = 0.6F, .top_k = 5, .seed = 8}},
        {"top-p 0.8", {.temperature = 1.0F, .top_k = 20, .top_p = 0.8F, .seed = 9}},
        {"min-p 0.2", {.temperature = 0.9F, .top_k = 20, .min_p = 0.2F, .seed = 10}},
        {"penalties",
         {.temperature       = 1.0F,
          .top_k             = 20,
          .top_p             = 0.95F,
          .presence_penalty  = 1.5F,
          .frequency_penalty = 0.25F,
          .seed              = 11}},
    };
    int failures = 0;
    for (const Case& test_case : cases) {
        HostSampler sampler(test_case.parameters, 64);
        std::vector<int> counts(64, 0);
        if (test_case.parameters.presence_penalty != 0.0F) {
            for (const TokenId token : {1, 1, 4, 7}) {
                sampler.commit(token);
                ++counts[static_cast<std::size_t>(token)];
            }
        }
        std::vector<int> samples;
        for (std::int32_t position = 0; position < 200000; ++position) {
            samples.push_back(sampler.sample(logits, position, HostSampler::Purpose::Decode));
        }
        failures +=
            verify_distribution(test_case.label, samples,
                                distribution_oracle(logits, 64, test_case.parameters, counts));
        // The draw is a pure function of (seed, position, purpose).
        failures += check(sampler.sample(logits, 4242, HostSampler::Purpose::Decode) ==
                              sampler.sample(logits, 4242, HostSampler::Purpose::Decode),
                          "sampling is not a pure function of its key");
    }
    return failures;
}

int uniform_contract() {
    int failures   = 0;
    int same_draws = 0;
    for (std::int32_t position = 0; position < 1000; ++position) {
        const float prefill = HostSampler::uniform(5, position, 0);
        const float decode  = HostSampler::uniform(5, position, 1);
        failures += check(prefill >= 0.0F && prefill < 1.0F && decode >= 0.0F && decode < 1.0F,
                          "uniform draw outside [0,1)");
        same_draws += prefill == decode ? 1 : 0;
    }
    failures += check(same_draws < 5, "prefill and decode purposes share RNG draws");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += greedy_contract();
    failures += top1_contract();
    failures += distribution_contract();
    failures += uniform_contract();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
