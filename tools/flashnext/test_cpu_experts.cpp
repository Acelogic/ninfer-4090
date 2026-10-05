// Tests and benchmarks the CPU expert engine on the real model.
//   accuracy: one token, 10 random experts, against a double-precision reference built from the exact
//             dequantizers (the difference is the engine's 8-bit activations)
//   batching: 3 tokens sharing experts give the same results as 3 single-token calls
//   run_batch: prefill-sized calls (skewed routing, part of the pairs on the GPU) match run() per token,
//             and the double-precision reference
//   speed:    decode-shaped calls over several layers, projected to 48 layers; with --batch, prefill
//             chunks of the given sizes, all pairs on the CPU and with a GPU expert cache
//   All of the above for 8-bit and for 16-bit (precise) activations.
//   export:   export_expert reproduces the GGUF bytes of every expert of every loaded layer; its speed
// Usage: test_cpu_experts <shards...> [--layers 0,2,4,...] [--threads 16] [--tokens 40] [--batch 64,256,1024,4096]
//        [--no-pin] [--mode int8|precise|both] [--no-export] [--mtp <MTP head GGUF>: test its layer instead]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "flashnext/cpu_experts.h"
#include "flashnext/quants.h"

using namespace ninfer::flashnext;
using clk = std::chrono::steady_clock;

static std::vector<double> reference(const GgufModel & m, int il, const float * x, const int * ids, const float * w) {
    const std::string p = "blk." + std::to_string(il) + ".";
    const GgufTensor & tg = m.tensor(p + "ffn_gate_exps.weight");
    const GgufTensor & tu = m.tensor(p + "ffn_up_exps.weight");
    const GgufTensor & td = m.tensor(p + "ffn_down_exps.weight");
    const std::size_t gr = row_bytes(tg.type, 2560), dr = row_bytes(td.type, 640);
    std::vector<double> out(2560, 0.0);
    std::vector<float> row(2560), h(640);
    for (int k = 0; k < 10; ++k) {
        const int e = ids[k];
        for (int r = 0; r < 640; ++r) {
            double g = 0, u = 0;
            dequantize_row(tg.type, tg.data + (std::size_t(e) * 640 + r) * gr, row.data(), 2560);
            for (int j = 0; j < 2560; ++j) g += double(row[j]) * x[j];
            dequantize_row(tu.type, tu.data + (std::size_t(e) * 640 + r) * gr, row.data(), 2560);
            for (int j = 0; j < 2560; ++j) u += double(row[j]) * x[j];
            h[r] = float(g / (1.0 + std::exp(-g)) * u);
        }
        for (int r = 0; r < 2560; ++r) {
            dequantize_row(td.type, td.data + (std::size_t(e) * 2560 + r) * dr, row.data(), 640);
            double y = 0;
            for (int j = 0; j < 640; ++j) y += double(row[j]) * h[j];
            out[r] += double(w[k]) * y;
        }
    }
    return out;
}

// Routing for prefill tests: expert popularity follows a Zipf law (exponent 0.85) over a random order of the layer's
// 512 experts. The 138 most popular (27%) stand for a GPU expert cache; they receive about 70% of the selections, and
// on_cpu marks the rest.
struct Routing {
    std::vector<std::int32_t> ids;
    std::vector<float> w;
    std::vector<std::uint8_t> on_cpu;
};

static Routing skewed_routing(std::mt19937 & rng, int n_tokens) {
    std::vector<int> order(512);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<double> cdf(512);
    double s = 0;
    for (int r = 0; r < 512; ++r) cdf[r] = (s += std::pow(r + 1.0, -0.85));
    std::vector<std::uint8_t> cached(512, 0);
    for (int r = 0; r < 138; ++r) cached[order[r]] = 1;
    std::uniform_real_distribution<double> ud(0, s);
    Routing R;
    R.ids.resize(std::size_t(n_tokens) * 10);
    R.w.resize(R.ids.size());
    R.on_cpu.resize(R.ids.size());
    for (int t = 0; t < n_tokens; ++t) {
        std::int32_t * ids = R.ids.data() + 10 * t;
        for (int k = 0; k < 10;) {
            const int rank = std::min(511, int(std::lower_bound(cdf.begin(), cdf.end(), ud(rng)) - cdf.begin()));
            bool dup = false;
            for (int j = 0; j < k; ++j) dup |= ids[j] == order[rank];
            if (!dup) ids[k++] = order[rank];
        }
        float sum = 0;
        for (int k = 0; k < 10; ++k) sum += (R.w[10 * t + k] = 0.2f + float(rng() % 100) / 100.0f);
        for (int k = 0; k < 10; ++k) {
            R.w[10 * t + k] /= sum;
            R.on_cpu[10 * t + k] = !cached[ids[k]];
        }
    }
    return R;
}

// ||a - b|| / ||a||; 0 if both are zero.
static double rel_diff(const float * a, const float * b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) {
        num += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
        den += double(a[i]) * a[i];
    }
    if (den == 0) return num == 0 ? 0 : INFINITY;
    return std::sqrt(num / den);
}

// run() over n tokens, 16 at a time.
static void run_chunked(CpuExperts & ex, int il, int n, const float * x, const std::int32_t * ids, const float * w,
                        const std::uint8_t * on, float * out) {
    for (int t0 = 0; t0 < n; t0 += CpuExperts::kMaxTokens) {
        const int m = std::min(CpuExperts::kMaxTokens, n - t0);
        ex.run(il, m, x + std::size_t(t0) * 2560, ids + 10 * t0, w + 10 * t0, on ? on + 10 * t0 : nullptr, out + std::size_t(t0) * 2560);
    }
}

struct Options {
    std::vector<int> layers = {0, 2, 4};
    std::vector<int> batch_sizes;
    int threads = 16, tokens = 40;
    bool pin = true;
};

// export_expert: every expert of every loaded layer must reproduce the GGUF bytes; then its speed on 1 and on
// `threads` threads, in GGUF bytes written.
static int test_export(const GgufModel & m, const CpuExperts & ex, const Options & o) {
    int failures = 0;
    double bytes = 0, single = 0;
    for (int il : o.layers) {
        const CpuExperts::ExportSizes sz = ex.export_sizes(il);
        const std::string p = "blk." + std::to_string(il) + ".";
        const GgufTensor & tg = m.tensor(p + "ffn_gate_exps.weight");
        const GgufTensor & tu = m.tensor(p + "ffn_up_exps.weight");
        const GgufTensor & td = m.tensor(p + "ffn_down_exps.weight");
        std::vector<std::uint8_t> g(sz.gate_up), u(sz.gate_up), d(sz.down);
        int bad = 0;
        for (int e = 0; e < 512; ++e) {
            const auto a = clk::now();
            ex.export_expert(il, e, g.data(), u.data(), d.data());
            single += std::chrono::duration<double>(clk::now() - a).count();
            bytes += 2.0 * double(sz.gate_up) + double(sz.down);
            bad += std::memcmp(g.data(), tg.data + std::size_t(e) * sz.gate_up, sz.gate_up) != 0 ||
                   std::memcmp(u.data(), tu.data + std::size_t(e) * sz.gate_up, sz.gate_up) != 0 ||
                   std::memcmp(d.data(), td.data + std::size_t(e) * sz.down, sz.down) != 0;
        }
        failures += bad != 0;
        printf("layer %2d export_expert %s/%s: %d of 512 experts differ from the GGUF bytes  %s\n", il, type_name(tg.type), type_name(td.type), bad,
               bad ? "FAIL" : "ok");
    }
    std::atomic<int> next{0};
    const int jobs = int(o.layers.size()) * 512;
    const auto a = clk::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < o.threads; ++t)
        pool.emplace_back([&] {
            std::vector<std::uint8_t> g, u, d;
            for (int k = next++; k < jobs; k = next++) {
                const int il = o.layers[std::size_t(k / 512)];
                const CpuExperts::ExportSizes sz = ex.export_sizes(il);
                g.resize(sz.gate_up), u.resize(sz.gate_up), d.resize(sz.down);
                ex.export_expert(il, k % 512, g.data(), u.data(), d.data());
            }
        });
    for (auto & t : pool) t.join();
    const double multi = std::chrono::duration<double>(clk::now() - a).count();
    printf("export_expert: %.2f GB/s on 1 thread, %.1f GB/s on %d threads (%.2f MB per expert on average)\n", bytes / single / 1e9,
           bytes / multi / 1e9, o.threads, bytes / jobs / 1e6);
    return failures;
}

static int test_mode(const GgufModel & m, const Options & o, bool precise, bool do_export) {
    const std::vector<int> & layers = o.layers;
    const std::vector<int> & batch_sizes = o.batch_sizes;
    const int tokens = o.tokens;
    auto t0 = clk::now();
    CpuExperts ex(m, {o.threads, layers, o.pin, precise});
    printf("== %s activations: loaded %zu layers in %.1f s, %.2f GiB resident\n", precise ? "16-bit (precise)" : "8-bit", layers.size(),
           std::chrono::duration<double>(clk::now() - t0).count(), ex.resident_bytes() / 1073741824.0);
    // 8-bit activations give about 1.3% against exact math, 16-bit about 5e-5. run_batch's gate/up equal run()'s bit for
    // bit with 8-bit activations; with 16-bit ones they differ by float rounding, which flips a few of h's 32767-step
    // roundings, so the two agree to about 1e-5 rather than 1e-7.
    const double exact_max = precise ? 1e-3 : 0.03, same_max = precise ? 2e-5 : 1e-5;

    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0, 1);
    auto random_experts = [&](int * ids, float * w) {
        for (int k = 0; k < 10;) {
            const int e = int(rng() % 512);
            bool dup = false;
            for (int j = 0; j < k; ++j) dup |= ids[j] == e;
            if (!dup) ids[k++] = e;
        }
        float s = 0;
        for (int k = 0; k < 10; ++k) s += (w[k] = 0.2f + float(rng() % 100) / 100.0f);
        for (int k = 0; k < 10; ++k) w[k] /= s;
    };

    int failures = 0;
    std::vector<float> x(3 * 2560), out(3 * 2560), single(2560);
    for (int il : layers) {
        // accuracy
        for (auto & v : x) v = nd(rng);
        int ids[30];
        float w[30];
        random_experts(ids, w);
        ex.run(il, 1, x.data(), ids, w, nullptr, out.data());
        const std::vector<double> ref = reference(m, il, x.data(), ids, w);
        double num = 0, den = 0;
        for (int r = 0; r < 2560; ++r) { num += (out[r] - ref[r]) * (out[r] - ref[r]); den += ref[r] * ref[r]; }
        const double rel = std::sqrt(num / den);
        // batching: tokens 1 and 2 reuse token 0's experts in a different order, with their own x
        for (int t = 1; t < 3; ++t)
            for (int k = 0; k < 10; ++k) { ids[t * 10 + k] = ids[(k + 3 * t) % 10]; w[t * 10 + k] = w[(k + 3 * t) % 10]; }
        ex.run(il, 3, x.data(), ids, w, nullptr, out.data());
        // the experts are summed in a different order, so allow float rounding differences
        bool same = true;
        for (int t = 0; t < 3; ++t) {
            ex.run(il, 1, x.data() + t * 2560, ids + t * 10, w + t * 10, nullptr, single.data());
            double dn = 0, dd = 0;
            for (int r = 0; r < 2560; ++r) {
                const double a = single[r], b = out[t * 2560 + r];
                dn += (a - b) * (a - b);
                dd += a * a;
            }
            same &= std::sqrt(dn / dd) < 1e-5;
        }
        const bool ok = rel < exact_max && same;
        failures += !ok;
        printf("layer %2d %-14s rel. error vs exact %.4f%%, batched matches single: %s  %s\n", il, ex.format_name(il), 100 * rel,
               same ? "yes" : "NO", ok ? "ok" : "FAIL");
    }

    // run_batch against run(). 300 tokens with skewed routing and a random 70% of the pairs on the CPU; one expert sits
    // in every token's slot 0 and stays on the CPU, so it has more pairs than one pass of a gate/up task takes; token 7
    // has no pairs and must come out zero, as must a call with no pair on the CPU. Then 1 and 40 tokens with every pair
    // on the CPU, and the exact reference for two of them.
    for (int il : layers) {
        const int T = 300;
        Routing R = skewed_routing(rng, T);
        const int forced = R.ids[0];
        for (int t = 0; t < T; ++t) {
            int at = -1;
            for (int k = 0; k < 10; ++k)
                if (R.ids[10 * t + k] == forced) at = k;
            if (at > 0) {
                std::swap(R.ids[10 * t], R.ids[10 * t + at]);
                std::swap(R.w[10 * t], R.w[10 * t + at]);
            } else if (at < 0) {
                R.ids[10 * t] = forced;
            }
            for (int k = 0; k < 10; ++k) R.on_cpu[10 * t + k] = k == 0 || rng() % 10 < 7;
        }
        for (int k = 0; k < 10; ++k) R.on_cpu[70 + k] = 0;
        std::vector<float> xb(std::size_t(T) * 2560), ob(xb.size()), os(xb.size());
        for (auto & v : xb) v = nd(rng);
        ex.run_batch(il, T, xb.data(), R.ids.data(), R.w.data(), R.on_cpu.data(), ob.data());
        run_chunked(ex, il, T, xb.data(), R.ids.data(), R.w.data(), R.on_cpu.data(), os.data());
        double worst = 0;
        for (int t = 0; t < T; ++t) worst = std::max(worst, rel_diff(os.data() + std::size_t(t) * 2560, ob.data() + std::size_t(t) * 2560, 2560));
        bool zero = true;
        for (int r = 0; r < 2560; ++r) zero &= ob[7 * 2560 + r] == 0.0f;

        // every pair on the GPU: all zero
        std::vector<std::uint8_t> none(50, 0);
        std::fill(ob.begin(), ob.begin() + 5 * 2560, 1.0f);
        ex.run_batch(il, 5, xb.data(), R.ids.data(), R.w.data(), none.data(), ob.data());
        for (int r = 0; r < 5 * 2560; ++r) zero &= ob[r] == 0.0f;

        // a single token, and 40 tokens, with every pair on the CPU
        ex.run_batch(il, 1, xb.data(), R.ids.data(), R.w.data(), nullptr, ob.data());
        run_chunked(ex, il, 1, xb.data(), R.ids.data(), R.w.data(), nullptr, os.data());
        double worst_all = rel_diff(os.data(), ob.data(), 2560), exact = 0;
        const int T2 = 40;
        ex.run_batch(il, T2, xb.data(), R.ids.data(), R.w.data(), nullptr, ob.data());
        run_chunked(ex, il, T2, xb.data(), R.ids.data(), R.w.data(), nullptr, os.data());
        for (int t = 0; t < T2; ++t) worst_all = std::max(worst_all, rel_diff(os.data() + std::size_t(t) * 2560, ob.data() + std::size_t(t) * 2560, 2560));
        for (int t = 0; t < 2; ++t) {
            const std::vector<double> ref = reference(m, il, xb.data() + std::size_t(t) * 2560, R.ids.data() + 10 * t, R.w.data() + 10 * t);
            double num = 0, den = 0;
            for (int r = 0; r < 2560; ++r) {
                const double d = ob[std::size_t(t) * 2560 + r] - ref[r];
                num += d * d;
                den += ref[r] * ref[r];
            }
            exact = std::max(exact, std::sqrt(num / den));
        }
        const bool ok = worst < same_max && worst_all < same_max && zero && exact < exact_max;
        failures += !ok;
        printf("layer %2d run_batch vs run(): max rel. diff %.1e (300 tokens, part on CPU), %.1e (1 and 40 tokens, all); idle tokens zero: %s; "
               "rel. error vs exact %.4f%%  %s\n",
               il, worst, worst_all, zero ? "yes" : "NO", 100 * exact, ok ? "ok" : "FAIL");
    }

    // speed: one token per layer, then three tokens with no shared experts and with full sharing
    for (int nt : {1, 3}) {
        for (int share : {0, 1}) {
            if (nt == 1 && share) continue;
            double total = 0;
            std::size_t bytes = 0;
            for (int tok = 0; tok < tokens; ++tok) {
                for (auto & v : x) v = nd(rng);
                auto a = clk::now();
                for (int il : layers) {
                    int ids[30];
                    float w[30];
                    for (int t = 0; t < nt; ++t) random_experts(ids + 10 * t, w + 10 * t);
                    if (share)
                        for (int t = 1; t < nt; ++t) std::memcpy(ids + 10 * t, ids, 40);
                    ex.run(il, nt, x.data(), ids, w, nullptr, out.data());
                    if (tok == 0) bytes += ex.expert_bytes(il) * std::size_t(share ? 10 : 10 * nt);
                }
                if (tok >= 3) total += std::chrono::duration<double>(clk::now() - a).count();
            }
            const double per_call = total / (tokens - 3);
            const double scale = 48.0 / double(layers.size());
            printf("%d token(s)%s: %.2f ms for %zu layers, %.1f GB/s; projected 48 layers: %.1f ms -> %.1f tokens/s\n", nt,
                   nt == 1 ? "" : (share ? ", shared experts" : ", distinct experts"), per_call * 1e3, layers.size(),
                   bytes / per_call / 1e9, per_call * scale * 1e3, nt / (per_call * scale));
        }
    }

    if (do_export) failures += test_export(m, ex, o);

    // prefill speed: chunks of each size on every layer, (a) all pairs on the CPU, (b) the cached 27% of experts on the GPU
    if (!batch_sizes.empty()) {
        const int max_t = *std::max_element(batch_sizes.begin(), batch_sizes.end());
        std::vector<Routing> routes;
        for (std::size_t li = 0; li < layers.size(); ++li) routes.push_back(skewed_routing(rng, max_t));
        std::vector<float> xb(std::size_t(max_t) * 2560), ob(xb.size());
        for (auto & v : xb) v = nd(rng);
        constexpr double kMacsPerPair = 3.0 * 2560 * 640;  // gate, up, down
        printf("prefill, %zu layers, projected to 48:\n", layers.size());
        printf("  tokens  pairs     CPU pairs  experts  ms/layer   GB/s  %s TOPS  tokens/s\n", precise ? "int16" : " int8");
        for (int T : batch_sizes) {
            for (int gpu : {0, 1}) {
                double bytes = 0, pairs = 0, experts = 0;
                for (std::size_t li = 0; li < layers.size(); ++li) {
                    std::vector<int> seen(512, 0);
                    for (int i = 0; i < 10 * T; ++i)
                        if (!gpu || routes[li].on_cpu[i]) {
                            pairs += 1;
                            seen[routes[li].ids[i]] = 1;
                        }
                    const int n = std::accumulate(seen.begin(), seen.end(), 0);
                    experts += n;
                    bytes += double(n) * double(ex.expert_bytes(layers[li]));
                }
                const int reps = std::max(3, std::min(20, 8192 / T));
                double total = 0;
                for (int rep = -1; rep < reps; ++rep)
                    for (std::size_t li = 0; li < layers.size(); ++li) {
                        const Routing & R = routes[li];
                        auto a = clk::now();
                        ex.run_batch(layers[li], T, xb.data(), R.ids.data(), R.w.data(), gpu ? R.on_cpu.data() : nullptr, ob.data());
                        if (rep >= 0) total += std::chrono::duration<double>(clk::now() - a).count();
                    }
                const double nl = double(layers.size()), per_layer = total / (reps * nl);
                printf("  %6d  %-9s %5.1f%%  %7.0f  %8.2f  %5.1f  %9.2f  %8.0f\n", T, gpu ? "GPU cache" : "all CPU", 100 * pairs / (10.0 * T * nl),
                       experts / nl, per_layer * 1e3, bytes / nl / per_layer / 1e9, 2 * kMacsPerPair * pairs / nl / per_layer / 1e12,
                       T / (48 * per_layer));
            }
        }
    }
    return failures;
}

// The MTP head's layer (Q8_0 gate/up and down, loaded with add_layer): one token against the double-precision reference,
// 1..4 tokens sharing experts against single-token calls, and the speed of one token's 10 experts.
static int test_mtp(const GgufModel & m, const std::string & mtp_path, const Options & o) {
    GgufModel g(std::vector<std::string>{mtp_path});
    const int il = int(m.get_int("qwen4exp.block_count"));  // the MTP block follows the main model's layers
    CpuExpertsConfig cfg;
    cfg.threads = o.threads;
    cfg.pin_threads = o.pin;
    cfg.precise_activations = false;  // the MTP layer uses 16-bit activations whatever this says
    cfg.layers = {0};
    CpuExperts ex(m, cfg);
    ex.add_layer(g, il);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0, 1);
    int failures = 0;
    constexpr int T = 4;
    std::vector<float> x(std::size_t(T) * 2560), w(T * 10), out(std::size_t(T) * 2560), one(2560);
    std::vector<std::int32_t> ids(T * 10);
    for (auto & v : x) v = nd(rng);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < 10; ++k) {
            int e = 0;
            bool dup = true;
            while (dup) {  // tokens 1.. share their first 3 experts with token 0
                e = (t > 0 && k < 3) ? ids[k] : int(rng() % 512);
                dup = false;
                for (int j = 0; j < k; ++j) dup |= ids[t * 10 + j] == e;
            }
            ids[t * 10 + k] = e;
            w[t * 10 + k] = 0.05f + float(rng() % 100) / 1000.0f;
        }
    double worst_ref = 0, worst_batch = 0;
    ex.run(il, T, x.data(), ids.data(), w.data(), nullptr, out.data());
    for (int t = 0; t < T; ++t) {
        const std::vector<double> ref = reference(g, il, x.data() + std::size_t(t) * 2560, ids.data() + 10 * t, w.data() + 10 * t);
        std::vector<float> rf(ref.begin(), ref.end());
        worst_ref = std::max(worst_ref, rel_diff(rf.data(), out.data() + std::size_t(t) * 2560, 2560));
        ex.run(il, 1, x.data() + std::size_t(t) * 2560, ids.data() + 10 * t, w.data() + 10 * t, nullptr, one.data());
        worst_batch = std::max(worst_batch, rel_diff(one.data(), out.data() + std::size_t(t) * 2560, 2560));
    }
    const bool ok = worst_ref < 2e-4 && worst_batch < 1e-6;
    failures += !ok;
    const int reps = 50;
    const auto t0 = clk::now();
    for (int r = 0; r < reps; ++r) ex.run(il, 1, x.data(), ids.data() + 10 * (r % T), w.data(), nullptr, one.data());
    const double us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / reps;
    printf("MTP layer %d (%s, %.2f MB per expert): rel. error vs exact %.2e, 4 tokens vs 1 at a time %.2e; one token's 10 experts %.0f us "
           "(%.1f GB/s)  %s\n",
           il, ex.format_name(il), ex.expert_bytes(il) / 1e6, worst_ref, worst_batch, us, 10.0 * ex.expert_bytes(il) / (us * 1e3), ok ? "ok" : "FAIL");
    return failures;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> shards;
    Options o;
    std::string mode = "both", mtp;
    bool do_export = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--layers" || a == "--batch") {
            std::vector<int> & list = a == "--layers" ? o.layers : o.batch_sizes;
            list.clear();
            std::stringstream ss(argv[++i]);
            for (std::string v; std::getline(ss, v, ',');) list.push_back(std::stoi(v));
        } else if (a == "--threads") o.threads = std::stoi(argv[++i]);
        else if (a == "--tokens") o.tokens = std::stoi(argv[++i]);
        else if (a == "--no-pin") o.pin = false;
        else if (a == "--mode") mode = argv[++i];
        else if (a == "--no-export") do_export = false;
        else if (a == "--mtp") mtp = argv[++i];
        else shards.push_back(a);
    }
    GgufModel m(shards);
    int failures = 0;
    if (!mtp.empty()) {
        failures += test_mtp(m, mtp, o);
        printf(failures ? "FAILED\n" : "ALL OK\n");
        return failures ? 1 : 0;
    }
    if (mode != "precise") failures += test_mode(m, o, false, do_export);
    if (mode != "int8") failures += test_mode(m, o, true, do_export && mode == "precise");
    printf(failures ? "FAILED\n" : "ALL OK\n");
    return failures ? 1 : 0;
}
