// Tests and benchmarks the CPU expert engine on the real model.
//   accuracy: one token, 10 random experts, against a double-precision reference built from the exact
//             dequantizers (the difference is the engine's 8-bit activations)
//   batching: 3 tokens sharing experts give bit-identical results to 3 single-token calls
//   speed:    decode-shaped calls over several layers, projected to 48 layers
// Usage: test_cpu_experts <shards...> [--layers 0,2,4,...] [--threads 16] [--tokens 40]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
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

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> shards;
    std::vector<int> layers = {0, 2, 4};
    int threads = 16, tokens = 40;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--layers") {
            layers.clear();
            std::stringstream ss(argv[++i]);
            for (std::string v; std::getline(ss, v, ',');) layers.push_back(std::stoi(v));
        } else if (a == "--threads") threads = std::stoi(argv[++i]);
        else if (a == "--tokens") tokens = std::stoi(argv[++i]);
        else shards.push_back(a);
    }
    GgufModel m(shards);
    auto t0 = clk::now();
    CpuExperts ex(m, {threads, layers});
    printf("loaded %zu layers in %.1f s, %.2f GiB resident\n", layers.size(),
           std::chrono::duration<double>(clk::now() - t0).count(), ex.resident_bytes() / 1073741824.0);

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
        const bool ok = rel < 0.03 && same;
        failures += !ok;
        printf("layer %2d %-14s rel. error vs exact %.3f%%, batched matches single: %s  %s\n", il, ex.format_name(il), 100 * rel,
               same ? "yes" : "NO", ok ? "ok" : "FAIL");
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
    printf(failures ? "FAILED\n" : "ALL OK\n");
    return failures ? 1 : 0;
}
