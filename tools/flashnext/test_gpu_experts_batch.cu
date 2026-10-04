// Tests and benchmarks the batched (prompt processing) GPU expert kernels on real experts.
//   accuracy: every token's output against a double-precision reference built from the exact
//             dequantizers, for 1, 7 and 256 tokens with about 30% of the pairs left to the CPU (they
//             must add nothing; one token has no cached pair and must come out exactly 0), in every
//             BatchMath; a second run and a CUDA graph replay must match the first bit for bit, and so
//             must a replay after the routing changed in device memory and a direct run on that routing
//   speed:    140 cached experts of a layer, routing skewed so that about 70% of the pairs are cached
//             and a few slots get many more tokens than others, for 256, 1024 and 4096 tokens
// Usage: test_gpu_experts_batch <shards...> [--layers 0,2,4|none] [--bench 0,2,4|none] [--sizes 256,1024,4096]
//        [--modes fp32,fp16x2,fp16] [--reps 10]
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "flashnext/cuda/device.h"
#include "flashnext/cuda/experts.h"
#include "flashnext/cuda/experts_batch.h"
#include "flashnext/cuda/ops.h"
#include "flashnext/quants.h"

using namespace ninfer::flashnext;
namespace fc = ninfer::flashnext::cuda;

namespace {

constexpr int kE = 2560, kF = 640, kK = 10;

std::vector<int> parse_list(const char * s) {
    std::vector<int> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');)
        if (t != "none") v.push_back(std::stoi(t));
    return v;
}

struct Routing {
    std::vector<int> ids;             // [n][10] expert ids
    std::vector<std::int32_t> slots;  // [n][10] pool slot or -1
    std::vector<float> w;             // [n][10] router weights, summing to 1 per token
};

// Each pick is a cached expert with probability `hit` (Zipf-like over the cached ones, so a few slots
// get many tokens), otherwise a uniformly drawn uncached one; the 10 picks of a token are distinct.
// Token `none` (if >= 0) gets no cached expert at all.
Routing make_routing(int n, const std::vector<int> & cached, double hit, int none, std::mt19937 & rng) {
    std::vector<int> slot_of(fc::kExperts, -1), uncached;
    for (std::size_t i = 0; i < cached.size(); ++i) slot_of[std::size_t(cached[i])] = int(i);
    for (int e = 0; e < fc::kExperts; ++e)
        if (slot_of[std::size_t(e)] < 0) uncached.push_back(e);
    std::vector<double> zw(cached.size());
    for (std::size_t r = 0; r < zw.size(); ++r) zw[r] = 1.0 / (double(r) + 8.0);
    std::discrete_distribution<int> zipf(zw.begin(), zw.end());
    std::uniform_real_distribution<double> u(0, 1);
    Routing R;
    R.ids.resize(std::size_t(n) * kK);
    R.slots.resize(R.ids.size());
    R.w.resize(R.ids.size());
    for (int t = 0; t < n; ++t) {
        std::set<int> used;
        float sum = 0;
        for (int k = 0; k < kK; ++k) {
            int e;
            do e = (t != none && u(rng) < hit) ? cached[std::size_t(zipf(rng))] : uncached[rng() % uncached.size()];
            while (used.count(e));
            used.insert(e);
            const std::size_t p = std::size_t(t) * kK + std::size_t(k);
            R.ids[p] = e;
            R.slots[p] = slot_of[std::size_t(e)];
            R.w[p] = 0.2f + float(u(rng));
            sum += R.w[p];
        }
        for (int k = 0; k < kK; ++k) R.w[std::size_t(t) * kK + std::size_t(k)] /= sum;
    }
    return R;
}

// FFN inputs: normal values, a per-token gain over two decades, and on every 16th token one channel
// 30 times larger (models have such outliers; they test the fp16 range handling).
std::vector<float> make_x(int n, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0, 1);
    std::uniform_real_distribution<float> ug(-1, 1);
    std::vector<float> x(std::size_t(n) * kE);
    for (int t = 0; t < n; ++t) {
        const float gain = std::pow(10.0f, ug(rng));
        for (int i = 0; i < kE; ++i) x[std::size_t(t) * kE + std::size_t(i)] = gain * nd(rng);
        if (t % 16 == 5) x[std::size_t(t) * kE + std::size_t(rng() % kE)] *= 30.0f;
    }
    return x;
}

struct Pool {
    fc::ExpertLayout lay;
    fc::DeviceBuffer buf;
};

Pool make_pool(const GgufTensor & tg, const GgufTensor & tu, const GgufTensor & td, const std::vector<int> & experts) {
    Pool P;
    P.lay = fc::expert_layout(tg.type, td.type);
    std::vector<std::uint8_t> host(experts.size() * P.lay.slot_bytes);
    std::vector<std::thread> th;
    const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    for (unsigned i = 0; i < nt; ++i)
        th.emplace_back([&, i] {
            for (std::size_t j = i; j < experts.size(); j += nt) fc::pack_expert(P.lay, tg, tu, td, experts[j], host.data() + j * P.lay.slot_bytes);
        });
    for (auto & t : th) t.join();
    P.buf = fc::DeviceBuffer(host.size());
    fc::check(cudaMemcpy(P.buf.get(), host.data(), host.size(), cudaMemcpyHostToDevice), "pool");
    return P;
}

// Reference out[t] = sum over cached pairs of w * down(silu(gate x) * up x), all in double from the
// exactly dequantized weights. One worker per expert at a time.
std::vector<double> reference(const GgufTensor & tg, const GgufTensor & tu, const GgufTensor & td, const Routing & R, const std::vector<float> & x,
                              int n) {
    std::map<int, std::vector<int>> by_e;
    for (int p = 0; p < n * kK; ++p)
        if (R.slots[std::size_t(p)] >= 0) by_e[R.ids[std::size_t(p)]].push_back(p);
    std::vector<std::pair<int, std::vector<int>>> jobs(by_e.begin(), by_e.end());
    std::vector<double> ypair(std::size_t(n) * kK * kE, 0.0);
    std::atomic<std::size_t> next{0};
    const std::size_t gr = row_bytes(tg.type, kE), dr = row_bytes(td.type, kF);
    auto work = [&] {
        std::vector<float> G(std::size_t(kF) * kE), U(G.size()), D(G.size());
        std::vector<double> h(kF);
        for (std::size_t j; (j = next++) < jobs.size();) {
            const int e = jobs[j].first;
            for (int r = 0; r < kF; ++r) {
                dequantize_row(tg.type, tg.data + (std::size_t(e) * kF + std::size_t(r)) * gr, G.data() + std::size_t(r) * kE, kE);
                dequantize_row(tu.type, tu.data + (std::size_t(e) * kF + std::size_t(r)) * gr, U.data() + std::size_t(r) * kE, kE);
            }
            for (int r = 0; r < kE; ++r)
                dequantize_row(td.type, td.data + (std::size_t(e) * kE + std::size_t(r)) * dr, D.data() + std::size_t(r) * kF, kF);
            for (const int p : jobs[j].second) {
                const float * xt = x.data() + std::size_t(p / kK) * kE;
                for (int r = 0; r < kF; ++r) {
                    double g = 0, u = 0;
                    const float * gw = G.data() + std::size_t(r) * kE;
                    const float * uw = U.data() + std::size_t(r) * kE;
                    for (int i = 0; i < kE; ++i) {
                        g += double(gw[i]) * xt[i];
                        u += double(uw[i]) * xt[i];
                    }
                    h[std::size_t(r)] = g / (1.0 + std::exp(-g)) * u;
                }
                double * y = ypair.data() + std::size_t(p) * kE;
                for (int r = 0; r < kE; ++r) {
                    const float * dw = D.data() + std::size_t(r) * kF;
                    double s = 0;
                    for (int i = 0; i < kF; ++i) s += double(dw[i]) * h[std::size_t(i)];
                    y[r] = s;
                }
            }
        }
    };
    std::vector<std::thread> th;
    for (unsigned i = 0; i < std::max(1u, std::thread::hardware_concurrency()); ++i) th.emplace_back(work);
    for (auto & t : th) t.join();
    std::vector<double> out(std::size_t(n) * kE, 0.0);
    for (int t = 0; t < n; ++t)
        for (int k = 0; k < kK; ++k) {
            const std::size_t p = std::size_t(t) * kK + std::size_t(k);
            if (R.slots[p] < 0) continue;
            for (int i = 0; i < kE; ++i) out[std::size_t(t) * kE + std::size_t(i)] += double(R.w[p]) * ypair[p * kE + std::size_t(i)];
        }
    return out;
}

const char * math_name(fc::BatchMath m) {
    switch (m) {
    case fc::BatchMath::Fp32: return "fp32  ";
    case fc::BatchMath::Fp16x2: return "fp16x2";
    case fc::BatchMath::Fp16: return "fp16  ";
    }
    return "?";
}

std::vector<fc::BatchMath> parse_modes(const char * s) {
    std::vector<fc::BatchMath> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) {
        if (t == "fp32") v.push_back(fc::BatchMath::Fp32);
        else if (t == "fp16x2") v.push_back(fc::BatchMath::Fp16x2);
        else if (t == "fp16") v.push_back(fc::BatchMath::Fp16);
        else throw std::runtime_error("unknown math " + t);
    }
    return v;
}

struct DeviceRouting {
    fc::DeviceBuffer x, slots, w, out;
    DeviceRouting(const Routing & R, const std::vector<float> & x_host, int n)
        : x(x_host.size() * 4), slots(R.slots.size() * 4), w(R.w.size() * 4), out(std::size_t(n) * kE * 4) {
        fc::check(cudaMemcpy(x.get(), x_host.data(), x_host.size() * 4, cudaMemcpyHostToDevice), "x");
        fc::check(cudaMemcpy(slots.get(), R.slots.data(), R.slots.size() * 4, cudaMemcpyHostToDevice), "slots");
        fc::check(cudaMemcpy(w.get(), R.w.data(), R.w.size() * 4, cudaMemcpyHostToDevice), "w");
    }
};

int run(int argc, char ** argv) {
    std::vector<std::string> shards;
    std::vector<int> layers = {0, 2, 4}, bench = {0, 2, 4}, sizes = {256, 1024, 4096};
    std::vector<fc::BatchMath> modes = {fc::BatchMath::Fp32, fc::BatchMath::Fp16x2, fc::BatchMath::Fp16};
    int reps = 10;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--layers" && i + 1 < argc) layers = parse_list(argv[++i]);
        else if (a == "--bench" && i + 1 < argc) bench = parse_list(argv[++i]);
        else if (a == "--sizes" && i + 1 < argc) sizes = parse_list(argv[++i]);
        else if (a == "--modes" && i + 1 < argc) modes = parse_modes(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::stoi(argv[++i]);
        else shards.push_back(a);
    }
    if (shards.empty()) {
        fprintf(stderr, "usage: test_gpu_experts_batch <shards...> [--layers 0,2,4|none] [--bench 0,2,4|none] [--sizes 256,1024,4096] "
                        "[--modes fp32,fp16x2,fp16] [--reps 10]\n");
        return 2;
    }
    GgufModel m(shards);
    fc::experts_batch_init();
    cudaStream_t s;
    fc::check(cudaStreamCreate(&s), "stream");
    cudaEvent_t e0, e1;
    fc::check(cudaEventCreate(&e0), "event");
    fc::check(cudaEventCreate(&e1), "event");
    int failures = 0;

    // ---- accuracy ----
    constexpr int kTestTokens = 256, kTestCached = 40;
    fc::DeviceBuffer ws(fc::experts_batch_workspace_bytes(kTestTokens));
    for (const int il : layers) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const GgufTensor & tg = m.tensor(p + "ffn_gate_exps.weight");
        const GgufTensor & tu = m.tensor(p + "ffn_up_exps.weight");
        const GgufTensor & td = m.tensor(p + "ffn_down_exps.weight");
        std::mt19937 rng(1000 + il);
        std::vector<int> all(fc::kExperts);
        for (int e = 0; e < fc::kExperts; ++e) all[std::size_t(e)] = e;
        std::shuffle(all.begin(), all.end(), rng);
        const std::vector<int> cached(all.begin(), all.begin() + kTestCached);
        const Pool pool = make_pool(tg, tu, td, cached);
        const Routing R = make_routing(kTestTokens, cached, 0.7, 3, rng);
        const std::vector<float> x = make_x(kTestTokens, rng);
        const std::vector<double> ref = reference(tg, tu, td, R, x, kTestTokens);
        DeviceRouting d(R, x, kTestTokens);
        int npairs = 0;
        for (const auto v : R.slots) npairs += v >= 0;

        for (const fc::BatchMath math : modes) {
            double worst[3] = {0, 0, 0};
            bool zero_ok = true, repeat_ok = true, graph_ok = true;
            const int ns[3] = {1, 7, kTestTokens};
            std::vector<float> out(std::size_t(kTestTokens) * kE), again(out.size());
            for (int ni = 0; ni < 3; ++ni) {
                const int n = ns[ni];
                // stale workspace and output must not leak into the result: fill both with NaN
                fc::check(cudaMemset(ws.get(), 0xFF, ws.bytes()), "ws");
                fc::check(cudaMemset(d.out.get(), 0xFF, d.out.bytes()), "out");
                fc::experts_gpu_batch(pool.lay, pool.buf.as<std::uint8_t>(), n, d.x.as<float>(), d.slots.as<std::int32_t>(), d.w.as<float>(),
                                      d.out.as<float>(), ws.get(), s, math);
                fc::check(cudaStreamSynchronize(s), "run");
                fc::check(cudaMemcpy(out.data(), d.out.get(), std::size_t(n) * kE * 4, cudaMemcpyDeviceToHost), "out");
                for (int t = 0; t < n; ++t) {
                    double num = 0, den = 0;
                    for (int i = 0; i < kE; ++i) {
                        const double r = ref[std::size_t(t) * kE + std::size_t(i)], v = out[std::size_t(t) * kE + std::size_t(i)];
                        num += (v - r) * (v - r);
                        den += r * r;
                    }
                    if (den == 0) {
                        for (int i = 0; i < kE; ++i) zero_ok &= out[std::size_t(t) * kE + std::size_t(i)] == 0.0f;
                        continue;
                    }
                    const double rel = std::sqrt(num / den);
                    worst[ni] = std::isfinite(rel) ? std::max(worst[ni], rel) : 1e30;
                }
            }
            // bitwise repeatability, directly and through a CUDA graph (n = 256)
            fc::experts_gpu_batch(pool.lay, pool.buf.as<std::uint8_t>(), kTestTokens, d.x.as<float>(), d.slots.as<std::int32_t>(), d.w.as<float>(),
                                  d.out.as<float>(), ws.get(), s, math);
            fc::check(cudaStreamSynchronize(s), "run");
            fc::check(cudaMemcpy(again.data(), d.out.get(), again.size() * 4, cudaMemcpyDeviceToHost), "out");
            repeat_ok = std::memcmp(out.data(), again.data(), out.size() * 4) == 0;
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            fc::check(cudaMemset(d.out.get(), 0xFF, d.out.bytes()), "out");
            fc::check(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal), "capture");
            fc::experts_gpu_batch(pool.lay, pool.buf.as<std::uint8_t>(), kTestTokens, d.x.as<float>(), d.slots.as<std::int32_t>(), d.w.as<float>(),
                                  d.out.as<float>(), ws.get(), s, math);
            fc::check(cudaStreamEndCapture(s, &graph), "end capture");
            fc::check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
            fc::check(cudaGraphLaunch(exec, s), "graph launch");
            fc::check(cudaStreamSynchronize(s), "graph run");
            fc::check(cudaMemcpy(again.data(), d.out.get(), again.size() * 4, cudaMemcpyDeviceToHost), "out");
            graph_ok = std::memcmp(out.data(), again.data(), out.size() * 4) == 0;
            // the graph must read the routing when it runs: drop every third pair, replay, compare with a direct run
            std::vector<std::int32_t> slots2 = R.slots;
            for (std::size_t i = 0; i < slots2.size(); i += 3) slots2[i] = -1;
            fc::check(cudaMemcpy(d.slots.get(), slots2.data(), slots2.size() * 4, cudaMemcpyHostToDevice), "slots");
            fc::check(cudaGraphLaunch(exec, s), "graph launch");
            fc::check(cudaStreamSynchronize(s), "graph run");
            std::vector<float> direct(out.size());
            fc::check(cudaMemcpy(again.data(), d.out.get(), again.size() * 4, cudaMemcpyDeviceToHost), "out");
            fc::experts_gpu_batch(pool.lay, pool.buf.as<std::uint8_t>(), kTestTokens, d.x.as<float>(), d.slots.as<std::int32_t>(), d.w.as<float>(),
                                  d.out.as<float>(), ws.get(), s, math);
            fc::check(cudaStreamSynchronize(s), "run");
            fc::check(cudaMemcpy(direct.data(), d.out.get(), direct.size() * 4, cudaMemcpyDeviceToHost), "out");
            graph_ok = graph_ok && std::memcmp(direct.data(), again.data(), out.size() * 4) == 0 &&
                       std::memcmp(direct.data(), out.data(), out.size() * 4) != 0;
            fc::check(cudaMemcpy(d.slots.get(), R.slots.data(), R.slots.size() * 4, cudaMemcpyHostToDevice), "slots");
            cudaGraphExecDestroy(exec);
            cudaGraphDestroy(graph);

            const double limit = math == fc::BatchMath::Fp32 ? 2e-5 : math == fc::BatchMath::Fp16x2 ? 2e-5 : 1e-3;
            const bool ok = std::max({worst[0], worst[1], worst[2]}) < limit && zero_ok && repeat_ok && graph_ok;
            failures += !ok;
            printf("layer %2d %s/%s %s: %d of %d pairs cached; worst rel. error n=1 %.2e, n=7 %.2e, n=256 %.2e (limit %.0e); "
                   "uncached-only token zero: %s; repeat: %s; graph: %s  %s\n",
                   il, type_name(tg.type), type_name(td.type), math_name(math), npairs, kTestTokens * kK, worst[0], worst[1], worst[2], limit,
                   zero_ok ? "yes" : "NO", repeat_ok ? "bitwise" : "DIFFERS", graph_ok ? "bitwise" : "DIFFERS", ok ? "ok" : "FAIL");
        }
    }

    // ---- speed ----
    constexpr int kBenchCached = 140;
    const int max_n = sizes.empty() ? 1 : *std::max_element(sizes.begin(), sizes.end());
    fc::DeviceBuffer bws(fc::experts_batch_workspace_bytes(max_n));
    if (!bench.empty()) printf("workspace for %d tokens: %.0f MiB\n", max_n, double(bws.bytes()) / 1048576.0);
    for (const int il : bench) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const GgufTensor & tg = m.tensor(p + "ffn_gate_exps.weight");
        const GgufTensor & tu = m.tensor(p + "ffn_up_exps.weight");
        const GgufTensor & td = m.tensor(p + "ffn_down_exps.weight");
        std::mt19937 rng(77 + il);
        std::vector<int> all(fc::kExperts);
        for (int e = 0; e < fc::kExperts; ++e) all[std::size_t(e)] = e;
        std::shuffle(all.begin(), all.end(), rng);
        const std::vector<int> cached(all.begin(), all.begin() + kBenchCached);
        const Pool pool = make_pool(tg, tu, td, cached);
        printf("bench layer %d %s/%s: %d slots, %.0f MiB\n", il, type_name(tg.type), type_name(td.type), kBenchCached,
               double(pool.buf.bytes()) / 1048576.0);
        for (const int n : sizes) {
            const Routing R = make_routing(n, cached, 0.7, -1, rng);
            const std::vector<float> x = make_x(n, rng);
            DeviceRouting d(R, x, n);
            std::vector<int> per(kBenchCached, 0);
            int npairs = 0;
            for (const auto v : R.slots)
                if (v >= 0) ++npairs, ++per[std::size_t(v)];
            const int most = *std::max_element(per.begin(), per.end());
            const double flop = double(npairs) * 2.0 * (double(kE) * 2 * kF + double(kF) * kE);
            for (const fc::BatchMath math : modes) {
                auto run = [&] {
                    fc::experts_gpu_batch(pool.lay, pool.buf.as<std::uint8_t>(), n, d.x.as<float>(), d.slots.as<std::int32_t>(), d.w.as<float>(),
                                          d.out.as<float>(), bws.get(), s, math);
                };
                for (int r = 0; r < 2; ++r) run();
                fc::check(cudaEventRecord(e0, s), "event");
                for (int r = 0; r < reps; ++r) run();
                fc::check(cudaEventRecord(e1, s), "event");
                fc::check(cudaEventSynchronize(e1), "sync");
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                ms /= float(reps);
                printf("  n=%4d %s: %5d cached pairs (%.0f%%, busiest slot %4d tokens): %7.3f ms, %6.1f TFLOPS\n", n, math_name(math), npairs,
                       100.0 * npairs / (n * kK), most, ms, flop / (ms * 1e-3) / 1e12);
            }
        }
    }
    printf(failures ? "FAILED\n" : "ALL OK\n");
    return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        printf("error: %s\n", e.what());
        return 1;
    }
}
