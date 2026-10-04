// Tests and benchmarks the GPU GEMV kernels on every dense matrix of the real model.
//   accuracy: sampled rows of every matrix, 1 to 4 tokens, against a double-precision reference built
//             from the exact dequantizers; token t of a T-token call must equal the 1-token call bit for bit
//   speed:    one decode step's worth of dense matrix-vector products (every matrix once, in model
//             order, so the working set is far larger than L2), for 1 and 4 tokens
// Usage: test_gpu_gemv <shards...> [--filter substring] [--reps 10]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "flashnext/cuda/gemv.h"
#include "flashnext/quants.h"

using namespace ninfer::flashnext;
namespace fc = ninfer::flashnext::cuda;

static bool is_gemv_matrix(const GgufTensor & t) {
    if (t.shape.size() < 2 || t.shape[1] < 2) return false;
    for (std::size_t i = 2; i < t.shape.size(); ++i)
        if (t.shape[i] != 1) return false;
    if (t.name == "token_embd.weight" || t.name.find("conv1d") != std::string::npos) return false;
    return t.type == GgufType::Q8_0 || t.type == GgufType::F32 || t.type == GgufType::BF16 || t.type == GgufType::Q6_K;
}

static std::string kind_of(const std::string & name) {
    std::string k = name;
    if (k.rfind("blk.", 0) == 0) k = k.substr(k.find('.', 4) + 1);
    return k;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> shards;
    std::string filter;
    int reps = 10;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--filter") filter = argv[++i];
        else if (a == "--reps") reps = std::stoi(argv[++i]);
        else shards.push_back(a);
    }
    GgufModel m(shards);

    // model order: blk.0 .. blk.47, then the output projections
    std::vector<const GgufTensor *> mats;
    for (const auto & [name, t] : m.tensors())
        if (is_gemv_matrix(t) && (filter.empty() || name.find(filter) != std::string::npos)) mats.push_back(&t);
    auto layer_of = [](const std::string & n) { return n.rfind("blk.", 0) == 0 ? std::stoi(n.substr(4)) : 1000; };
    std::stable_sort(mats.begin(), mats.end(), [&](auto * a, auto * b) { return layer_of(a->name) < layer_of(b->name); });

    std::vector<fc::DeviceWeight> dev;
    std::size_t bytes = 0;
    int max_k = 0, max_n = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (const GgufTensor * t : mats) {
        dev.push_back(fc::upload_gemv_weight(*t));
        bytes += t->bytes;
        max_k = std::max(max_k, dev.back().view.k);
        max_n = std::max(max_n, dev.back().view.n);
    }
    printf("uploaded %zu matrices, %.2f GiB, in %.1f s\n", mats.size(), bytes / 1073741824.0,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> hx(4 * std::size_t(max_k));
    for (auto & v : hx) v = nd(rng);
    fc::DeviceBuffer dx(hx.size() * 4), dy(4 * std::size_t(max_n) * 4), dy1(std::size_t(max_n) * 4);
    fc::check(cudaMemcpy(dx.get(), hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x");

    // accuracy
    int failures = 0;
    std::map<std::string, double> worst;  // per kind
    std::vector<float> y(4 * std::size_t(max_n)), y1(max_n), row(max_k);
    for (std::size_t i = 0; i < mats.size(); ++i) {
        const GgufTensor & t = *mats[i];
        const fc::GpuWeight & w = dev[i].view;
        const std::size_t rb = row_bytes(t.type, w.k);
        // x for token tok is hx[tok * K ...]: the device copy uses the same [T][K] layout with stride K
        std::vector<float> xk(4 * std::size_t(w.k));
        for (int tok = 0; tok < 4; ++tok) std::memcpy(xk.data() + tok * w.k, hx.data() + tok * std::size_t(max_k), 4 * w.k);
        fc::check(cudaMemcpy(dx.get(), xk.data(), xk.size() * 4, cudaMemcpyHostToDevice), "x");
        std::vector<int> rows;
        const int samples = std::min(w.n, 64);
        for (int s = 0; s < samples; ++s) rows.push_back(int((std::int64_t(s) * (w.n - 1)) / std::max(1, samples - 1)));
        std::vector<double> ref(4 * rows.size());
        for (std::size_t s = 0; s < rows.size(); ++s) {
            dequantize_row(t.type, t.data + std::size_t(rows[s]) * rb, row.data(), w.k);
            for (int tok = 0; tok < 4; ++tok) {
                double acc = 0;
                for (int j = 0; j < w.k; ++j) acc += double(row[j]) * xk[std::size_t(tok) * w.k + j];
                ref[tok * rows.size() + s] = acc;
            }
        }
        fc::gemv(w, dx.as<float>(), dy1.as<float>(), 1, nullptr);
        fc::check(cudaMemcpy(y1.data(), dy1.get(), std::size_t(w.n) * 4, cudaMemcpyDeviceToHost), "y1");
        double rel = 0;
        bool consistent = true;
        for (int nt = 1; nt <= 4; ++nt) {
            fc::gemv(w, dx.as<float>(), dy.as<float>(), nt, nullptr);
            fc::check(cudaMemcpy(y.data(), dy.get(), std::size_t(nt) * w.n * 4, cudaMemcpyDeviceToHost), "y");
            consistent &= std::memcmp(y.data(), y1.data(), std::size_t(w.n) * 4) == 0;
            for (int tok = 0; tok < nt; ++tok) {
                double num = 0, den = 0;
                for (std::size_t s = 0; s < rows.size(); ++s) {
                    const double r = ref[tok * rows.size() + s], g = y[std::size_t(tok) * w.n + rows[s]];
                    num += (g - r) * (g - r);
                    den += r * r;
                }
                rel = std::max(rel, std::sqrt(num / std::max(den, 1e-300)));
            }
        }
        const bool ok = rel < 1e-5 && consistent;
        if (!ok) {
            ++failures;
            printf("FAIL %-36s %-5s K=%-6d N=%-7d rel %.2e, multi-token %s\n", t.name.c_str(), type_name(t.type), w.k, w.n, rel,
                   consistent ? "consistent" : "DIFFERS");
        }
        double & wv = worst[kind_of(t.name) + " " + type_name(t.type) + " " + std::to_string(w.k) + "->" + std::to_string(w.n)];
        wv = std::max(wv, rel);
    }
    printf("\naccuracy (worst relative error over sampled rows, 1..4 tokens):\n");
    for (const auto & [k, v] : worst) printf("  %-52s %.2e\n", k.c_str(), v);

    // speed: one decode step's dense products in model order
    cudaStream_t s;
    fc::check(cudaStreamCreate(&s), "stream");
    std::vector<cudaEvent_t> ev(mats.size() + 1);
    for (auto & e : ev) fc::check(cudaEventCreate(&e), "event");
    for (int nt : {1, 4}) {
        std::map<std::string, std::pair<double, std::size_t>> per_kind;  // ms, bytes
        double total = 0;
        for (int r = 0; r < reps + 2; ++r) {
            for (std::size_t i = 0; i < mats.size(); ++i) {
                cudaEventRecord(ev[i], s);
                fc::gemv(dev[i].view, dx.as<float>(), dy.as<float>(), nt, s);
            }
            cudaEventRecord(ev[mats.size()], s);
            fc::check(cudaEventSynchronize(ev[mats.size()]), "sync");
            if (r < 2) continue;  // warm-up
            float ms = 0;
            cudaEventElapsedTime(&ms, ev[0], ev[mats.size()]);
            total += ms;
            for (std::size_t i = 0; i < mats.size(); ++i) {
                cudaEventElapsedTime(&ms, ev[i], ev[i + 1]);
                auto & pk = per_kind[kind_of(mats[i]->name)];
                pk.first += ms;
                pk.second += mats[i]->bytes;
            }
        }
        total /= reps;
        printf("\n%d token(s): %.3f ms per decode step for %.2f GiB, %.0f GB/s\n", nt, total, bytes / 1073741824.0, bytes / (total * 1e6));
        for (const auto & [k, v] : per_kind)
            printf("  %-34s %7.3f ms %6.0f GB/s\n", k.c_str(), v.first / reps, (v.second / reps) / (v.first / reps * 1e6));

        // the same step as one CUDA graph: what remains is kernel time, not launch overhead
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        fc::check(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal), "capture");
        for (std::size_t i = 0; i < mats.size(); ++i) fc::gemv(dev[i].view, dx.as<float>(), dy.as<float>(), nt, s);
        fc::check(cudaStreamEndCapture(s, &graph), "capture");
        fc::check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
        for (int r = 0; r < 2; ++r) cudaGraphLaunch(exec, s);
        cudaEventRecord(ev[0], s);
        for (int r = 0; r < reps; ++r) cudaGraphLaunch(exec, s);
        cudaEventRecord(ev[1], s);
        fc::check(cudaEventSynchronize(ev[1]), "sync");
        float ms = 0;
        cudaEventElapsedTime(&ms, ev[0], ev[1]);
        printf("  as one CUDA graph: %.3f ms per decode step, %.0f GB/s\n", ms / reps, bytes / (ms / reps * 1e6));
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
    }
    printf(failures ? "\nFAILED (%d)\n" : "\nALL OK\n", failures);
    return failures ? 1 : 0;
}
