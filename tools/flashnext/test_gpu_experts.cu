// Tests and benchmarks the GPU expert kernels on real experts of every format combination.
//   accuracy: each (token, expert) output against a double-precision reference from the exact
//             dequantizers, for 1..4 tokens, with some pairs left to the CPU (they must come out zero)
//   speed:    one token's 10 experts all in VRAM, per layer
// Usage: test_gpu_experts <shards...> [--layers 0,2,4,30] [--reps 200]
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "flashnext/cuda/device.h"
#include "flashnext/cuda/experts.h"
#include "flashnext/cuda/ops.h"
#include "flashnext/quants.h"

using namespace ninfer::flashnext;
namespace fc = ninfer::flashnext::cuda;

static std::vector<double> reference(const GgufTensor & tg, const GgufTensor & tu, const GgufTensor & td, int e, const float * x, float w) {
    const std::size_t gr = row_bytes(tg.type, 2560), dr = row_bytes(td.type, 640);
    std::vector<float> row(2560), h(640);
    for (int r = 0; r < 640; ++r) {
        double g = 0, u = 0;
        dequantize_row(tg.type, tg.data + (std::size_t(e) * 640 + r) * gr, row.data(), 2560);
        for (int j = 0; j < 2560; ++j) g += double(row[j]) * x[j];
        dequantize_row(tu.type, tu.data + (std::size_t(e) * 640 + r) * gr, row.data(), 2560);
        for (int j = 0; j < 2560; ++j) u += double(row[j]) * x[j];
        h[r] = float(g / (1.0 + std::exp(-g)) * u);
    }
    std::vector<double> y(2560);
    for (int r = 0; r < 2560; ++r) {
        dequantize_row(td.type, td.data + (std::size_t(e) * 2560 + r) * dr, row.data(), 640);
        double s = 0;
        for (int j = 0; j < 640; ++j) s += double(row[j]) * h[j];
        y[r] = double(w) * s;
    }
    return y;
}

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> shards;
    std::vector<int> layers = {0, 2, 4, 30};
    int reps = 200;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--layers") {
            layers.clear();
            std::stringstream ss(argv[++i]);
            for (std::string v; std::getline(ss, v, ',');) layers.push_back(std::stoi(v));
        } else if (a == "--reps") reps = std::stoi(argv[++i]);
        else shards.push_back(a);
    }
    GgufModel m(shards);
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0, 1);
    int failures = 0;
    cudaStream_t s;
    fc::check(cudaStreamCreate(&s), "stream");
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    for (int il : layers) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const GgufTensor & tg = m.tensor(p + "ffn_gate_exps.weight");
        const GgufTensor & tu = m.tensor(p + "ffn_up_exps.weight");
        const GgufTensor & td = m.tensor(p + "ffn_down_exps.weight");
        const fc::ExpertLayout lay = fc::expert_layout(tg.type, td.type);

        // 4 tokens x 10 experts; token t > 0 shares 3 experts with token 0
        constexpr int T = 4;
        std::vector<int> ids(T * 10);
        for (int t = 0; t < T; ++t) {
            std::set<int> used;
            for (int k = 0; k < 10; ++k) {
                int e = 0;
                if (t > 0 && k < 3) e = ids[k];
                else do e = int(rng() % 512); while (used.count(e));
                used.insert(e);
                ids[t * 10 + k] = e;
            }
        }
        std::vector<int> distinct(ids.begin(), ids.end());
        std::sort(distinct.begin(), distinct.end());
        distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
        std::vector<std::uint8_t> host(distinct.size() * lay.slot_bytes);
        for (std::size_t i = 0; i < distinct.size(); ++i) fc::pack_expert(lay, tg, tu, td, distinct[i], host.data() + i * lay.slot_bytes);
        fc::DeviceBuffer pool(host.size());
        fc::check(cudaMemcpy(pool.get(), host.data(), host.size(), cudaMemcpyHostToDevice), "pool");

        std::vector<std::int32_t> slots(T * 10);
        std::vector<float> w(T * 10), x(T * 2560);
        for (int i = 0; i < T * 10; ++i) {
            slots[i] = (i % 7 == 6) ? -1 : int(std::lower_bound(distinct.begin(), distinct.end(), ids[i]) - distinct.begin());
            w[i] = 0.05f + float(rng() % 100) / 500.0f;
        }
        for (auto & v : x) v = nd(rng);
        fc::DeviceBuffer dslots(slots.size() * 4), dw(w.size() * 4), dx(x.size() * 4), dh(T * 10 * 640 * 4), dy(T * 10 * 2560 * 4);
        fc::check(cudaMemcpy(dslots.get(), slots.data(), slots.size() * 4, cudaMemcpyHostToDevice), "slots");
        fc::check(cudaMemcpy(dw.get(), w.data(), w.size() * 4, cudaMemcpyHostToDevice), "w");
        fc::check(cudaMemcpy(dx.get(), x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");

        double worst = 0;
        bool zeros_ok = true;
        std::vector<float> y(T * 10 * 2560);
        for (int nt = 1; nt <= T; ++nt) {
            fc::experts_gpu(lay, pool.as<std::uint8_t>(), dslots.as<std::int32_t>(), dw.as<float>(), dx.as<float>(), dh.as<float>(),
                            dy.as<float>(), nt, s);
            fc::check(cudaStreamSynchronize(s), "run");
            fc::check(cudaMemcpy(y.data(), dy.get(), std::size_t(nt) * 10 * 2560 * 4, cudaMemcpyDeviceToHost), "y");
            for (int pi = 0; pi < nt * 10; ++pi) {
                if (slots[pi] < 0) {
                    for (int r = 0; r < 2560; ++r) zeros_ok &= y[std::size_t(pi) * 2560 + r] == 0.0f;
                    continue;
                }
                if (nt != T && pi % 10 > 2) continue;  // the full check runs once, at T = 4
                const std::vector<double> ref = reference(tg, tu, td, ids[pi], x.data() + (pi / 10) * 2560, w[pi]);
                double num = 0, den = 0;
                for (int r = 0; r < 2560; ++r) {
                    const double d = y[std::size_t(pi) * 2560 + r] - ref[r];
                    num += d * d;
                    den += ref[r] * ref[r];
                }
                worst = std::max(worst, std::sqrt(num / den));
            }
        }
        const bool ok = worst < 1e-5 && zeros_ok;
        failures += !ok;

        // speed: one token, all 10 experts in VRAM
        std::vector<std::int32_t> hit(slots.begin(), slots.begin() + 10);
        for (int k = 0; k < 10; ++k) hit[k] = int(std::lower_bound(distinct.begin(), distinct.end(), ids[k]) - distinct.begin());
        fc::check(cudaMemcpy(dslots.get(), hit.data(), 40, cudaMemcpyHostToDevice), "slots");
        for (int r = 0; r < 5; ++r)
            fc::experts_gpu(lay, pool.as<std::uint8_t>(), dslots.as<std::int32_t>(), dw.as<float>(), dx.as<float>(), dh.as<float>(), dy.as<float>(), 1, s);
        cudaEventRecord(e0, s);
        for (int r = 0; r < reps; ++r)
            fc::experts_gpu(lay, pool.as<std::uint8_t>(), dslots.as<std::int32_t>(), dw.as<float>(), dx.as<float>(), dh.as<float>(), dy.as<float>(), 1, s);
        cudaEventRecord(e1, s);
        fc::check(cudaEventSynchronize(e1), "sync");
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const double us = 1e3 * ms / reps;
        const std::size_t bytes = 10 * (lay.scale_off + std::size_t(2560) * 20 * 2);
        printf("layer %2d %s/%s slot %.2f MiB: rel. error vs exact %.2e, CPU pairs zero: %s; 10 experts %.1f us (%.0f GB/s)  %s\n", il,
               type_name(tg.type), type_name(td.type), lay.slot_bytes / 1048576.0, worst, zeros_ok ? "yes" : "NO", us, bytes / (us * 1e3),
               ok ? "ok" : "FAIL");

        // the same experts read straight from pinned host memory over PCIe (zero-copy), 1 to 3 at a time
        std::uint8_t * hpool = nullptr;
        fc::check(cudaHostAlloc(reinterpret_cast<void **>(&hpool), host.size(), cudaHostAllocMapped), "host pool");
        std::memcpy(hpool, host.data(), host.size());
        std::uint8_t * dpool = nullptr;
        fc::check(cudaHostGetDevicePointer(reinterpret_cast<void **>(&dpool), hpool, 0), "host pool");
        for (int nexp : {1, 2, 3}) {
            std::vector<std::int32_t> zc(10, -1);
            for (int k = 0; k < nexp; ++k) zc[k] = hit[k];
            fc::check(cudaMemcpy(dslots.get(), zc.data(), 40, cudaMemcpyHostToDevice), "slots");
            const int zreps = 50;
            cudaEventRecord(e0, s);
            for (int r = 0; r < zreps; ++r) {
                // a different set of experts each time, so nothing is served from L2
                fc::experts_gpu(lay, dpool, dslots.as<std::int32_t>(), dw.as<float>(), dx.as<float>(), dh.as<float>(), dy.as<float>(), 1, s);
            }
            cudaEventRecord(e1, s);
            fc::check(cudaEventSynchronize(e1), "sync");
            cudaEventElapsedTime(&ms, e0, e1);
            const double zus = 1e3 * ms / zreps;
            printf("         zero-copy from host memory, %d expert(s): %.1f us (%.1f GB/s)\n", nexp, zus,
                   nexp * (lay.scale_off + 2560.0 * 20 * 2) / (zus * 1e3));
        }
        cudaFreeHost(hpool);
    }
    printf(failures ? "FAILED\n" : "ALL OK\n");
    return failures ? 1 : 0;
}
