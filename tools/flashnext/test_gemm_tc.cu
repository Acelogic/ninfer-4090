// Tests and benchmarks the tensor-core prompt GEMM for Q8_0 weights (gemm_q8_tc) against cuBLAS SGEMM on the
// model's real matrices: for sampled rows, the relative error of both against a double-precision product of
// the exact dequantized weights; and the time of each for 256, 2048 and 8192 tokens.
// Usage: test_gemm_tc <shards...> [--sizes 256,2048,8192]
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "flashnext/cuda/device.h"
#include "flashnext/cuda/gemm.h"
#include "flashnext/cuda/gemv.h"
#include "flashnext/gguf.h"
#include "flashnext/quants.h"
#include "flashnext/shards.h"

using namespace ninfer::flashnext;
namespace fc = ninfer::flashnext::cuda;

int main(int argc, char ** argv) {
    try {
        std::vector<std::string> shards;
        std::vector<int> sizes{256, 2048, 8192};
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--sizes") {
                sizes.clear();
                std::stringstream ss(argv[++i]);
                for (std::string t; std::getline(ss, t, ',');) sizes.push_back(std::stoi(t));
            } else {
                shards.push_back(a);
            }
        }
        if (shards.size() == 1) shards = gguf_shard_paths(shards[0]);
        GgufModel model(shards);
        cudaStream_t s = nullptr;
        fc::check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        const char * names[] = {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight", "blk.3.attn_q.weight",
                                "blk.0.hc_attn_down.weight", "blk.0.hc_attn_up.weight", "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_down_shexp.weight",
                                "blk.47.hc_ffn_up.weight"};
        int max_t = *std::max_element(sizes.begin(), sizes.end());
        fc::Gemm sg(std::size_t(12288) * 2560, s), tc(std::size_t(12288) * 2560, s);
        tc.set_tensor_cores(true);
        cudaEvent_t e0, e1, e2;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        cudaEventCreate(&e2);
        std::mt19937 rng(7);
        int failures = 0;
        double t_sg = 0, t_tc = 0;
        for (const char * name : names) {
            const GgufTensor & t = model.tensor(name);
            fc::DeviceWeight w = fc::upload_gemv_weight(t);
            const int K = w.view.k, N = w.view.n;
            // activations: rows of very different magnitudes, a few large outliers
            std::normal_distribution<float> nd(0.f, 1.f);
            std::vector<float> x(std::size_t(max_t) * K);
            for (int r = 0; r < max_t; ++r) {
                const float mag = std::exp2(float(int(rng() % 20) - 10));
                for (int k = 0; k < K; ++k) x[std::size_t(r) * K + k] = mag * nd(rng) * (rng() % 97 == 0 ? 30.f : 1.f);
            }
            fc::DeviceBuffer dx(x.size() * 4), y1(std::size_t(max_t) * N * 4), y2(std::size_t(max_t) * N * 4);
            fc::check(cudaMemcpy(dx.get(), x.data(), dx.bytes(), cudaMemcpyHostToDevice), "x");
            std::vector<float> wrow(static_cast<std::size_t>(K));
            const std::size_t rb = row_bytes(t.type, K);
            for (int T : sizes) {
                sg.run(w.view, dx.as<float>(), y1.as<float>(), T);
                tc.run(w.view, dx.as<float>(), y2.as<float>(), T);
                fc::check(cudaStreamSynchronize(s), "run");
                // sampled rows against double precision
                std::vector<float> a(std::size_t(T) * N), b(std::size_t(T) * N);
                fc::check(cudaMemcpy(a.data(), y1.get(), a.size() * 4, cudaMemcpyDeviceToHost), "y1");
                fc::check(cudaMemcpy(b.data(), y2.get(), b.size() * 4, cudaMemcpyDeviceToHost), "y2");
                double ea = 0, eb = 0, den = 0;
                const int rows[4] = {0, T / 3, (2 * T) / 3, T - 1};
                for (int n = 0; n < N; n += std::max(1, N / 512)) {
                    dequantize_row(t.type, t.data + std::size_t(n) * rb, wrow.data(), K);
                    for (int r : rows) {
                        double ref = 0;
                        for (int k = 0; k < K; ++k) ref += double(wrow[std::size_t(k)]) * double(x[std::size_t(r) * K + k]);
                        const double da = a[std::size_t(r) * N + n] - ref, db = b[std::size_t(r) * N + n] - ref;
                        ea += da * da;
                        eb += db * db;
                        den += ref * ref;
                    }
                }
                const int reps = T >= 4096 ? 5 : 20;
                cudaEventRecord(e0, s);
                for (int i = 0; i < reps; ++i) sg.run(w.view, dx.as<float>(), y1.as<float>(), T);
                cudaEventRecord(e1, s);
                for (int i = 0; i < reps; ++i) tc.run(w.view, dx.as<float>(), y2.as<float>(), T);
                cudaEventRecord(e2, s);
                fc::check(cudaStreamSynchronize(s), "bench");
                float ms1 = 0, ms2 = 0;
                cudaEventElapsedTime(&ms1, e0, e1);
                cudaEventElapsedTime(&ms2, e1, e2);
                const double flop = 2.0 * T * double(N) * K;
                const double ra = std::sqrt(ea / den), rbb = std::sqrt(eb / den);
                std::printf("%-28s %5d x %5d  T=%5d: SGEMM %.2e rel, %7.3f ms (%5.1f TFLOPS); tensor cores %.2e rel, %7.3f ms (%5.1f TFLOPS)%s\n", name, N, K, T,
                            ra, ms1 / reps, flop / (ms1 / reps) / 1e9, rbb, ms2 / reps, flop / (ms2 / reps) / 1e9, rbb > 2e-6 ? "  <-- FAIL" : "");
                if (T == sizes.back()) {
                    t_sg += ms1 / reps;
                    t_tc += ms2 / reps;
                }
                failures += rbb > 2e-6;
            }
        }
        std::printf("total at T=%d: SGEMM %.2f ms, tensor cores %.2f ms\n%s\n", sizes.back(), t_sg, t_tc, failures ? "FAILED" : "all passed");
        return failures ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
