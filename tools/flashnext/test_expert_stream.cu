// Tests the prompt path's expert streaming pieces on real experts:
//   convert:  every expert of the chosen layers is copied from CpuExperts' resident copy (registered with
//             cudaHostRegister, as the engine does) to the GPU and converted into the slot layout; each slot
//             must equal pack_expert's image from the GGUF byte for byte. Reports the copy and conversion rates.
//   phased:   experts_phased_* over key ranges (cached and "streamed" experts in separate pools, keys in
//             a processing order unrelated to the expert ids) must give bitwise the output of
//             experts_gpu_batch on one pool, for several token counts; reports both speeds.
// Usage: test_expert_stream <shards...> [--layers 0,2,4] [--sizes 256,2048,8192] [--group 32]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "flashnext/cpu_experts.h"
#include "flashnext/cuda/device.h"
#include "flashnext/cuda/expert_stream.h"
#include "flashnext/cuda/experts.h"
#include "flashnext/cuda/experts_batch.h"
#include "flashnext/cuda/ops.h"
#include "flashnext/gguf.h"
#include "flashnext/shards.h"

using namespace ninfer::flashnext;
namespace fc = ninfer::flashnext::cuda;
using clk = std::chrono::steady_clock;

namespace {

std::vector<int> parse_list(const std::string & s) {
    std::vector<int> v;
    std::stringstream ss(s);
    for (std::string t; std::getline(ss, t, ',');) v.push_back(std::stoi(t));
    return v;
}

double ms_since(clk::time_point t0) { return std::chrono::duration<double, std::milli>(clk::now() - t0).count(); }

}  // namespace

int main(int argc, char ** argv) {
    try {
        std::vector<std::string> shards;
        std::vector<int> layers{0, 2, 4}, sizes{256, 2048, 8192};
        int group = 32;
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--layers") layers = parse_list(argv[++i]);
            else if (a == "--sizes") sizes = parse_list(argv[++i]);
            else if (a == "--group") group = std::stoi(argv[++i]);
            else shards.push_back(a);
        }
        if (shards.size() == 1) shards = gguf_shard_paths(shards[0]);
        GgufModel model(shards);
        CpuExpertsConfig cfg;
        cfg.layers = layers;
        cfg.pin_threads = false;
        CpuExperts cpu(model, cfg);
        fc::init_kernels();
        fc::experts_batch_init();
        fc::expert_stream_init();
        cudaStream_t s = nullptr;
        fc::check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        int failures = 0;

        for (int il : layers) {
            const std::string b = "blk." + std::to_string(il) + ".";
            const GgufTensor & tg = model.tensor(b + "ffn_gate_exps.weight");
            const GgufTensor & tu = model.tensor(b + "ffn_up_exps.weight");
            const GgufTensor & td = model.tensor(b + "ffn_down_exps.weight");
            const CpuExperts::HostLayer H = cpu.host_layer(il);
            fc::HostExpertFormat F;
            F.gate_q4x = H.gate_q4x;
            F.down_q8 = H.down_q8;
            F.gate_bytes = H.gate_bytes;
            F.down_bytes = H.down_bytes;
            const fc::ExpertLayout L = fc::stream_layout(F);
            const fc::ExpertLayout Lg = fc::expert_layout(tg.type, td.type);
            if (L.slot_bytes != Lg.slot_bytes) throw std::runtime_error("stream layout differs from the GGUF's");
            std::printf("layer %d: %s gate/up, %s down; host blob %zu B, slot %zu B\n", il, F.gate_q4x ? "Q4X" : "Q4L", F.down_q8 ? "Q8_0" : "IQ4L",
                        F.blob_bytes(), L.slot_bytes);

            // register the layer's resident copy, as the engine does
            auto t0 = clk::now();
            const cudaError_t reg = cudaHostRegister(const_cast<std::uint8_t *>(H.base), H.bytes, cudaHostRegisterPortable);
            std::printf("  cudaHostRegister %.2f GiB: %s (%.0f ms)\n", H.bytes / 1073741824.0, reg == cudaSuccess ? "ok" : cudaGetErrorString(reg),
                        ms_since(t0));
            if (reg != cudaSuccess) {
                cudaGetLastError();
                ++failures;
            }

            // all 512 experts: one copy of the whole layer, then conversion in batches of kMaxConvert
            fc::DeviceBuffer raw(H.stride * fc::kExperts), pool(L.slot_bytes * fc::kExperts);
            cudaEvent_t e0, e1, e2;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventCreate(&e2);
            cudaEventRecord(e0, s);
            fc::check(cudaMemcpyAsync(raw.get(), H.base, H.stride * fc::kExperts, cudaMemcpyHostToDevice, s), "copy");
            cudaEventRecord(e1, s);
            for (int e0i = 0; e0i < fc::kExperts; e0i += fc::kMaxConvert) {
                fc::ConvertBatch cb;
                cb.n = std::min(fc::kMaxConvert, fc::kExperts - e0i);
                for (int j = 0; j < cb.n; ++j) {
                    cb.src[j] = raw.as<std::uint8_t>() + std::size_t(e0i + j) * H.stride;
                    cb.dst[j] = pool.as<std::uint8_t>() + std::size_t(e0i + j) * L.slot_bytes;
                }
                fc::convert_experts(F, L, cb, s);
            }
            cudaEventRecord(e2, s);
            fc::check(cudaStreamSynchronize(s), "convert");
            float copy_ms = 0, conv_ms = 0;
            cudaEventElapsedTime(&copy_ms, e0, e1);
            cudaEventElapsedTime(&conv_ms, e1, e2);
            std::printf("  copy %.1f MB in %.1f ms (%.1f GB/s); convert 512 experts in %.1f ms (%.1f us each, %.1f GB/s in)\n",
                        H.stride * 512 / 1e6, copy_ms, H.stride * 512 / 1e6 / copy_ms, conv_ms, 1e3 * conv_ms / 512, H.stride * 512 / 1e6 / conv_ms);

            // byte comparison against pack_expert from the GGUF
            std::vector<std::uint8_t> got(L.slot_bytes), want(L.slot_bytes);
            int bad = 0;
            for (int e = 0; e < fc::kExperts; ++e) {
                fc::check(cudaMemcpy(got.data(), pool.as<std::uint8_t>() + std::size_t(e) * L.slot_bytes, L.slot_bytes, cudaMemcpyDeviceToHost), "read");
                fc::pack_expert(Lg, tg, tu, td, e, want.data());
                if (got != want) {
                    if (bad < 3) {
                        std::size_t i = 0;
                        while (got[i] == want[i]) ++i;
                        std::printf("  expert %d differs at byte %zu (%d vs %d)\n", e, i, got[i], want[i]);
                    }
                    ++bad;
                }
            }
            std::printf("  convert: %d of 512 experts differ from pack_expert%s\n", bad, bad ? "  <-- FAIL" : "");
            failures += bad != 0;

            // phased vs one-pool batch: the pool holds every expert at slot = id; the phased run sees a
            // "resident" set (every 7th expert) first, then the others in groups, from a second copy
            fc::DeviceBuffer pool2(L.slot_bytes * fc::kExperts);
            fc::check(cudaMemcpy(pool2.get(), pool.get(), pool.bytes(), cudaMemcpyDeviceToDevice), "pool2");
            std::vector<int> order;
            for (int e = 0; e < fc::kExperts; e += 7) order.push_back(e);
            const int n_res = int(order.size());
            for (int e = 0; e < fc::kExperts; ++e)
                if (e % 7) order.push_back(e);
            std::vector<std::int32_t> key_of(fc::kExperts);
            std::vector<const std::uint8_t *> ptrs(fc::kExperts);
            for (int k = 0; k < fc::kExperts; ++k) {
                key_of[std::size_t(order[std::size_t(k)])] = k;
                ptrs[std::size_t(k)] = (k < n_res ? pool.as<std::uint8_t>() : pool2.as<std::uint8_t>()) + std::size_t(order[std::size_t(k)]) * L.slot_bytes;
            }
            fc::DeviceBuffer dptrs(ptrs.size() * sizeof(void *)), dkeymap(fc::kExperts * sizeof(std::int32_t)), didmap(fc::kExperts * sizeof(std::int32_t));
            fc::check(cudaMemcpy(dptrs.get(), ptrs.data(), dptrs.bytes(), cudaMemcpyHostToDevice), "ptrs");
            fc::check(cudaMemcpy(dkeymap.get(), key_of.data(), dkeymap.bytes(), cudaMemcpyHostToDevice), "keys");
            std::vector<std::int32_t> ident(fc::kExperts);
            std::iota(ident.begin(), ident.end(), 0);
            fc::check(cudaMemcpy(didmap.get(), ident.data(), didmap.bytes(), cudaMemcpyHostToDevice), "ids");
            std::mt19937 rng(1234 + il);
            for (int T : sizes) {
                // Zipf-like routing over all experts, 10 distinct per token
                std::vector<double> zw(fc::kExperts);
                for (int r = 0; r < fc::kExperts; ++r) zw[std::size_t(r)] = 1.0 / (double(r) + 20.0);
                std::discrete_distribution<int> zipf(zw.begin(), zw.end());
                std::vector<int> perm(fc::kExperts);
                std::iota(perm.begin(), perm.end(), 0);
                std::shuffle(perm.begin(), perm.end(), rng);
                std::vector<std::int32_t> ids(std::size_t(T) * fc::kUsed);
                std::vector<float> w(ids.size()), x(std::size_t(T) * fc::kEmbd);
                std::normal_distribution<float> nd(0.f, 1.f);
                for (float & v : x) v = nd(rng);
                for (int t = 0; t < T; ++t) {
                    std::set<int> used;
                    for (int k = 0; k < fc::kUsed; ++k) {
                        int e;
                        do e = perm[std::size_t(zipf(rng))];
                        while (used.count(e));
                        used.insert(e);
                        ids[std::size_t(t) * fc::kUsed + std::size_t(k)] = e;
                        w[std::size_t(t) * fc::kUsed + std::size_t(k)] = 0.05f + 0.1f * float(k);
                    }
                }
                fc::DeviceBuffer d_ids(ids.size() * 4), d_slots(ids.size() * 4), d_keys(ids.size() * 4), d_w(w.size() * 4), d_x(x.size() * 4),
                    o1(x.size() * 4), o2(x.size() * 4), ws1(fc::experts_batch_workspace_bytes(T)), ws2(fc::experts_phased_workspace_bytes(T));
                fc::check(cudaMemcpy(d_ids.get(), ids.data(), d_ids.bytes(), cudaMemcpyHostToDevice), "ids");
                fc::check(cudaMemcpy(d_w.get(), w.data(), d_w.bytes(), cudaMemcpyHostToDevice), "w");
                fc::check(cudaMemcpy(d_x.get(), x.data(), d_x.bytes(), cudaMemcpyHostToDevice), "x");
                fc::moe_slots(d_ids.as<std::int32_t>(), didmap.as<std::int32_t>(), d_slots.as<std::int32_t>(), T, s);
                fc::moe_slots(d_ids.as<std::int32_t>(), dkeymap.as<std::int32_t>(), d_keys.as<std::int32_t>(), T, s);
                auto run1 = [&] {
                    fc::experts_gpu_batch(L, pool.as<std::uint8_t>(), T, d_x.as<float>(), d_slots.as<std::int32_t>(), d_w.as<float>(), o1.as<float>(),
                                          ws1.get(), s);
                };
                auto run2 = [&] {
                    fc::experts_phased_begin(T, d_x.as<float>(), d_keys.as<std::int32_t>(), ws2.get(), s);
                    fc::experts_phased_run(L, dptrs.as<const std::uint8_t *>(), 0, n_res, T, d_x.as<float>(), ws2.get(), s);
                    for (int k0 = n_res; k0 < fc::kExperts; k0 += group)
                        fc::experts_phased_run(L, dptrs.as<const std::uint8_t *>(), k0, std::min(fc::kExperts, k0 + group), T, d_x.as<float>(),
                                               ws2.get(), s);
                    fc::experts_phased_end(T, d_keys.as<std::int32_t>(), d_w.as<float>(), o2.as<float>(), ws2.get(), s);
                };
                run1();
                run2();
                fc::check(cudaStreamSynchronize(s), "batch");
                std::vector<float> r1(x.size()), r2(x.size());
                fc::check(cudaMemcpy(r1.data(), o1.get(), o1.bytes(), cudaMemcpyDeviceToHost), "o1");
                fc::check(cudaMemcpy(r2.data(), o2.get(), o2.bytes(), cudaMemcpyDeviceToHost), "o2");
                const bool same = std::memcmp(r1.data(), r2.data(), r1.size() * 4) == 0;
                double nz = 0;
                for (float v : r1) nz += v != 0.f;
                const int reps = T >= 4096 ? 3 : 10;
                cudaEventRecord(e0, s);
                for (int r = 0; r < reps; ++r) run1();
                cudaEventRecord(e1, s);
                for (int r = 0; r < reps; ++r) run2();
                cudaEventRecord(e2, s);
                fc::check(cudaStreamSynchronize(s), "bench");
                float a = 0, c = 0;
                cudaEventElapsedTime(&a, e0, e1);
                cudaEventElapsedTime(&c, e1, e2);
                std::printf("  T=%5d: phased (resident %d keys, groups of %d) vs one pool: %s; %.2f ms vs %.2f ms per layer (%.1f%% nonzero)\n", T,
                            n_res, group, same ? "bitwise identical" : "DIFFERENT  <-- FAIL", c / reps, a / reps, 100.0 * nz / double(r1.size()));
                failures += !same;
            }
            if (reg == cudaSuccess) cudaHostUnregister(const_cast<std::uint8_t *>(H.base));
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
            cudaEventDestroy(e2);
        }
        std::printf("%s\n", failures ? "FAILED" : "all passed");
        return failures ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
