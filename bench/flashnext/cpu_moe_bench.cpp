// CPU expert benchmark for Qwen3.8-Flash-Next (qwen4exp) on the decode path.
//
// 1. Measures how fast this machine can stream RAM with all cores (the decode ceiling for
//    experts that live in system memory).
// 2. Loads the real routed-expert tensors of several layers from the GGUF and runs the decode
//    work for one token per layer: 10 random experts, gate+up (IQ3_S here), SwiGLU, down
//    (IQ4_NL), weighted sum. The dot products use ggml-cpu's own kernels; threading is ours.
//
// Usage: cpu_moe_bench <gguf shard 1> [<shard 2> ...] [--layers N] [--threads T] [--tokens K]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <immintrin.h>

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "q4l_kernels.h"

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

// ---- a tiny spin-barrier thread pool: one job per call, every thread runs fn(tid) ----
struct Pool {
    explicit Pool(int n) : n(n) {
        for (int t = 1; t < n; ++t) workers.emplace_back([this, t] { loop(t); });
    }
    ~Pool() {
        stop = true;
        gen.fetch_add(1, std::memory_order_release);
        for (auto & w : workers) w.join();
    }
    template <class F> void run(F && f) {
        job = [&](int t) { f(t); };
        done.store(0, std::memory_order_relaxed);
        gen.fetch_add(1, std::memory_order_release);
        job(0);
        while (done.load(std::memory_order_acquire) != n - 1) _mm_pause();
    }
    void loop(int t) {
        unsigned seen = 0;
        for (;;) {
            unsigned g;
            while ((g = gen.load(std::memory_order_acquire)) == seen) _mm_pause();
            seen = g;
            if (stop) return;
            job(t);
            done.fetch_add(1, std::memory_order_acq_rel);
        }
    }
    int n;
    std::vector<std::thread> workers;
    std::function<void(int)> job;
    std::atomic<unsigned> gen{0};
    std::atomic<int> done{0};
    std::atomic<bool> stop{false};
};

static void * aligned(size_t bytes) { return _aligned_malloc(bytes, 64); }

// ---- 1. RAM bandwidth ----
static void bandwidth(Pool & pool, int threads) {
    const size_t bytes = size_t(8) << 30;
    auto * buf = (uint8_t *) aligned(bytes);
    pool.run([&](int t) {  // touch in parallel so pages spread like real weights
        size_t a = bytes / threads * t, b = (t == threads - 1) ? bytes : bytes / threads * (t + 1);
        memset(buf + a, t + 1, b - a);
    });
    std::vector<double> sums(threads);
    for (int rep = 0; rep < 3; ++rep) {
        auto t0 = clk::now();
        pool.run([&](int t) {
            size_t a = bytes / threads * t, b = (t == threads - 1) ? bytes : bytes / threads * (t + 1);
            __m256i acc = _mm256_setzero_si256();
            for (size_t i = a; i < b; i += 128) {
                acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *) (buf + i)));
                acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *) (buf + i + 32)));
                acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *) (buf + i + 64)));
                acc = _mm256_add_epi64(acc, _mm256_load_si256((const __m256i *) (buf + i + 96)));
            }
            alignas(32) int64_t v[4];
            _mm256_store_si256((__m256i *) v, acc);
            sums[t] = double(v[0] + v[1] + v[2] + v[3]);
        });
        double s = secs(t0, clk::now());
        printf("ram read  %d threads: %6.1f GB/s\n", threads, bytes / s / 1e9);
    }
    _aligned_free(buf);
}

// ---- 2. expert tensors from the GGUF ----
struct Tensor {
    ggml_type type = GGML_TYPE_COUNT;
    int64_t ne0 = 0, ne1 = 0, ne2 = 0;  // [in, out, n_expert]
    size_t row_bytes = 0, expert_bytes = 0;
    uint8_t * data = nullptr;
};

static bool load_tensor(const std::vector<std::string> & shards, const std::string & name, Tensor & out) {
    for (const auto & path : shards) {
        gguf_init_params p{true, nullptr};
        gguf_context * ctx = gguf_init_from_file(path.c_str(), p);
        if (!ctx) continue;
        int64_t id = gguf_find_tensor(ctx, name.c_str());
        if (id < 0) { gguf_free(ctx); continue; }
        out.type = gguf_get_tensor_type(ctx, id);
        size_t off = gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, id);
        size_t size = gguf_get_tensor_size(ctx, id);
        gguf_free(ctx);
        out.data = (uint8_t *) aligned(size);
        FILE * f = fopen(path.c_str(), "rb");
        _fseeki64(f, (long long) off, SEEK_SET);
        size_t got = fread(out.data, 1, size, f);
        fclose(f);
        if (got != size) { fprintf(stderr, "short read %s\n", name.c_str()); return false; }
        out.expert_bytes = size;  // divided by n_expert by the caller
        return true;
    }
    return false;
}

int main(int argc, char ** argv) {
    std::vector<std::string> shards;
    int layers = 8, threads = 16, tokens = 64;
    bool q4l = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--layers") layers = atoi(argv[++i]);
        else if (a == "--threads") threads = atoi(argv[++i]);
        else if (a == "--tokens") tokens = atoi(argv[++i]);
        else if (a == "--q4l") q4l = true;
        else shards.push_back(a);
    }
    ggml_cpu_init();
    printf("ggml-cpu: avx2=%d avx512=%d avx512_vnni=%d avx512_bf16=%d avx512_vbmi=%d\n",
           ggml_cpu_has_avx2(), ggml_cpu_has_avx512(), ggml_cpu_has_avx512_vnni(), ggml_cpu_has_avx512_bf16(), ggml_cpu_has_avx512_vbmi());

    for (int t : {1, 4, 8, 16}) {
        if (t > threads) break;
        Pool p(t);
        bandwidth(p, t);
    }
    if (shards.empty()) return 0;

    const int64_t n_embd = 2560, n_ff = 640, n_expert = 512, n_used = 10;
    struct Layer { Tensor gate, up, down; };
    std::vector<Layer> L(layers);
    size_t loaded = 0;
    for (int il = 0; il < layers; ++il) {
        // spread the sampled layers across the model, like real decode
        int src = il * (48 / layers);
        for (auto [name, t] : {std::pair<const char *, Tensor *>{"ffn_gate_exps", &L[il].gate}, {"ffn_up_exps", &L[il].up}, {"ffn_down_exps", &L[il].down}}) {
            std::string full = "blk." + std::to_string(src) + "." + name + ".weight";
            if (!load_tensor(shards, full, *t)) { fprintf(stderr, "missing %s\n", full.c_str()); return 1; }
            loaded += t->expert_bytes;
            t->expert_bytes /= n_expert;
        }
        L[il].gate.ne0 = n_embd; L[il].gate.ne1 = n_ff;
        L[il].up.ne0 = n_embd;   L[il].up.ne1 = n_ff;
        L[il].down.ne0 = n_ff;   L[il].down.ne1 = n_embd;
        for (Tensor * t : {&L[il].gate, &L[il].up, &L[il].down}) t->row_bytes = ggml_row_size(t->type, t->ne0);
    }
    printf("loaded %d layers: %.2f GiB; types gate=%s up=%s down=%s (layer %d)\n", layers, loaded / 1073741824.0,
           ggml_type_name(L[0].gate.type), ggml_type_name(L[0].up.type), ggml_type_name(L[0].down.type), 0);

    // Q4L: repack gate/up from IQ3_S, losslessly, and check both kernels against a float reference.
    std::vector<std::vector<block_q4l>> qg(layers), qu(layers);
    const int64_t nb_row = n_embd / 256;
    if (q4l) {
        auto r0 = clk::now();
        for (int il = 0; il < layers; ++il) {
            for (auto [src, dst] : {std::pair<Tensor *, std::vector<block_q4l> *>{&L[il].gate, &qg[il]}, {&L[il].up, &qu[il]}}) {
                if (src->type != GGML_TYPE_IQ3_S) { fprintf(stderr, "layer %d gate/up is %s, not iq3_s\n", il, ggml_type_name(src->type)); return 1; }
                dst->resize(size_t(n_expert) * n_ff * nb_row);
                repack_iq3s_to_q4l((const block_iq3s_src *) src->data, dst->data(), int64_t(dst->size()));
            }
        }
        printf("repacked gate/up to q4l in %.1f s (%.2f GiB)\n", secs(r0, clk::now()), 2.0 * layers * n_expert * n_ff * nb_row * sizeof(block_q4l) / 1073741824.0);
        std::mt19937 vr(7);
        std::vector<float> xv(n_embd), wrow(n_embd);
        for (auto & v : xv) v = std::normal_distribution<float>(0, 1)(vr);
        ActQ8 act;
        quantize_act(xv.data(), int(n_embd), act);
        const auto * tg = ggml_get_type_traits_cpu(GGML_TYPE_IQ3_S);
        std::vector<uint8_t> xq(ggml_row_size(tg->vec_dot_type, n_embd));
        ggml_get_type_traits_cpu(tg->vec_dot_type)->from_float(xv.data(), xq.data(), n_embd);
        double err_ours = 0, err_ggml = 0, mag = 0;
        for (int t = 0; t < 256; ++t) {
            const int e = int(vr() % n_expert), row = int(vr() % n_ff);
            const uint8_t * src = L[0].gate.data + size_t(e) * L[0].gate.expert_bytes + size_t(row) * L[0].gate.row_bytes;
            ggml_get_type_traits(GGML_TYPE_IQ3_S)->to_float(src, wrow.data(), n_embd);
            double ref = 0;
            for (int j = 0; j < n_embd; ++j) ref += double(wrow[j]) * xv[j];
            float og, ou, gg;
            const block_q4l * gq = qg[0].data() + (size_t(e) * n_ff + row) * nb_row;
            q4l_dot2(gq, gq, act, int(nb_row), og, ou);
            tg->vec_dot(int(n_embd), &gg, 0, src, 0, xq.data(), 0, 1);
            err_ours += std::fabs(og - ref); err_ggml += std::fabs(gg - ref); mag += std::fabs(ref);
        }
        printf("accuracy vs float reference (256 rows): q4l mean rel err %.4f%%, ggml iq3_s %.4f%%\n", 100 * err_ours / mag, 100 * err_ggml / mag);
    }

    Pool pool(threads);
    std::mt19937 rng(1234);
    std::vector<float> x(n_embd), out(n_embd), h(n_used * n_ff), gbuf(n_used * n_ff), ubuf(n_used * n_ff);
    std::vector<float> w(n_used);
    for (auto & v : x) v = std::normal_distribution<float>(0, 1)(rng);

    double best = 1e9, total = 0, t_gu = 0, t_dn = 0, t_misc = 0;
    size_t bytes_per_token = 0, bytes_gu = 0, bytes_dn = 0;
    for (int tok = 0; tok < tokens; ++tok) {
        auto t0 = clk::now();
        size_t bytes = 0;
        for (int il = 0; il < layers; ++il) {
            Layer & ly = L[il];
            int e[n_used];
            for (int k = 0; k < n_used; ++k) { e[k] = int(rng() % n_expert); w[k] = 0.1f; }
            const auto * tg = ggml_get_type_traits_cpu(ly.gate.type);
            const auto * td = ggml_get_type_traits_cpu(ly.down.type);
            // quantize the input once for gate/up
            std::vector<uint8_t> xq(ggml_row_size(tg->vec_dot_type, n_embd));
            ggml_get_type_traits_cpu(tg->vec_dot_type)->from_float(x.data(), xq.data(), n_embd);
            // gate and up: n_used*n_ff rows each, split across threads
            const int64_t rows = n_used * n_ff;
            auto p0 = clk::now();
            ActQ8 act;
            if (q4l) quantize_act(x.data(), int(n_embd), act);
            if (q4l) pool.run([&](int t) {
                int64_t a = rows * t / threads, b = rows * (t + 1) / threads;
                for (int64_t r = a; r < b; ++r) {
                    const int k = int(r / n_ff), row = int(r % n_ff);
                    const size_t off = (size_t(e[k]) * n_ff + row) * nb_row;
                    q4l_dot2(qg[il].data() + off, qu[il].data() + off, act, int(nb_row), gbuf[r], ubuf[r]);
                }
            });
            else pool.run([&](int t) {
                int64_t a = rows * t / threads, b = rows * (t + 1) / threads;
                for (int64_t r = a; r < b; ++r) {
                    int k = int(r / n_ff), row = int(r % n_ff);
                    const uint8_t * gr = ly.gate.data + size_t(e[k]) * ly.gate.expert_bytes + size_t(row) * ly.gate.row_bytes;
                    const uint8_t * ur = ly.up.data + size_t(e[k]) * ly.up.expert_bytes + size_t(row) * ly.up.row_bytes;
                    tg->vec_dot(int(n_embd), &gbuf[r], 0, gr, 0, xq.data(), 0, 1);
                    ggml_get_type_traits_cpu(ly.up.type)->vec_dot(int(n_embd), &ubuf[r], 0, ur, 0, xq.data(), 0, 1);
                }
            });
            auto p1 = clk::now();
            for (int64_t r = 0; r < rows; ++r) { float g = gbuf[r]; h[r] = g / (1.0f + std::exp(-g)) * ubuf[r]; }
            // quantize each expert's h for down
            const size_t hq_row = ggml_row_size(td->vec_dot_type, n_ff);
            std::vector<uint8_t> hq(hq_row * n_used);
            for (int k = 0; k < n_used; ++k) ggml_get_type_traits_cpu(td->vec_dot_type)->from_float(&h[k * n_ff], hq.data() + k * hq_row, n_ff);
            // down: each thread owns a slice of output rows and loops over the experts
            auto p2 = clk::now();
            pool.run([&](int t) {
                int64_t a = n_embd * t / threads, b = n_embd * (t + 1) / threads;
                for (int64_t r = a; r < b; ++r) {
                    float acc = 0;
                    for (int k = 0; k < n_used; ++k) {
                        float s;
                        const uint8_t * dr = ly.down.data + size_t(e[k]) * ly.down.expert_bytes + size_t(r) * ly.down.row_bytes;
                        td->vec_dot(int(n_ff), &s, 0, dr, 0, hq.data() + k * hq_row, 0, 1);
                        acc += w[k] * s;
                    }
                    out[r] = acc;
                }
            });
            auto p3 = clk::now();
            if (tok >= 4) { t_gu += secs(p0, p1); t_misc += secs(p1, p2); t_dn += secs(p2, p3); }
            const size_t gu_bytes = q4l ? size_t(2) * n_ff * nb_row * sizeof(block_q4l) : ly.gate.expert_bytes + ly.up.expert_bytes;
            if (tok == 0) { bytes_gu += n_used * gu_bytes; bytes_dn += n_used * ly.down.expert_bytes; }
            bytes += n_used * (gu_bytes + ly.down.expert_bytes);
        }
        double s = secs(t0, clk::now());
        if (tok >= 4) { best = std::min(best, s); total += s; }  // skip warm-up
        bytes_per_token = bytes;
    }
    const int measured = tokens - 4;
    const double mean = total / measured;
    const double scale = 48.0 / layers;  // project the sampled layers onto all 48
    printf("expert decode, %d threads: %.2f ms per token for %d layers (mean), %.1f GB/s effective\n",
           threads, mean * 1e3, layers, bytes_per_token / mean / 1e9);
    printf("  gate+up (%s): %.2f ms/token %.1f GB/s | swiglu+quant: %.2f ms | down (%s): %.2f ms/token %.1f GB/s\n",
           q4l ? "q4l" : ggml_type_name(L[0].gate.type), t_gu / measured * 1e3, bytes_gu / (t_gu / measured) / 1e9, t_misc / measured * 1e3,
           ggml_type_name(L[0].down.type), t_dn / measured * 1e3, bytes_dn / (t_dn / measured) / 1e9);
    printf("projected to 48 layers: %.1f ms/token -> %.1f tok/s from CPU experts alone (best %.1f tok/s)\n",
           mean * scale * 1e3, 1.0 / (mean * scale), 1.0 / (best * scale));
    return 0;
}
