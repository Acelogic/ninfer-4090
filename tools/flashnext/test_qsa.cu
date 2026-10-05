// Checks the QSA kernels (src/flashnext/cuda/qsa.cu) against a CPU port of the FP32 reference
// (ReferenceModel::Impl::qsa_select and attention in src/flashnext/reference.cpp), and times them.
//
//   1. random data: block keys and indexer queries must equal the port bit for bit, selections must be
//      identical (including constructed exact ties), attention is compared with a double-precision port
//   2. real data (--dumps): the reference's own intermediates on a long prompt (ref_generate --dump with
//      --chunk 4096), fed through the kernels the way the engine will call them (prefill chunks, then
//      decode steps): blocks, queries and selections must equal the reference's; attention error is
//      reported against the reference (FP32 K/V) and against the double port on the fp16 cache
//   3. timings at contexts of 2K, 32K and 262K tokens (synthetic data), decode and prefill
//   4. KV streaming parity (synthetic data): two attention layers run side by side as fully resident caches
//      and as KvStreamCache (host copy + a small page cache + staging), through prompt chunks that fit the
//      page cache, chunks beyond it (staged, or in groups), decode steps of 1-4 tokens, rolled-back drafts and jumps back to earlier
//      positions; every attention output must be bit-identical, the host copy must equal the resident cache,
//      and the page cache must never overflow. Also the MTP layer's K/V ring against a full cache.
//
// Usage: test_qsa [-m <GGUF shard 1> --dumps <dir> [--layers 3,23,47]] [--no-timing | --timing-only | --kv-only]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "flashnext/cuda/device.h"
#include "flashnext/cuda/ops.h"
#include "flashnext/cuda/qsa.h"
#include "flashnext/gguf.h"
#include "flashnext/kv_cache.h"
#include "flashnext/quants.h"

using namespace ninfer::flashnext;
using namespace ninfer::flashnext::cuda;
namespace fs = std::filesystem;

namespace {

constexpr float kEps = 1e-6f;          // f_norm_rms_eps of the model (9.999999974752427e-07 as float)
constexpr double kRopeBase = 1e7;
constexpr int kW = kQsaWidth;
const float kScale = 1.0f / 16.0f;     // 1/sqrt(head_dim)

int g_failures = 0;
void expect(bool ok, const char * what) {
    if (!ok) {
        std::printf("  FAIL: %s\n", what);
        ++g_failures;
    }
}

// ------------------------------------------------------------------------------------------------
// CPU port of the reference (same float operations in the same order)

std::vector<double> inv_freq_table() {
    std::vector<double> f(kQsaRot / 2);
    for (int i = 0; i < kQsaRot / 2; ++i) f[std::size_t(i)] = std::pow(kRopeBase, -2.0 * i / kQsaRot);
    return f;
}

void port_rms_norm(const float * x, float * y, int n, float eps, const float * w) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += double(x[i] * x[i]);
    const float mean = float(sum / double(n));
    const float scale = 1.0f / std::sqrt(mean + eps);
    for (int i = 0; i < n; ++i) y[i] = (x[i] * scale) * w[i];
}

void port_rope(float * x, std::int64_t pos, const std::vector<double> & inv_freq) {
    const int half = kQsaRot / 2;
    for (int i = 0; i < half; ++i) {
        const double theta = double(pos) * inv_freq[std::size_t(i)];
        const float c = float(std::cos(theta)), s = float(std::sin(theta));
        const float x0 = x[i], x1 = x[i + half];
        x[i] = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
    }
}

// The reference's AVX-512 dot over 128 floats, lane by lane: acc0[l] covers l, 32+l, 64+l, 96+l and
// acc1[l] covers 16+l, ...; then acc0 + acc1 and _mm512_reduce_add_ps's halving tree.
float port_dot128(const float * a, const float * b) {
    float acc[32];
    for (int j = 0; j < 32; ++j) {
        float c = std::fma(a[j], b[j], 0.0f);
        for (int k = 1; k < 4; ++k) c = std::fma(a[32 * k + j], b[32 * k + j], c);
        acc[j] = c;
    }
    float v[16];
    for (int l = 0; l < 16; ++l) v[l] = acc[l] + acc[16 + l];
    for (int len = 8; len >= 1; len /= 2) {
        for (int l = 0; l < len; ++l) v[l] = v[l] + v[l + len];
    }
    return v[0];
}

void port_block(const float * raw, std::int64_t b, const float * k_norm, const std::vector<double> & inv_freq, float * out) {
    float pooled[kQsaDim];
    for (int d = 0; d < kQsaDim; ++d) {
        float acc = raw[(b * kQsaRatio) * kQsaDim + d];
        for (int i = 1; i < kQsaRatio; ++i) acc = acc + raw[(b * kQsaRatio + i) * kQsaDim + d];
        pooled[d] = acc * (1.0f / float(kQsaRatio));
    }
    port_rms_norm(pooled, out, kQsaDim, kEps, k_norm);
    port_rope(out, b * kQsaRatio, inv_freq);
}

void port_query(float * q, std::int64_t pos, const float * q_norm, const std::vector<double> & inv_freq) {
    for (int h = 0; h < kQsaHeads; ++h) {
        port_rms_norm(q + h * kQsaDim, q + h * kQsaDim, kQsaDim, kEps, q_norm);
        port_rope(q + h * kQsaDim, pos, inv_freq);
    }
}

// cells of the query at pos (q: its normed and rotated [4][128]); exactly qsa_select of the reference
std::vector<std::int32_t> port_select(const float * q, const float * blocks, std::int64_t pos) {
    std::vector<std::int32_t> cells;
    if (pos + 1 <= kW) {
        for (std::int64_t p = 0; p <= pos; ++p) cells.push_back(std::int32_t(p));
        return cells;
    }
    const std::int64_t tail_start = (pos + 1) / kQsaRatio * kQsaRatio, n_blocks = tail_start / kQsaRatio;
    std::vector<std::pair<float, std::int32_t>> score(static_cast<std::size_t>(n_blocks));
    for (std::int64_t b = 0; b < n_blocks; ++b) {
        float s = 0.0f;
        for (int h = 0; h < kQsaHeads; ++h) s = s + std::max(0.0f, port_dot128(q + h * kQsaDim, blocks + b * kQsaDim));
        score[std::size_t(b)] = {s, std::int32_t(b)};
    }
    std::sort(score.begin(), score.end(), [](const auto & a, const auto & b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
    const std::int64_t want = kW - (pos + 1 - tail_start);
    for (std::int64_t i = 0; i < n_blocks && std::int64_t(cells.size()) < want; ++i) {
        const std::int64_t b = score[std::size_t(i)].second;
        for (int j = 0; j < kQsaRatio && std::int64_t(cells.size()) < want; ++j) cells.push_back(std::int32_t(b * kQsaRatio + j));
    }
    for (std::int64_t p = tail_start; p <= pos; ++p) cells.push_back(std::int32_t(p));
    std::sort(cells.begin(), cells.end());
    return cells;
}

// double-precision attention of one token: out [24][256]
void port_attention(const float * q, const float * gate, const std::vector<float> & kc, const std::vector<float> & vc,
                    const std::int32_t * cells, int n, double * out) {
    std::vector<double> s(static_cast<std::size_t>(n));
    for (int h = 0; h < kHeads; ++h) {
        const int hk = h / kGroup;
        double mx = -INFINITY;
        for (int j = 0; j < n; ++j) {
            const float * k = kc.data() + (std::size_t(cells[j]) * kKvHeads + hk) * kHeadDim;
            double d = 0.0;
            for (int i = 0; i < kHeadDim; ++i) d += double(q[h * kHeadDim + i]) * double(k[i]);
            s[std::size_t(j)] = d * double(kScale);
            mx = std::max(mx, s[std::size_t(j)]);
        }
        double sum = 0.0;
        for (int j = 0; j < n; ++j) sum += (s[std::size_t(j)] = std::exp(s[std::size_t(j)] - mx));
        for (int i = 0; i < kHeadDim; ++i) {
            double acc = 0.0;
            for (int j = 0; j < n; ++j) acc += s[std::size_t(j)] * double(vc[(std::size_t(cells[j]) * kKvHeads + hk) * kHeadDim + i]);
            const double g = double(gate[h * kHeadDim + i]);
            out[h * kHeadDim + i] = acc / sum / (1.0 + std::exp(-g));
        }
    }
}

// ------------------------------------------------------------------------------------------------
// device helpers

template <class T> void upload(const DeviceBuffer & d, const T * h, std::size_t n, std::size_t offset = 0) {
    check(cudaMemcpy(d.as<T>() + offset, h, n * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template <class T> void download(T * h, const DeviceBuffer & d, std::size_t n, std::size_t offset = 0) {
    check(cudaMemcpy(h, d.as<T>() + offset, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
}
void set_pos(const DeviceBuffer & d, std::int64_t pos) { upload(d, &pos, 1); }

std::vector<half> to_half(const float * x, std::size_t n) {
    std::vector<half> h(n);
    for (std::size_t i = 0; i < n; ++i) h[i] = __float2half_rn(x[i]);
    return h;
}
std::vector<float> half_round(const std::vector<float> & x) {
    std::vector<float> y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = __half2float(__float2half_rn(x[i]));
    return y;
}

struct Rng {
    std::mt19937_64 g;
    explicit Rng(std::uint64_t seed) : g(seed) {}
    float normal() { return std::normal_distribution<float>(0.0f, 1.0f)(g); }
    float uniform(float a, float b) { return std::uniform_real_distribution<float>(a, b)(g); }
    void fill_normal(std::vector<float> & v, float sd = 1.0f) { for (auto & x : v) x = normal() * sd; }
};

// relative L2 error per (token, head) row of [T][24][256]; returns the worst
double attn_error(const float * got, const double * want, int T) {
    double worst = 0.0;
    for (int r = 0; r < T * kHeads; ++r) {
        double num = 0.0, den = 0.0;
        for (int i = 0; i < kHeadDim; ++i) {
            const double d = double(got[r * kHeadDim + i]) - want[r * kHeadDim + i];
            num += d * d;
            den += want[r * kHeadDim + i] * want[r * kHeadDim + i];
        }
        worst = std::max(worst, std::sqrt(num / std::max(den, 1e-300)));
    }
    return worst;
}

// The QSA state of one attention layer as the engine holds it.
struct Layer {
    std::int64_t max_ctx;
    int max_t;
    DeviceBuffer pos, inv_freq, k_norm, q_norm, raw, blocks, kc, vc, q, gate, qi, cells, n_cells, sel_work, attn_work, out;
    Layer(std::int64_t ctx, int t) : max_ctx(ctx), max_t(t) {
        pos = DeviceBuffer(sizeof(std::int64_t));
        inv_freq = DeviceBuffer(kQsaRot / 2 * sizeof(double));
        k_norm = DeviceBuffer(kQsaDim * sizeof(float));
        q_norm = DeviceBuffer(kQsaDim * sizeof(float));
        raw = DeviceBuffer(std::size_t(ctx) * kQsaDim * sizeof(float));
        blocks = DeviceBuffer(std::size_t(ctx / kQsaRatio) * kQsaDim * sizeof(float));
        kc = DeviceBuffer(std::size_t(ctx) * kKvHeads * kHeadDim * sizeof(half));
        vc = DeviceBuffer(std::size_t(ctx) * kKvHeads * kHeadDim * sizeof(half));
        q = DeviceBuffer(std::size_t(t) * kHeads * kHeadDim * sizeof(float));
        gate = DeviceBuffer(std::size_t(t) * kHeads * kHeadDim * sizeof(float));
        out = DeviceBuffer(std::size_t(t) * kHeads * kHeadDim * sizeof(float));
        qi = DeviceBuffer(std::size_t(t) * kQsaHeads * kQsaDim * sizeof(float));
        cells = DeviceBuffer(std::size_t(t) * kW * sizeof(std::int32_t));
        n_cells = DeviceBuffer(std::size_t(t) * sizeof(std::int32_t));
        sel_work = DeviceBuffer(qsa_select_work_bytes(t, ctx));
        attn_work = DeviceBuffer(attn_sparse_work_floats(t) * sizeof(float));
        const auto f = inv_freq_table();
        upload(inv_freq, f.data(), f.size());
    }
    std::size_t bytes() const {
        return raw.bytes() + blocks.bytes() + kc.bytes() + vc.bytes() + q.bytes() + gate.bytes() + out.bytes() + qi.bytes() + cells.bytes() +
               sel_work.bytes() + attn_work.bytes();
    }
    // one engine step after the raw key rows and the K/V rows of pos0..pos0+T-1 are in place
    void blocks_step(int T) { qsa_update_blocks(raw.as<float>(), k_norm.as<float>(), inv_freq.as<double>(), blocks.as<float>(), pos.as<std::int64_t>(), T, kEps, 0); }
    void query_step(int T) { qsa_query(qi.as<float>(), q_norm.as<float>(), inv_freq.as<double>(), pos.as<std::int64_t>(), T, kEps, 0); }
    void select_step(int T) {
        qsa_select(qi.as<float>(), blocks.as<float>(), pos.as<std::int64_t>(), T, max_ctx, sel_work.get(), cells.as<std::int32_t>(),
                   n_cells.as<std::int32_t>(), 0);
    }
    void attn_step(int T) {
        attn_sparse(q.as<float>(), gate.as<float>(), kc.as<half>(), vc.as<half>(), cells.as<std::int32_t>(), n_cells.as<std::int32_t>(), T,
                    kScale, attn_work.as<float>(), out.as<float>(), 0);
    }
    std::vector<std::vector<std::int32_t>> get_cells(int T) const {
        std::vector<std::int32_t> c(std::size_t(T) * kW), n(static_cast<std::size_t>(T));
        download(c.data(), cells, c.size());
        download(n.data(), n_cells, n.size());
        std::vector<std::vector<std::int32_t>> out(static_cast<std::size_t>(T));
        for (int t = 0; t < T; ++t) out[std::size_t(t)].assign(c.begin() + std::ptrdiff_t(t) * kW, c.begin() + std::ptrdiff_t(t) * kW + n[std::size_t(t)]);
        return out;
    }
};

bool same_bits(const float * a, const float * b, std::size_t n, std::size_t * first = nullptr) {
    for (std::size_t i = 0; i < n; ++i) {
        if (std::memcmp(&a[i], &b[i], 4) != 0) {
            if (first) *first = i;
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------------------------------------------
// 1. random data

void test_random() {
    std::printf("== random data\n");
    const std::int64_t ctx = 16384;
    Layer L(ctx, 1024);
    Rng rng(42);
    std::vector<float> raw(std::size_t(ctx) * kQsaDim), kn(kQsaDim), qn(kQsaDim);
    rng.fill_normal(raw);
    for (auto & x : kn) x = rng.uniform(0.5f, 1.5f);
    for (auto & x : qn) x = rng.uniform(0.5f, 1.5f);
    upload(L.raw, raw.data(), raw.size());
    upload(L.k_norm, kn.data(), kn.size());
    upload(L.q_norm, qn.data(), qn.size());
    const auto inv = inv_freq_table();
    check(cudaMemset(L.blocks.get(), 0xff, L.blocks.bytes()), "memset");  // NaN sentinel: no block may be written twice wrongly or early

    // blocks, updated by a mix of prefill chunks and decode steps
    const int steps[] = {1024, 3, 1, 4, 2, 1000, 512, 7, 1, 1, 1, 1, 4, 4, 1024, 1024, 333, 2, 1};
    std::int64_t pos0 = 0;
    for (int T : steps) {
        set_pos(L.pos, pos0);
        L.blocks_step(T);
        pos0 += T;
    }
    check(cudaDeviceSynchronize(), "blocks");
    const std::int64_t nb = pos0 / kQsaRatio;
    std::vector<float> gb(std::size_t(ctx / kQsaRatio) * kQsaDim), pb(std::size_t(nb) * kQsaDim);
    download(gb.data(), L.blocks, gb.size());
    double worst_rel = 0.0;
    for (std::int64_t b = 0; b < nb; ++b) {
        port_block(raw.data(), b, kn.data(), inv, pb.data() + b * kQsaDim);
        for (int d = 0; d < kQsaDim; ++d) {
            const double ref = pb[std::size_t(b * kQsaDim + d)], got = gb[std::size_t(b * kQsaDim + d)];
            worst_rel = std::max(worst_rel, std::fabs(got - ref) / std::max(1e-6, std::fabs(ref)));
        }
    }
    std::size_t first = 0;
    const bool blocks_ok = same_bits(gb.data(), pb.data(), pb.size(), &first);
    bool untouched = true;
    for (std::size_t i = pb.size(); i < gb.size(); ++i) untouched = untouched && std::isnan(gb[i]);
    std::printf("  blocks: %lld complete blocks after %lld tokens, bit-identical to the port: %s (worst rel diff %.2e); later rows untouched: %s\n",
                (long long) nb, (long long) pos0, blocks_ok ? "yes" : "NO", worst_rel, untouched ? "yes" : "NO");
    expect(blocks_ok && untouched, "block keys");
    // the rest of the context, then every block
    const std::int64_t mid = pos0;
    for (; pos0 < ctx; pos0 += std::min<std::int64_t>(1024, ctx - pos0)) {
        set_pos(L.pos, pos0);
        L.blocks_step(int(std::min<std::int64_t>(1024, ctx - pos0)));
    }
    std::vector<float> blocks_h(gb.size());
    download(gb.data(), L.blocks, gb.size());
    for (std::int64_t b = 0; b < ctx / kQsaRatio; ++b) port_block(raw.data(), b, kn.data(), inv, blocks_h.data() + b * kQsaDim);
    const bool all_ok = same_bits(gb.data(), blocks_h.data(), gb.size());
    std::printf("  blocks: all %lld blocks of the context bit-identical: %s\n", (long long) (ctx / kQsaRatio), all_ok ? "yes" : "NO");
    expect(all_ok, "all block keys");
    pos0 = mid;

    // queries and selections at several (pos0, T)
    struct Case { std::int64_t pos0; int T; };
    const Case cases[] = {{0, 4}, {2040, 16}, {2047, 4}, {2048, 1}, {2050, 3}, {5001, 4}, {7000, 1}, {pos0 - 1024, 1024}};
    for (const Case & c : cases) {
        std::vector<float> qp(std::size_t(c.T) * kQsaHeads * kQsaDim);
        rng.fill_normal(qp);
        upload(L.qi, qp.data(), qp.size());
        set_pos(L.pos, c.pos0);
        L.query_step(c.T);
        L.select_step(c.T);
        check(cudaDeviceSynchronize(), "select");
        std::vector<float> gq(qp.size());
        download(gq.data(), L.qi, gq.size());
        for (int t = 0; t < c.T; ++t) port_query(qp.data() + std::size_t(t) * kQsaHeads * kQsaDim, c.pos0 + t, qn.data(), inv);
        const bool q_ok = same_bits(gq.data(), qp.data(), qp.size());
        const auto got = L.get_cells(c.T);
        int same = 0;
        for (int t = 0; t < c.T; ++t) same += got[std::size_t(t)] == port_select(qp.data() + std::size_t(t) * kQsaHeads * kQsaDim, blocks_h.data(), c.pos0 + t);
        std::printf("  pos0 %5lld T %4d: queries bit-identical %s, selections identical %d/%d\n", (long long) c.pos0, c.T, q_ok ? "yes" : "NO", same,
                    c.T);
        expect(q_ok && same == c.T, "queries / selection");
    }

    // exact ties: (a) every block key equal and every query head opposite: all scores 0, the lowest
    // blocks win; (b) 37 distinct keys repeated: large groups of equal scores straddle the cut
    {
        std::vector<float> keys(std::size_t(ctx / kQsaRatio) * kQsaDim), v(kQsaDim);
        rng.fill_normal(v);
        for (std::size_t b = 0; b < keys.size() / kQsaDim; ++b) std::memcpy(keys.data() + b * kQsaDim, v.data(), kQsaDim * sizeof(float));
        upload(L.blocks, keys.data(), keys.size());
        const int T = 4;
        const std::int64_t p0 = 9000;
        std::vector<float> qv(std::size_t(T) * kQsaHeads * kQsaDim);
        for (std::size_t i = 0; i < qv.size(); ++i) qv[i] = -v[i % kQsaDim];
        upload(L.qi, qv.data(), qv.size());  // used as already-normed queries: select only
        set_pos(L.pos, p0);
        L.select_step(T);
        auto got = L.get_cells(T);
        int same = 0;
        for (int t = 0; t < T; ++t) same += got[std::size_t(t)] == port_select(qv.data() + std::size_t(t) * kQsaHeads * kQsaDim, keys.data(), p0 + t);
        std::printf("  all-zero scores: selections identical %d/%d (first cells %d %d .. last %d)\n", same, T, got[0][0], got[0][1], got[0].back());
        expect(same == T, "zero-score ties");

        std::vector<float> distinct(37 * kQsaDim);
        rng.fill_normal(distinct);
        std::vector<int> pick(keys.size() / kQsaDim);
        for (auto & p : pick) p = int(rng.g() % 37);
        for (std::size_t b = 0; b < pick.size(); ++b) std::memcpy(keys.data() + b * kQsaDim, distinct.data() + std::size_t(pick[b]) * kQsaDim, kQsaDim * sizeof(float));
        upload(L.blocks, keys.data(), keys.size());
        const int T2 = 64;
        const std::int64_t p2 = 12000;
        std::vector<float> qr(std::size_t(T2) * kQsaHeads * kQsaDim);
        rng.fill_normal(qr);
        upload(L.qi, qr.data(), qr.size());
        set_pos(L.pos, p2);
        L.select_step(T2);
        got = L.get_cells(T2);
        same = 0;
        for (int t = 0; t < T2; ++t) same += got[std::size_t(t)] == port_select(qr.data() + std::size_t(t) * kQsaHeads * kQsaDim, keys.data(), p2 + t);
        std::printf("  37 repeated keys (ties at the cut): selections identical %d/%d\n", same, T2);
        expect(same == T2, "repeated-key ties");
        upload(L.blocks, blocks_h.data(), blocks_h.size());
    }

    // attention: dense and sparse cell lists, T = 1, 4, 64, 1024
    std::vector<float> kf(std::size_t(ctx) * kKvHeads * kHeadDim), vf(kf.size());
    rng.fill_normal(kf);
    rng.fill_normal(vf);
    kf = half_round(kf);
    vf = half_round(vf);
    upload(L.kc, to_half(kf.data(), kf.size()).data(), kf.size());
    upload(L.vc, to_half(vf.data(), vf.size()).data(), vf.size());
    const Case acases[] = {{100, 1}, {2049, 4}, {9000, 1}, {9000, 4}, {12000, 64}, {pos0 - 1024, 1024}};
    for (const Case & c : acases) {
        const std::size_t nq = std::size_t(c.T) * kHeads * kHeadDim;
        std::vector<float> q(nq), g(nq), qp(std::size_t(c.T) * kQsaHeads * kQsaDim);
        rng.fill_normal(q, 0.3f);
        rng.fill_normal(g);
        rng.fill_normal(qp);
        upload(L.q, q.data(), nq);
        upload(L.gate, g.data(), nq);
        upload(L.qi, qp.data(), qp.size());
        set_pos(L.pos, c.pos0);
        L.query_step(c.T);
        L.select_step(c.T);
        L.attn_step(c.T);
        check(cudaDeviceSynchronize(), "attn");
        const auto cells = L.get_cells(c.T);
        std::vector<float> out(nq);
        download(out.data(), L.out, nq);
        std::vector<double> want(nq);
        for (int t = 0; t < c.T; ++t) {
            port_attention(q.data() + std::size_t(t) * kHeads * kHeadDim, g.data() + std::size_t(t) * kHeads * kHeadDim, kf, vf,
                           cells[std::size_t(t)].data(), int(cells[std::size_t(t)].size()), want.data() + std::size_t(t) * kHeads * kHeadDim);
        }
        const double err = attn_error(out.data(), want.data(), c.T);
        std::printf("  attention pos0 %5lld T %4d (%zu cells for the first token): worst row relative error %.2e vs the double port\n",
                    (long long) c.pos0, c.T, cells[0].size(), err);
        expect(err < 1e-5, "attention");
    }
}

// ------------------------------------------------------------------------------------------------
// 2. real data: the reference's dumps

struct Dump {
    std::int32_t ne[4] = {0, 0, 0, 0};
    std::vector<float> v;
};
Dump load_dump(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("missing dump " + path);
    Dump d;
    f.read(reinterpret_cast<char *>(d.ne), sizeof(d.ne));
    d.v.resize(std::size_t(d.ne[0]) * d.ne[1] * d.ne[2] * d.ne[3]);
    f.read(reinterpret_cast<char *>(d.v.data()), std::streamsize(d.v.size() * sizeof(float)));
    return d;
}

std::vector<std::string> shard_paths(const std::string & first) {
    std::smatch mm;
    const std::string name = fs::path(first).filename().string();
    static const std::regex pat(R"((.*)-(\d{5})-of-(\d{5})\.gguf)");
    if (!std::regex_match(name, mm, pat)) return {first};
    const int count = std::stoi(mm[3].str());
    std::vector<std::string> out;
    for (int i = 1; i <= count; ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "-%05d-of-%05d.gguf", i, count);
        out.push_back((fs::path(first).parent_path() / (mm[1].str() + buf)).string());
    }
    return out;
}

std::vector<float> f32_tensor(const GgufModel & m, const std::string & name) {
    const GgufTensor & t = m.tensor(name);
    std::vector<float> v(std::size_t(t.elements()));
    dequantize_row(t.type, t.data, v.data(), t.elements());
    return v;
}

void test_real(const std::string & model_path, const std::string & dir, const std::vector<int> & layers) {
    std::printf("== real data: reference dumps in %s\n", dir.c_str());
    GgufModel gguf(shard_paths(model_path));
    for (int il : layers) {
        const std::string sfx = "-" + std::to_string(il) + ".bin";
        const Dump raw = load_dump(dir + "/indexer_k_raw" + sfx), qproj = load_dump(dir + "/indexer_q_proj" + sfx),
                   qref = load_dump(dir + "/indexer_q" + sfx), bref = load_dump(dir + "/indexer_blocks" + sfx),
                   sref = load_dump(dir + "/indexer_sel" + sfx), Q = load_dump(dir + "/Qcur" + sfx), Qf = load_dump(dir + "/Qcur_full" + sfx),
                   K = load_dump(dir + "/Kcur" + sfx), V = load_dump(dir + "/Vcur" + sfx), A = load_dump(dir + "/attn_gated" + sfx);
        const int N = raw.ne[1];
        const std::vector<float> kn = f32_tensor(gguf, "blk." + std::to_string(il) + ".indexer.k_norm.weight");
        const std::vector<float> qn = f32_tensor(gguf, "blk." + std::to_string(il) + ".indexer.q_norm.weight");
        Layer L(4096, 1024);
        upload(L.k_norm, kn.data(), kn.size());
        upload(L.q_norm, qn.data(), qn.size());
        const std::vector<float> kh = half_round(K.v), vh = half_round(V.v);
        // prefill chunks, then decode steps of 1, 3 and 4 tokens
        std::vector<int> steps;
        for (int left = N - 8; left > 0; left -= std::min(left, 1024)) steps.push_back(std::min(left, 1024));
        steps.insert(steps.end(), {1, 3, 4});
        std::int64_t pos0 = 0;
        int q_bad = 0, sel_same = 0, sel_sparse = 0, sel_sparse_same = 0;
        double err_ref = 0.0, err_port = 0.0;
        for (int T : steps) {
            const std::size_t rq = std::size_t(kHeads) * kHeadDim;
            upload(L.raw, raw.v.data() + pos0 * kQsaDim, std::size_t(T) * kQsaDim, std::size_t(pos0) * kQsaDim);
            upload(L.kc, to_half(K.v.data() + pos0 * kKvHeads * kHeadDim, std::size_t(T) * kKvHeads * kHeadDim).data(),
                   std::size_t(T) * kKvHeads * kHeadDim, std::size_t(pos0) * kKvHeads * kHeadDim);
            upload(L.vc, to_half(V.v.data() + pos0 * kKvHeads * kHeadDim, std::size_t(T) * kKvHeads * kHeadDim).data(),
                   std::size_t(T) * kKvHeads * kHeadDim, std::size_t(pos0) * kKvHeads * kHeadDim);
            upload(L.qi, qproj.v.data() + pos0 * kQsaHeads * kQsaDim, std::size_t(T) * kQsaHeads * kQsaDim);
            upload(L.q, Q.v.data() + pos0 * rq, std::size_t(T) * rq);
            std::vector<float> g(std::size_t(T) * rq);
            for (int t = 0; t < T; ++t) {
                for (int h = 0; h < kHeads; ++h) {
                    std::memcpy(g.data() + (std::size_t(t) * kHeads + h) * kHeadDim,
                                Qf.v.data() + (std::size_t(pos0 + t) * kHeads + h) * 2 * kHeadDim + kHeadDim, kHeadDim * sizeof(float));
                }
            }
            upload(L.gate, g.data(), g.size());
            set_pos(L.pos, pos0);
            L.blocks_step(T);
            L.query_step(T);
            L.select_step(T);
            L.attn_step(T);
            check(cudaDeviceSynchronize(), "real step");
            std::vector<float> gq(std::size_t(T) * kQsaHeads * kQsaDim), out(std::size_t(T) * rq);
            download(gq.data(), L.qi, gq.size());
            download(out.data(), L.out, out.size());
            for (int t = 0; t < T; ++t) q_bad += !same_bits(gq.data() + std::size_t(t) * kQsaHeads * kQsaDim, qref.v.data() + (pos0 + t) * kQsaHeads * kQsaDim, kQsaHeads * kQsaDim);
            const auto cells = L.get_cells(T);
            std::vector<double> want(out.size());
            for (int t = 0; t < T; ++t) {
                const std::int64_t pos = pos0 + t;
                const float * row = sref.v.data() + std::size_t(pos) * kW;
                std::vector<std::int32_t> r;
                if (row[0] < 0.0f) {
                    for (std::int64_t p = 0; p <= pos; ++p) r.push_back(std::int32_t(p));
                } else {
                    for (int j = 0; j < kW && row[j] >= 0.0f; ++j) r.push_back(std::int32_t(row[j]));
                }
                const bool same = cells[std::size_t(t)] == r;
                sel_same += same;
                if (pos + 1 > kW) {
                    ++sel_sparse;
                    sel_sparse_same += same;
                }
                port_attention(Q.v.data() + std::size_t(pos) * rq, g.data() + std::size_t(t) * rq, kh, vh, cells[std::size_t(t)].data(),
                               int(cells[std::size_t(t)].size()), want.data() + std::size_t(t) * rq);
            }
            err_port = std::max(err_port, attn_error(out.data(), want.data(), T));
            std::vector<double> ref(out.size());
            for (std::size_t i = 0; i < ref.size(); ++i) ref[i] = A.v[std::size_t(pos0) * rq + i];
            err_ref = std::max(err_ref, attn_error(out.data(), ref.data(), T));
            pos0 += T;
        }
        const int nb = int(pos0 / kQsaRatio);
        std::vector<float> gb(std::size_t(nb) * kQsaDim);
        download(gb.data(), L.blocks, gb.size());
        std::size_t first = 0;
        const bool blocks_ok = bref.ne[1] == nb && same_bits(gb.data(), bref.v.data(), gb.size(), &first);
        std::printf("  layer %2d: %d tokens in steps", il, N);
        for (int T : steps) std::printf(" %d", T);
        std::printf("\n    blocks bit-identical to the reference: %s (%d blocks)%s\n", blocks_ok ? "yes" : "NO", nb,
                    blocks_ok ? "" : (" first difference at float " + std::to_string(first)).c_str());
        std::printf("    queries bit-identical: %d/%d tokens\n", N - q_bad, N);
        std::printf("    selections identical to the reference: %d/%d (sparse queries %d/%d)\n", sel_same, N, sel_sparse_same, sel_sparse);
        std::printf("    attention: worst row relative error %.2e vs the double port (fp16 K/V), %.2e vs the reference (fp32 K/V)\n", err_port, err_ref);
        expect(blocks_ok && q_bad == 0 && sel_same == N && err_port < 1e-5 && err_ref < 1e-2, "real-data layer");
    }
}

// ------------------------------------------------------------------------------------------------
// 3. timings

__global__ void k_spin(float * x, int n) {  // keeps the GPU busy so that it runs at its boost clock
    float v = x[threadIdx.x];
    for (int i = 0; i < n; ++i) v = v * 0.999f + 0.001f;
    x[threadIdx.x] = v;
}
void warm_up() {
    DeviceBuffer x(1024 * sizeof(float));
    check(cudaMemset(x.get(), 0, x.bytes()), "memset");
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(300)) {
        k_spin<<<512, 1024>>>(x.as<float>(), 20000);
        check(cudaDeviceSynchronize(), "spin");
    }
}

// best of 5 trials of `reps` calls (the GPU is shared, so the fastest trial is the uncontended one)
float time_ms(const std::function<void()> & fn, int reps) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    fn();
    float best = INFINITY;
    for (int trial = 0; trial < 5; ++trial) {
        cudaEventRecord(a);
        for (int i = 0; i < reps; ++i) fn();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, a, b);
        best = std::min(best, ms / float(reps));
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return best;
}

void fill_device_normal(const DeviceBuffer & d, std::size_t n, Rng & rng, bool as_half) {
    const std::size_t chunk = 1 << 22;
    std::vector<float> f(chunk);
    for (std::size_t i = 0; i < n; i += chunk) {
        const std::size_t m = std::min(chunk, n - i);
        for (std::size_t j = 0; j < m; ++j) f[j] = rng.normal();
        if (as_half) upload(d, to_half(f.data(), m).data(), m, i);
        else upload(d, f.data(), m, i);
    }
}

void test_timing() {
    std::printf("== timings (synthetic data, ms per call)\n");
    std::printf("  %7s %5s %9s %9s %9s %9s %9s\n", "context", "T", "blocks", "query", "select", "attention", "total");
    Rng rng(7);
    for (std::int64_t ctx : {std::int64_t(2048), std::int64_t(32768), std::int64_t(262144)}) {
        Layer L(ctx, 1024);
        std::vector<float> kn(kQsaDim, 1.0f);
        upload(L.k_norm, kn.data(), kn.size());
        upload(L.q_norm, kn.data(), kn.size());
        fill_device_normal(L.raw, std::size_t(ctx) * kQsaDim, rng, false);
        fill_device_normal(L.kc, std::size_t(ctx) * kKvHeads * kHeadDim, rng, true);
        fill_device_normal(L.vc, std::size_t(ctx) * kKvHeads * kHeadDim, rng, true);
        fill_device_normal(L.q, std::size_t(1024) * kHeads * kHeadDim, rng, false);
        fill_device_normal(L.gate, std::size_t(1024) * kHeads * kHeadDim, rng, false);
        for (std::int64_t p = 0; p < ctx; p += 1024) {  // every block of the context
            set_pos(L.pos, p);
            L.blocks_step(int(std::min<std::int64_t>(1024, ctx - p)));
        }
        std::vector<float> qp(std::size_t(1024) * kQsaHeads * kQsaDim);
        rng.fill_normal(qp);
        for (int T : {1, 4, 1024}) {
            const std::int64_t p0 = ctx - T;
            set_pos(L.pos, p0);
            upload(L.qi, qp.data(), std::size_t(T) * kQsaHeads * kQsaDim);
            L.query_step(T);  // the select and attention timings use normed queries
            const int reps = T == 1024 ? 3 : 50;
            warm_up();
            const float tb = time_ms([&] { L.blocks_step(T); }, reps);
            const float tq = time_ms([&] { L.query_step(T); }, reps);
            upload(L.qi, qp.data(), std::size_t(T) * kQsaHeads * kQsaDim);
            L.query_step(T);
            const float ts = time_ms([&] { L.select_step(T); }, reps);
            const float ts1 = time_ms([&] { qsa_scores(L.qi.as<float>(), L.blocks.as<float>(), L.pos.as<std::int64_t>(), T, L.max_ctx, L.sel_work.get(), 0); }, reps);
            const float ts2 = time_ms([&] { qsa_pick(L.sel_work.get(), L.pos.as<std::int64_t>(), T, L.max_ctx, L.cells.as<std::int32_t>(), L.n_cells.as<std::int32_t>(), 0); }, reps);
            const float ta = time_ms([&] { L.attn_step(T); }, reps);
            std::printf("  %7lld %5d %9.3f %9.3f %9.3f %9.3f %9.3f   (select = scores %.3f + pick %.3f)\n", (long long) ctx, T, tb, tq, ts, ta,
                        tb + tq + ts + ta, ts1, ts2);
            // spot-check the selection against the port at this size
            if (ctx > 2048) {
                std::vector<float> qh(std::size_t(T) * kQsaHeads * kQsaDim), bh(std::size_t(ctx / kQsaRatio) * kQsaDim);
                download(qh.data(), L.qi, qh.size());
                download(bh.data(), L.blocks, bh.size());
                const auto got = L.get_cells(T);
                int checked = 0, same = 0;
                std::vector<int> ts = {0, T / 2, T - 1};
                ts.erase(std::unique(ts.begin(), ts.end()), ts.end());
                for (int t : ts) {
                    ++checked;
                    same += got[std::size_t(t)] == port_select(qh.data() + std::size_t(t) * kQsaHeads * kQsaDim, bh.data(), p0 + t);
                }
                if (same != checked) std::printf("    selection spot check: %d/%d identical\n", same, checked);
                expect(same == checked, "selection at scale");
            }
        }
        std::printf("    (device memory for this context: %.0f MB)\n", double(L.bytes()) / 1048576.0);
    }
}

// ------------------------------------------------------------------------------------------------
// 4. KV streaming parity

void test_kv_stream() {
    std::printf("== KV streaming parity (synthetic data)\n");
    const std::int64_t ctx = 49152, resident = KvStreamCache::kMinResident;  // the smallest page cache: most evictions
    const int max_t = 2048, n_layers = 2;
    cudaStream_t s = nullptr;
    check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    Rng rng(1234);
    const std::size_t qf = std::size_t(kHeads) * 2 * kHeadDim, kvw = std::size_t(kKvHeads) * kHeadDim, qo = std::size_t(kHeads) * kHeadDim;
    DeviceBuffer pos(sizeof(std::int64_t)), inv(kRot / 2 * sizeof(double)), qn(kHeadDim * 4), kn(kHeadDim * 4);
    DeviceBuffer q_full(max_t * qf * 4), k_in(max_t * kvw * 4), v_in(max_t * kvw * 4), qi(std::size_t(max_t) * kQsaHeads * kQsaDim * 4);
    DeviceBuffer blocks(std::size_t(ctx / kQsaRatio + 1) * kQsaDim * 4), sel_work(qsa_select_work_bytes(max_t, ctx));
    DeviceBuffer cells_a(std::size_t(max_t) * kW * 4), cells_b(std::size_t(max_t) * kW * 4), n_cells(max_t * 4);
    DeviceBuffer q_a(max_t * qo * 4), g_a(max_t * qo * 4), q_b(max_t * qo * 4), g_b(max_t * qo * 4), o_a(max_t * qo * 4), o_b(max_t * qo * 4);
    DeviceBuffer work(attn_sparse_work_floats(max_t) * 4);
    std::vector<DeviceBuffer> kc, vc;
    for (int l = 0; l < n_layers; ++l) {
        kc.emplace_back(std::size_t(ctx) * kvw * 2);
        vc.emplace_back(std::size_t(ctx) * kvw * 2);
    }
    {
        std::vector<double> f(kRot / 2);
        for (int i = 0; i < kRot / 2; ++i) f[std::size_t(i)] = std::pow(kRopeBase, -2.0 * i / kRot);
        upload(inv, f.data(), f.size());
        std::vector<float> w(kHeadDim);
        for (auto & x : w) x = rng.uniform(0.5f, 1.5f);
        upload(qn, w.data(), w.size());
        for (auto & x : w) x = rng.uniform(0.5f, 1.5f);
        upload(kn, w.data(), w.size());
        std::vector<float> b(std::size_t(ctx / kQsaRatio + 1) * kQsaDim);
        rng.fill_normal(b);
        upload(blocks, b.data(), b.size());
    }
    KvStreamCache kvc(ctx, resident, 40, s);
    for (int l = 0; l < n_layers; ++l) kvc.add_layer();
    DeviceBuffer stage(kvc.stage_bytes(ctx));

    std::vector<float> h(std::size_t(max_t) * qf);
    int steps = 0, bad = 0, compared = 0;
    // one step at p0: both layers through both paths
    auto step = [&](std::int64_t p0, int T) {
        check(cudaMemcpyAsync(pos.get(), &p0, sizeof(p0), cudaMemcpyHostToDevice, s), "pos");
        check(cudaStreamSynchronize(s), "pos");
        kvc.begin_step(p0, T, T > kMaxTokens ? stage.get() : nullptr);
        for (int l = 0; l < n_layers; ++l) {
            for (DeviceBuffer * b : {&q_full, &k_in, &v_in}) {
                const std::size_t n = std::size_t(T) * (b == &q_full ? qf : kvw);
                for (std::size_t i = 0; i < n; ++i) h[i] = rng.normal();
                check(cudaMemcpyAsync(b->get(), h.data(), n * 4, cudaMemcpyHostToDevice, s), "upload");
                check(cudaStreamSynchronize(s), "upload");
            }
            for (std::size_t i = 0; i < std::size_t(T) * kQsaHeads * kQsaDim; ++i) h[i] = rng.normal();
            check(cudaMemcpyAsync(qi.get(), h.data(), std::size_t(T) * kQsaHeads * kQsaDim * 4, cudaMemcpyHostToDevice, s), "upload");
            // resident
            attn_prep(q_full.as<float>(), k_in.as<float>(), v_in.as<float>(), qn.as<float>(), kn.as<float>(), inv.as<double>(), q_a.as<float>(),
                      g_a.as<float>(), kc[std::size_t(l)].as<half>(), vc[std::size_t(l)].as<half>(), pos.as<std::int64_t>(), T, kEps, s);
            qsa_select(qi.as<float>(), blocks.as<float>(), pos.as<std::int64_t>(), T, ctx, sel_work.get(), cells_a.as<std::int32_t>(),
                       n_cells.as<std::int32_t>(), s);
            check(cudaMemcpyAsync(cells_b.get(), cells_a.get(), std::size_t(T) * kW * 4, cudaMemcpyDeviceToDevice, s), "cells");
            attn_sparse(q_a.as<float>(), g_a.as<float>(), kc[std::size_t(l)].as<half>(), vc[std::size_t(l)].as<half>(), cells_a.as<std::int32_t>(),
                        n_cells.as<std::int32_t>(), T, kScale, work.as<float>(), o_a.as<float>(), s);
            // streamed
            attn_prep(q_full.as<float>(), k_in.as<float>(), v_in.as<float>(), qn.as<float>(), kn.as<float>(), inv.as<double>(), q_b.as<float>(),
                      g_b.as<float>(), kvc.store(l), pos.as<std::int64_t>(), T, kEps, s);
            kvc.attend(l, q_b.as<float>(), g_b.as<float>(), cells_b.as<std::int32_t>(), n_cells.as<std::int32_t>(), pos.as<std::int64_t>(),
                       kScale, work.as<float>(), o_b.as<float>());
            std::vector<float> a(std::size_t(T) * qo), b(std::size_t(T) * qo);
            check(cudaMemcpyAsync(a.data(), o_a.get(), a.size() * 4, cudaMemcpyDeviceToHost, s), "download");
            check(cudaMemcpyAsync(b.data(), o_b.get(), b.size() * 4, cudaMemcpyDeviceToHost, s), "download");
            check(cudaStreamSynchronize(s), "step");
            ++compared;
            if (!same_bits(a.data(), b.data(), a.size())) {
                if (bad < 5) std::printf("  layer %d step at %lld (T %d): attention differs\n", l, (long long) p0, T);
                ++bad;
            }
        }
        ++steps;
    };
    std::int64_t p = 0;
    for (int T : {1024, 1000, 2048, 2048, 1500}) { step(p, T); p += T; }        // chunks within the page cache (prefix)
    for (int T : {2048, 700, 2048, 2048, 333}) { step(p, T); p += T; }          // chunks beyond it (staged)
    for (int i = 0; i < 120; ++i) {                                             // decode, drafts rolled back
        const int T = 1 + int(rng.uniform(0.0f, 3.999f));
        step(p, T);
        p += 1 + int(rng.uniform(0.0f, float(T) - 0.001f));
    }
    p = 20000;                                                                  // back to an earlier position, deep branch
    step(p, 600); p += 600;
    for (int i = 0; i < 40; ++i) { step(p, 4); p += 2; }
    p = 5000;                                                                   // back within the page cache
    step(p, 500); p += 500;
    for (int i = 0; i < 40; ++i) { const int T = 1 + (i % 4); step(p, T); p += T; }
    step(p, 37); p += 37;                                                       // a short prompt chunk (prefix)
    p = 30000;
    step(p, 9); p += 9;                                                         // short chunks deep: in groups of 4 queries
    step(p, 40); p += 40;
    step(p, 58); p += 58;                                                       // up to max(40, p0 / 512 = 58): groups
    step(p, 59); p += 59;                                                       // one token more: staged
    for (int i = 0; i < 20; ++i) { step(p, 1); p += 1; }
    // the host copy must equal the resident cache for every position written
    const KvStreamStats st = kvc.stats();
    std::printf("  %d steps, %d attention outputs bit-identical to the resident cache: %d; %llu page lookups, %llu misses, %llu staged chunks, "
                "%llu grouped chunks%s\n",
                steps, compared, compared - bad, (unsigned long long) st.lookups, (unsigned long long) st.misses,
                (unsigned long long) st.staged_chunks, (unsigned long long) st.grouped_chunks, st.overflow ? ", OVERFLOW" : "");
    expect(bad == 0, "streamed attention bit-identical to resident");
    expect(!st.overflow, "no page cache overflow");
    expect(st.misses > 0 && st.staged_chunks > 0 && st.grouped_chunks > 0, "the test exercised misses, staging and groups");

    // cost of a decode step's attention, resident vs streamed with every page already resident (the
    // selection is copied back before each streamed call, since attend() rewrites it)
    for (int T : {1, 3}) {
        const std::int64_t p0 = p;
        check(cudaMemcpyAsync(pos.get(), &p0, sizeof(p0), cudaMemcpyHostToDevice, s), "pos");
        qsa_select(qi.as<float>(), blocks.as<float>(), pos.as<std::int64_t>(), T, ctx, sel_work.get(), cells_a.as<std::int32_t>(),
                   n_cells.as<std::int32_t>(), s);
        kvc.begin_step(p0, T, nullptr);
        auto timed = [&](const std::function<void()> & fn) {
            cudaEvent_t a, b;
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            for (int i = 0; i < 20; ++i) fn();
            float best = INFINITY;
            for (int trial = 0; trial < 5; ++trial) {
                cudaEventRecord(a, s);
                for (int i = 0; i < 200; ++i) fn();
                cudaEventRecord(b, s);
                cudaEventSynchronize(b);
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                best = std::min(best, ms / 200);
            }
            cudaEventDestroy(a);
            cudaEventDestroy(b);
            return best * 1000.0f;
        };
        const std::size_t cb = std::size_t(T) * kW * 4;
        const float t_res = timed([&] {
            attn_sparse(q_a.as<float>(), g_a.as<float>(), kc[0].as<half>(), vc[0].as<half>(), cells_a.as<std::int32_t>(), n_cells.as<std::int32_t>(),
                        T, kScale, work.as<float>(), o_a.as<float>(), s);
        });
        const float t_copy = timed([&] { cudaMemcpyAsync(cells_b.get(), cells_a.get(), cb, cudaMemcpyDeviceToDevice, s); });
        const float t_str = timed([&] {
            cudaMemcpyAsync(cells_b.get(), cells_a.get(), cb, cudaMemcpyDeviceToDevice, s);
            kvc.attend(0, q_b.as<float>(), g_b.as<float>(), cells_b.as<std::int32_t>(), n_cells.as<std::int32_t>(), pos.as<std::int64_t>(), kScale,
                       work.as<float>(), o_b.as<float>());
        });
        std::printf("  decode attention at %lld, T %d: resident %.1f us, streamed (resolve + copy + attention) %.1f us\n", (long long) p0, T,
                    t_res, t_str - t_copy);
    }
    check(cudaStreamDestroy(s), "stream");
}

// The MTP layer's K/V ring: attention over the last 2051 positions read from a ring of R rows must equal
// the same attention over a full cache, for passes of up to `cap` tokens.
void test_mtp_ring() {
    std::printf("== MTP K/V ring parity (synthetic data)\n");
    const int cap = 512;
    const std::int64_t ctx = 12000, ring = (cap + kW + 255) / 256 * 256;
    Rng rng(99);
    const std::size_t qf = std::size_t(kHeads) * 2 * kHeadDim, kvw = std::size_t(kKvHeads) * kHeadDim, qo = std::size_t(kHeads) * kHeadDim;
    DeviceBuffer pos(8), inv(kRot / 2 * sizeof(double)), qn(kHeadDim * 4), kn(kHeadDim * 4);
    DeviceBuffer q_full(cap * qf * 4), k_in(cap * kvw * 4), v_in(cap * kvw * 4), cells(std::size_t(cap) * kW * 4), n_cells(cap * 4);
    DeviceBuffer q(cap * qo * 4), g(cap * qo * 4), o_a(cap * qo * 4), o_b(cap * qo * 4), work(attn_sparse_work_floats(cap) * 4);
    DeviceBuffer kc(std::size_t(ctx) * kvw * 2), vc(std::size_t(ctx) * kvw * 2), kr(std::size_t(ring) * kvw * 2), vr(std::size_t(ring) * kvw * 2);
    std::vector<double> f(kRot / 2);
    for (int i = 0; i < kRot / 2; ++i) f[std::size_t(i)] = std::pow(kRopeBase, -2.0 * i / kRot);
    upload(inv, f.data(), f.size());
    std::vector<float> w(kHeadDim, 1.0f);
    upload(qn, w.data(), w.size());
    upload(kn, w.data(), w.size());
    std::vector<float> h(std::size_t(cap) * qf);
    int bad = 0, n = 0;
    std::int64_t p = 0;
    while (p < ctx - cap) {
        const int T = std::min<int>(cap, 1 + int(rng.uniform(0.0f, 1.0f) * float(rng.uniform(0.0f, 1.0f) < 0.3f ? cap : 4)));
        set_pos(pos, p);
        for (DeviceBuffer * b : {&q_full, &k_in, &v_in}) {
            const std::size_t m = std::size_t(T) * (b == &q_full ? qf : kvw);
            for (std::size_t i = 0; i < m; ++i) h[i] = rng.normal();
            upload(*b, h.data(), m);
        }
        attn_prep(q_full.as<float>(), k_in.as<float>(), v_in.as<float>(), qn.as<float>(), kn.as<float>(), inv.as<double>(), q.as<float>(),
                  g.as<float>(), kc.as<half>(), vc.as<half>(), pos.as<std::int64_t>(), T, kEps, 0);
        window_cells(pos.as<std::int64_t>(), T, kW, cells.as<std::int32_t>(), n_cells.as<std::int32_t>(), 0);
        attn_sparse(q.as<float>(), g.as<float>(), kc.as<half>(), vc.as<half>(), cells.as<std::int32_t>(), n_cells.as<std::int32_t>(), T, kScale,
                    work.as<float>(), o_a.as<float>(), 0);
        KvStore st;
        st.k = kr.as<half>();
        st.v = vr.as<half>();
        st.ring = ring;
        attn_prep(q_full.as<float>(), k_in.as<float>(), v_in.as<float>(), qn.as<float>(), kn.as<float>(), inv.as<double>(), q.as<float>(),
                  g.as<float>(), st, pos.as<std::int64_t>(), T, kEps, 0);
        window_cells(pos.as<std::int64_t>(), T, kW, cells.as<std::int32_t>(), n_cells.as<std::int32_t>(), 0, ring);
        attn_sparse(q.as<float>(), g.as<float>(), kr.as<half>(), vr.as<half>(), cells.as<std::int32_t>(), n_cells.as<std::int32_t>(), T, kScale,
                    work.as<float>(), o_b.as<float>(), 0);
        std::vector<float> a(std::size_t(T) * qo), b(std::size_t(T) * qo);
        download(a.data(), o_a, a.size());
        download(b.data(), o_b, b.size());
        bad += !same_bits(a.data(), b.data(), a.size());
        ++n;
        p += T;
    }
    std::printf("  %d passes up to %lld positions with a %lld-row ring: bit-identical to the full cache %d/%d\n", n, (long long) p,
                (long long) ring, n - bad, n);
    expect(bad == 0, "MTP ring attention bit-identical");
}

}  // namespace

int main(int argc, char ** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::string model, dumps;
    std::vector<int> layers = {3, 23, 47};
    bool timing = true, random = true, kv_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "-m") model = next();
        else if (a == "--dumps") dumps = next();
        else if (a == "--layers") {
            layers.clear();
            std::string s = next(), cur;
            for (char ch : s + ",") {
                if (ch == ',') { if (!cur.empty()) layers.push_back(std::stoi(cur)); cur.clear(); }
                else cur += ch;
            }
        } else if (a == "--no-timing") timing = false;
        else if (a == "--timing-only") random = false;
        else if (a == "--kv-only") kv_only = true;
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    try {
        qsa_init();
        if (!kv_only) {
            if (random) test_random();
            if (!model.empty() && !dumps.empty()) test_real(model, dumps, layers);
            if (timing) test_timing();
        }
        if (random) {
            test_mtp_ring();
            test_kv_stream();
        }
    } catch (const std::exception & e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
    std::printf(g_failures ? "FAILED (%d)\n" : "ALL OK\n", g_failures);
    return g_failures ? 1 : 0;
}
