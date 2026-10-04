#include "flashnext/cpu_experts.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <immintrin.h>

#include "flashnext/quants.h"

namespace ninfer::flashnext {

namespace {

#pragma pack(push, 1)
// IQ4_XS block in the Q4L nibble layout with plain 8-bit block scales (ls - 32).
struct BlockQ4X {
    std::uint16_t d;
    std::int8_t scales[8];
    std::uint8_t qs[128];
};
// One 640-wide IQ4_NL row (20 blocks): the 20 fp16 scales, then ten 64-weight chunks of nibbles.
struct RowIQ4L {
    std::uint16_t d[20];
    std::uint8_t qs[320];
};
#pragma pack(pop)
static_assert(sizeof(BlockQ4X) == 138 && sizeof(RowIQ4L) == 360, "layouts");

enum class GateUp { Q4L, Q4X };
enum class Down { IQ4L, Q8_0 };

struct Buffer {
    std::uint8_t * p = nullptr;
    std::size_t n = 0;
    void alloc(std::size_t bytes) {
        n = bytes;
#if defined(_WIN32)
        p = static_cast<std::uint8_t *>(_aligned_malloc(bytes, 64));
#else
        p = static_cast<std::uint8_t *>(std::aligned_alloc(64, (bytes + 63) / 64 * 64));
#endif
        if (!p) throw std::bad_alloc();
    }
    ~Buffer() {
#if defined(_WIN32)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }
};

// Chunk layout used by Q4L, Q4X and IQ4L: in each 64-weight chunk, byte j holds code j (low nibble)
// and code j+32 (high nibble).
void pack_chunks(const std::uint8_t * codes, int n, std::uint8_t * qs) {
    for (int c = 0; c < n / 64; ++c)
        for (int j = 0; j < 32; ++j) qs[32 * c + j] = std::uint8_t(codes[64 * c + j] | (codes[64 * c + 32 + j] << 4));
}

void repack_iq4xs_block(const BlockIQ4_XS & s, BlockQ4X & d) {
    d.d = s.d;
    std::uint8_t codes[256];
    for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((s.scales_l[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((s.scales_h >> 2 * ib) & 3) << 4);
        d.scales[ib] = std::int8_t(ls - 32);
        for (int j = 0; j < 16; ++j) {
            codes[32 * ib + j] = s.qs[16 * ib + j] & 0xF;
            codes[32 * ib + 16 + j] = s.qs[16 * ib + j] >> 4;
        }
    }
    pack_chunks(codes, 256, d.qs);
}

void repack_iq4nl_row(const BlockIQ4_NL * s, RowIQ4L & d) {
    std::uint8_t codes[640];
    for (int b = 0; b < 20; ++b) {
        d.d[b] = s[b].d;
        for (int j = 0; j < 16; ++j) {
            codes[32 * b + j] = s[b].qs[j] & 0xF;
            codes[32 * b + 16 + j] = s[b].qs[j] >> 4;
        }
    }
    pack_chunks(codes, 640, d.qs);
}

inline float half_to_float(std::uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }

// Code -> Q4L weight + 16 (1..31), unsigned for maddubs.
inline __m512i lut_q4l() {
    alignas(16) static const std::uint8_t t[16] = {17, 19, 21, 23, 25, 27, 29, 31, 15, 13, 11, 9, 7, 5, 3, 1};
    return _mm512_broadcast_i32x4(_mm_load_si128(reinterpret_cast<const __m128i *>(t)));
}
// Code -> IQ4 value + 128 (1..241), unsigned for vpdpbusd.
inline __m512i lut_iq4() {
    alignas(16) static const std::uint8_t t[16] = {1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241};
    return _mm512_broadcast_i32x4(_mm_load_si128(reinterpret_cast<const __m128i *>(t)));
}

// 32 bytes of chunked nibbles -> 64 codes in weight order.
inline __m512i chunk_codes(const std::uint8_t * p) {
    const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
    return _mm512_and_si512(_mm512_inserti64x4(_mm512_castsi256_si512(raw), _mm256_srli_epi16(raw, 4), 1), _mm512_set1_epi8(0x0F));
}

}  // namespace

// Gate/up input: int8 per 256 with a float scale, and per-32 sums for the unsigned-offset fix.
struct alignas(64) XAct {
    std::int8_t q[2560];
    float d[10];
    std::int32_t sum32[80];
};
// Down input of one (token, expert) pair: int8 per 32 with a float scale. corr[c] holds, for chunk c,
// 16 x the block sums spread over the block's 8 lanes, which removes the +128 weight offset.
struct alignas(64) HAct {
    std::int8_t q[640];
    float d[20];
    __m512i corr[10];
};

struct CpuExperts::Layer {
    GateUp gu{};
    Down dn{};
    Buffer gate, up, down;
    std::size_t gu_row_bytes = 0, gu_expert_bytes = 0, dn_row_bytes = 0, dn_expert_bytes = 0;
};

struct CpuExperts::Scratch {
    XAct x[kMaxTokens];
    HAct h[kMaxTokens * kUsed];
    alignas(64) float g[kMaxTokens * kUsed][kFF];
    alignas(64) float u[kMaxTokens * kUsed][kFF];
};

namespace {

void quantize_x(const float * x, XAct & a) {
    for (int b = 0; b < 10; ++b) {
        float amax = 0;
        for (int j = 0; j < 256; ++j) amax = std::max(amax, std::fabs(x[256 * b + j]));
        const float d = amax / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
        a.d[b] = d;
        for (int j = 0; j < 256; ++j) a.q[256 * b + j] = std::int8_t(std::lrintf(x[256 * b + j] * id));
    }
    for (int k = 0; k < 80; ++k) {
        int s = 0;
        for (int j = 0; j < 32; ++j) s += a.q[32 * k + j];
        a.sum32[k] = s;
    }
}

void quantize_h(const float * h, HAct & a) {
    std::int32_t sums[20];
    for (int b = 0; b < 20; ++b) {
        float amax = 0;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(h[32 * b + j]));
        const float d = amax / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
        a.d[b] = d;
        int s = 0;
        for (int j = 0; j < 32; ++j) {
            const std::int8_t q = std::int8_t(std::lrintf(h[32 * b + j] * id));
            a.q[32 * b + j] = q;
            s += q;
        }
        sums[b] = s;
    }
    for (int c = 0; c < 10; ++c)
        a.corr[c] = _mm512_inserti64x4(_mm512_set1_epi32(16 * sums[2 * c]), _mm256_set1_epi32(16 * sums[2 * c + 1]), 1);
}

// Gate and up rows (Q4L) against NT tokens' activations, one pass over the weights.
template <int NT>
void gu_q4l(const BlockQ4L * g, const BlockQ4L * u, const XAct * const * xs, float * og, float * ou) {
    const __m512i lut = lut_q4l();
    float ag[NT] = {}, au[NT] = {};
    for (int b = 0; b < 10; ++b) {
        __m512i ig[NT], iu[NT];
        int cg[NT], cu[NT];
        for (int t = 0; t < NT; ++t) { ig[t] = iu[t] = _mm512_setzero_si512(); cg[t] = cu[t] = 0; }
        for (int c = 0; c < 4; ++c) {
            const int sg0 = 2 * (g[b].scales[c] & 0xF) + 1, sg1 = 2 * (g[b].scales[c] >> 4) + 1;
            const int su0 = 2 * (u[b].scales[c] & 0xF) + 1, su1 = 2 * (u[b].scales[c] >> 4) + 1;
            const __m512i scg = _mm512_inserti64x4(_mm512_set1_epi16(short(sg0)), _mm256_set1_epi16(short(sg1)), 1);
            const __m512i scu = _mm512_inserti64x4(_mm512_set1_epi16(short(su0)), _mm256_set1_epi16(short(su1)), 1);
            const __m512i wg = _mm512_shuffle_epi8(lut, chunk_codes(g[b].qs + 32 * c));
            const __m512i wu = _mm512_shuffle_epi8(lut, chunk_codes(u[b].qs + 32 * c));
            const int k = 8 * b + 2 * c;
            for (int t = 0; t < NT; ++t) {
                const __m512i x = _mm512_load_si512(xs[t]->q + 256 * b + 64 * c);
                ig[t] = _mm512_dpwssd_epi32(ig[t], _mm512_maddubs_epi16(wg, x), scg);
                iu[t] = _mm512_dpwssd_epi32(iu[t], _mm512_maddubs_epi16(wu, x), scu);
                cg[t] += sg0 * xs[t]->sum32[k] + sg1 * xs[t]->sum32[k + 1];
                cu[t] += su0 * xs[t]->sum32[k] + su1 * xs[t]->sum32[k + 1];
            }
        }
        const float dg = half_to_float(g[b].d), du = half_to_float(u[b].d);
        for (int t = 0; t < NT; ++t) {
            ag[t] += float(_mm512_reduce_add_epi32(ig[t]) - 16 * cg[t]) * dg * xs[t]->d[b];
            au[t] += float(_mm512_reduce_add_epi32(iu[t]) - 16 * cu[t]) * du * xs[t]->d[b];
        }
    }
    for (int t = 0; t < NT; ++t) { og[t] = ag[t]; ou[t] = au[t]; }
}

// Gate and up rows (Q4X, from IQ4_XS).
template <int NT>
void gu_q4x(const BlockQ4X * g, const BlockQ4X * u, const XAct * const * xs, float * og, float * ou) {
    const __m512i lut = lut_iq4();
    float ag[NT] = {}, au[NT] = {};
    for (int b = 0; b < 10; ++b) {
        __m512i ig[NT], iu[NT];
        for (int t = 0; t < NT; ++t) ig[t] = iu[t] = _mm512_setzero_si512();
        for (int c = 0; c < 4; ++c) {
            const __m512i sg = _mm512_inserti64x4(_mm512_set1_epi32(g[b].scales[2 * c]), _mm256_set1_epi32(g[b].scales[2 * c + 1]), 1);
            const __m512i su = _mm512_inserti64x4(_mm512_set1_epi32(u[b].scales[2 * c]), _mm256_set1_epi32(u[b].scales[2 * c + 1]), 1);
            const __m512i wg = _mm512_shuffle_epi8(lut, chunk_codes(g[b].qs + 32 * c));
            const __m512i wu = _mm512_shuffle_epi8(lut, chunk_codes(u[b].qs + 32 * c));
            const int k = 8 * b + 2 * c;
            for (int t = 0; t < NT; ++t) {
                const __m512i x = _mm512_load_si512(xs[t]->q + 256 * b + 64 * c);
                const __m512i corr = _mm512_inserti64x4(_mm512_set1_epi32(16 * xs[t]->sum32[k]), _mm256_set1_epi32(16 * xs[t]->sum32[k + 1]), 1);
                ig[t] = _mm512_add_epi32(ig[t], _mm512_mullo_epi32(_mm512_sub_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), wg, x), corr), sg));
                iu[t] = _mm512_add_epi32(iu[t], _mm512_mullo_epi32(_mm512_sub_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), wu, x), corr), su));
            }
        }
        const float dg = half_to_float(g[b].d), du = half_to_float(u[b].d);
        for (int t = 0; t < NT; ++t) {
            ag[t] += float(_mm512_reduce_add_epi32(ig[t])) * dg * xs[t]->d[b];
            au[t] += float(_mm512_reduce_add_epi32(iu[t])) * du * xs[t]->d[b];
        }
    }
    for (int t = 0; t < NT; ++t) { og[t] = ag[t]; ou[t] = au[t]; }
}

// One down row (IQ4L) against NT pairs' h.
template <int NT>
void dn_iq4l(const RowIQ4L * row, const HAct * const * hs, float * out) {
    const __m512i lut = lut_iq4();
    __m512 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_ps();
    for (int c = 0; c < 10; ++c) {
        const __m512i w = _mm512_shuffle_epi8(lut, chunk_codes(row->qs + 32 * c));
        const float d0 = half_to_float(row->d[2 * c]), d1 = half_to_float(row->d[2 * c + 1]);
        for (int t = 0; t < NT; ++t) {
            const __m512i v = _mm512_sub_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), w, _mm512_load_si512(hs[t]->q + 64 * c)), hs[t]->corr[c]);
            const __m512 s = _mm512_insertf32x8(_mm512_set1_ps(d0 * hs[t]->d[2 * c]), _mm256_set1_ps(d1 * hs[t]->d[2 * c + 1]), 1);
            acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(v), s, acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = _mm512_reduce_add_ps(acc[t]);
}

// One down row (Q8_0: 20 blocks) against NT pairs' h.
template <int NT>
void dn_q8(const BlockQ8_0 * row, const HAct * const * hs, float * out) {
    const __m512i flip = _mm512_set1_epi8(char(0x80));
    __m512 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_ps();
    for (int c = 0; c < 10; ++c) {
        const __m256i lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row[2 * c].qs));
        const __m256i hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(row[2 * c + 1].qs));
        const __m512i w = _mm512_xor_si512(_mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1), flip);  // +128
        const float d0 = half_to_float(row[2 * c].d), d1 = half_to_float(row[2 * c + 1].d);
        for (int t = 0; t < NT; ++t) {
            const __m512i v = _mm512_sub_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), w, _mm512_load_si512(hs[t]->q + 64 * c)), hs[t]->corr[c]);
            const __m512 s = _mm512_insertf32x8(_mm512_set1_ps(d0 * hs[t]->d[2 * c]), _mm256_set1_ps(d1 * hs[t]->d[2 * c + 1]), 1);
            acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(v), s, acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = _mm512_reduce_add_ps(acc[t]);
}

}  // namespace

CpuExperts::CpuExperts(const GgufModel & model, const CpuExpertsConfig & config)
    : pool_(config.threads), scratch_(std::make_unique<Scratch>()) {
    const int n_layer = int(model.get_int("qwen4exp.block_count"));
    std::vector<int> want = config.layers;
    if (want.empty())
        for (int i = 0; i < n_layer; ++i) want.push_back(i);
    layers_.resize(std::size_t(n_layer));
    for (int il : want) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const GgufTensor & tg = model.tensor(p + "ffn_gate_exps.weight");
        const GgufTensor & tu = model.tensor(p + "ffn_up_exps.weight");
        const GgufTensor & td = model.tensor(p + "ffn_down_exps.weight");
        if (tg.shape != std::vector<std::int64_t>{kEmbd, kFF, kExperts} || td.shape != std::vector<std::int64_t>{kFF, kEmbd, kExperts})
            throw std::runtime_error("unexpected expert shapes in layer " + std::to_string(il));
        auto L = std::make_unique<Layer>();
        if (tg.type == GgufType::IQ3_S && tu.type == GgufType::IQ3_S) {
            L->gu = GateUp::Q4L;
            L->gu_row_bytes = 10 * sizeof(BlockQ4L);
        } else if (tg.type == GgufType::IQ4_XS && tu.type == GgufType::IQ4_XS) {
            L->gu = GateUp::Q4X;
            L->gu_row_bytes = 10 * sizeof(BlockQ4X);
        } else {
            throw std::runtime_error("layer " + std::to_string(il) + ": gate/up format " + type_name(tg.type) + " is not supported");
        }
        if (td.type == GgufType::IQ4_NL) {
            L->dn = Down::IQ4L;
            L->dn_row_bytes = sizeof(RowIQ4L);
        } else if (td.type == GgufType::Q8_0) {
            L->dn = Down::Q8_0;
            L->dn_row_bytes = 20 * sizeof(BlockQ8_0);
        } else {
            throw std::runtime_error("layer " + std::to_string(il) + ": down format " + type_name(td.type) + " is not supported");
        }
        L->gu_expert_bytes = L->gu_row_bytes * kFF;
        L->dn_expert_bytes = L->dn_row_bytes * kEmbd;
        L->gate.alloc(L->gu_expert_bytes * kExperts);
        L->up.alloc(L->gu_expert_bytes * kExperts);
        L->down.alloc(L->dn_expert_bytes * kExperts);
        Layer * Lp = L.get();
        const int nt = pool_.size();
        pool_.run([&](int t) {
            for (int e = t; e < kExperts; e += nt) {
                if (Lp->gu == GateUp::Q4L) {
                    const std::size_t src_bytes = std::size_t(kFF) * 10 * sizeof(BlockIQ3_S);
                    repack_iq3s_to_q4l(reinterpret_cast<const BlockIQ3_S *>(tg.data + e * src_bytes),
                                       reinterpret_cast<BlockQ4L *>(Lp->gate.p + e * Lp->gu_expert_bytes), kFF * 10);
                    repack_iq3s_to_q4l(reinterpret_cast<const BlockIQ3_S *>(tu.data + e * src_bytes),
                                       reinterpret_cast<BlockQ4L *>(Lp->up.p + e * Lp->gu_expert_bytes), kFF * 10);
                } else {
                    const std::size_t src_bytes = std::size_t(kFF) * 10 * sizeof(BlockIQ4_XS);
                    for (int i = 0; i < kFF * 10; ++i) {
                        repack_iq4xs_block(reinterpret_cast<const BlockIQ4_XS *>(tg.data + e * src_bytes)[i],
                                           reinterpret_cast<BlockQ4X *>(Lp->gate.p + e * Lp->gu_expert_bytes)[i]);
                        repack_iq4xs_block(reinterpret_cast<const BlockIQ4_XS *>(tu.data + e * src_bytes)[i],
                                           reinterpret_cast<BlockQ4X *>(Lp->up.p + e * Lp->gu_expert_bytes)[i]);
                    }
                }
                if (Lp->dn == Down::IQ4L) {
                    const std::size_t src_row = 20 * sizeof(BlockIQ4_NL);
                    for (int r = 0; r < kEmbd; ++r)
                        repack_iq4nl_row(reinterpret_cast<const BlockIQ4_NL *>(td.data + (std::size_t(e) * kEmbd + r) * src_row),
                                         reinterpret_cast<RowIQ4L *>(Lp->down.p + e * Lp->dn_expert_bytes)[r]);
                } else {
                    std::memcpy(Lp->down.p + e * Lp->dn_expert_bytes, td.data + e * Lp->dn_expert_bytes, Lp->dn_expert_bytes);
                }
            }
        });
        resident_bytes_ += L->gate.n + L->up.n + L->down.n;
        layers_[std::size_t(il)] = std::move(L);
    }
}

CpuExperts::~CpuExperts() = default;

bool CpuExperts::has_layer(int layer) const {
    return layer >= 0 && layer < int(layers_.size()) && layers_[std::size_t(layer)] != nullptr;
}

std::size_t CpuExperts::expert_bytes(int layer) const {
    const Layer & L = *layers_.at(std::size_t(layer));
    return 2 * L.gu_expert_bytes + L.dn_expert_bytes;
}

const char * CpuExperts::format_name(int layer) const {
    const Layer & L = *layers_.at(std::size_t(layer));
    if (L.gu == GateUp::Q4L) return L.dn == Down::IQ4L ? "q4l/q4l/iq4l" : "q4l/q4l/q8_0";
    return L.dn == Down::IQ4L ? "q4x/q4x/iq4l" : "q4x/q4x/q8_0";
}

void CpuExperts::run(int layer, int n_tokens, const float * x, const std::int32_t * ids, const float * weights,
                     const std::uint8_t * on_cpu, float * out) {
    if (!has_layer(layer)) throw std::runtime_error("CpuExperts: layer " + std::to_string(layer) + " is not loaded");
    if (n_tokens < 1 || n_tokens > kMaxTokens) throw std::runtime_error("CpuExperts: n_tokens out of range");
    const Layer & L = *layers_[std::size_t(layer)];
    Scratch & S = *scratch_;

    // Group the (token, slot) pairs by expert: experts[u], and its pairs at pair_start[u]..pair_start[u+1].
    int experts[kMaxTokens * kUsed], pair_start[kMaxTokens * kUsed + 1], pair_token[kMaxTokens * kUsed];
    float pair_weight[kMaxTokens * kUsed];
    int n_experts = 0, n_pairs = 0;
    {
        int seen[kExperts];
        std::fill(seen, seen + kExperts, -1);
        int count[kMaxTokens * kUsed] = {};
        for (int t = 0; t < n_tokens; ++t)
            for (int k = 0; k < kUsed; ++k) {
                if (on_cpu && !on_cpu[t * kUsed + k]) continue;
                const int e = ids[t * kUsed + k];
                if (seen[e] < 0) { seen[e] = n_experts; experts[n_experts++] = e; }
                ++count[seen[e]];
            }
        pair_start[0] = 0;
        for (int u = 0; u < n_experts; ++u) pair_start[u + 1] = pair_start[u] + count[u];
        int fill[kMaxTokens * kUsed] = {};
        for (int t = 0; t < n_tokens; ++t)
            for (int k = 0; k < kUsed; ++k) {
                if (on_cpu && !on_cpu[t * kUsed + k]) continue;
                const int u = seen[ids[t * kUsed + k]];
                const int p = pair_start[u] + fill[u]++;
                pair_token[p] = t;
                pair_weight[p] = weights[t * kUsed + k];
            }
        n_pairs = pair_start[n_experts];
    }
    std::memset(out, 0, sizeof(float) * std::size_t(n_tokens) * kEmbd);
    if (n_pairs == 0) return;

    for (int t = 0; t < n_tokens; ++t) quantize_x(x + std::size_t(t) * kEmbd, S.x[t]);

    const int nt = pool_.size();
    // Gate/up: rows of all selected experts, spread evenly over the threads.
    const int gu_rows = n_experts * kFF;
    pool_.run([&](int th) {
        const int r0 = gu_rows * th / nt, r1 = gu_rows * (th + 1) / nt;
        for (int r = r0; r < r1; ++r) {
            const int u = r / kFF, row = r % kFF;
            const std::uint8_t * gp = L.gate.p + std::size_t(experts[u]) * L.gu_expert_bytes + std::size_t(row) * L.gu_row_bytes;
            const std::uint8_t * up = L.up.p + std::size_t(experts[u]) * L.gu_expert_bytes + std::size_t(row) * L.gu_row_bytes;
            for (int p0 = pair_start[u]; p0 < pair_start[u + 1]; p0 += 4) {
                const int n = std::min(4, pair_start[u + 1] - p0);
                const XAct * xs[4];
                float og[4], ou[4];
                for (int i = 0; i < n; ++i) xs[i] = &S.x[pair_token[p0 + i]];
                if (L.gu == GateUp::Q4L) {
                    const auto * g = reinterpret_cast<const BlockQ4L *>(gp);
                    const auto * uu = reinterpret_cast<const BlockQ4L *>(up);
                    switch (n) {
                    case 1: gu_q4l<1>(g, uu, xs, og, ou); break;
                    case 2: gu_q4l<2>(g, uu, xs, og, ou); break;
                    case 3: gu_q4l<3>(g, uu, xs, og, ou); break;
                    default: gu_q4l<4>(g, uu, xs, og, ou); break;
                    }
                } else {
                    const auto * g = reinterpret_cast<const BlockQ4X *>(gp);
                    const auto * uu = reinterpret_cast<const BlockQ4X *>(up);
                    switch (n) {
                    case 1: gu_q4x<1>(g, uu, xs, og, ou); break;
                    case 2: gu_q4x<2>(g, uu, xs, og, ou); break;
                    case 3: gu_q4x<3>(g, uu, xs, og, ou); break;
                    default: gu_q4x<4>(g, uu, xs, og, ou); break;
                    }
                }
                for (int i = 0; i < n; ++i) { S.g[p0 + i][row] = og[i]; S.u[p0 + i][row] = ou[i]; }
            }
        }
    });

    // SwiGLU and quantization of each pair's down input.
    pool_.run([&](int th) {
        float h[kFF];
        for (int p = th; p < n_pairs; p += nt) {
            for (int j = 0; j < kFF; ++j) {
                const float gv = S.g[p][j];
                h[j] = gv / (1.0f + std::exp(-gv)) * S.u[p][j];
            }
            quantize_h(h, S.h[p]);
        }
    });

    // Down: each thread owns a slice of output rows and runs every pair against it.
    pool_.run([&](int th) {
        const int r0 = kEmbd * th / nt, r1 = kEmbd * (th + 1) / nt;
        for (int r = r0; r < r1; ++r) {
            float acc[kMaxTokens] = {};
            for (int u = 0; u < n_experts; ++u) {
                const std::uint8_t * row = L.down.p + std::size_t(experts[u]) * L.dn_expert_bytes + std::size_t(r) * L.dn_row_bytes;
                for (int p0 = pair_start[u]; p0 < pair_start[u + 1]; p0 += 4) {
                    const int n = std::min(4, pair_start[u + 1] - p0);
                    const HAct * hs[4];
                    float d[4];
                    for (int i = 0; i < n; ++i) hs[i] = &S.h[p0 + i];
                    if (L.dn == Down::IQ4L) {
                        const auto * rr = reinterpret_cast<const RowIQ4L *>(row);
                        switch (n) {
                        case 1: dn_iq4l<1>(rr, hs, d); break;
                        case 2: dn_iq4l<2>(rr, hs, d); break;
                        case 3: dn_iq4l<3>(rr, hs, d); break;
                        default: dn_iq4l<4>(rr, hs, d); break;
                        }
                    } else {
                        const auto * rr = reinterpret_cast<const BlockQ8_0 *>(row);
                        switch (n) {
                        case 1: dn_q8<1>(rr, hs, d); break;
                        case 2: dn_q8<2>(rr, hs, d); break;
                        case 3: dn_q8<3>(rr, hs, d); break;
                        default: dn_q8<4>(rr, hs, d); break;
                        }
                    }
                    for (int i = 0; i < n; ++i) acc[pair_token[p0 + i]] += pair_weight[p0 + i] * d[i];
                }
            }
            for (int t = 0; t < n_tokens; ++t) out[std::size_t(t) * kEmbd + r] = acc[t];
        }
    });
}

}  // namespace ninfer::flashnext
