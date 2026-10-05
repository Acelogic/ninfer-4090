#include "flashnext/cpu_experts.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <immintrin.h>

#include "flashnext/iq3s_grid.h"
#include "flashnext/quants.h"

namespace ninfer::flashnext {

namespace {


// Q8_0: gate/up as in the GGUF (the MTP head's experts); run() only, always with 16-bit activations
enum class GateUp { Q4L, Q4X, Q8_0 };
enum class Down { IQ4L, Q8_0 };

struct Buffer {
    std::uint8_t * p = nullptr;
    std::size_t n = 0;
    Buffer() = default;
    Buffer(const Buffer &) = delete;
    Buffer & operator=(const Buffer &) = delete;
    // align: a power of two; page-aligned (4096) buffers can be registered with the CUDA driver as a whole
    void alloc(std::size_t bytes, std::size_t align = 64) {
        release();
#if defined(_WIN32)
        p = static_cast<std::uint8_t *>(_aligned_malloc((bytes + align - 1) / align * align, align));
#else
        p = static_cast<std::uint8_t *>(std::aligned_alloc(align, (bytes + align - 1) / align * align));
#endif
        if (!p) throw std::bad_alloc();
        n = bytes;
    }
    void release() {
#if defined(_WIN32)
        _aligned_free(p);
#else
        std::free(p);
#endif
        p = nullptr;
        n = 0;
    }
    ~Buffer() { release(); }
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

inline __m512i load_2x256(const std::uint8_t * lo, const std::uint8_t * hi) {
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(lo))),
                              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(hi)), 1);
}

// Code -> signed weight, for the 16-bit activation kernels.
inline __m512i lut_q4l_signed() {
    alignas(16) static const std::int8_t t[16] = {1, 3, 5, 7, 9, 11, 13, 15, -1, -3, -5, -7, -9, -11, -13, -15};
    return _mm512_broadcast_i32x4(_mm_load_si128(reinterpret_cast<const __m128i *>(t)));
}
inline __m512i lut_iq4_signed() { return _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i *>(kIQ4NLValues))); }

// 32 bytes of chunked nibbles -> 64 codes in weight order.
inline __m512i chunk_codes(const std::uint8_t * p) {
    const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(p));
    return _mm512_and_si512(_mm512_inserti64x4(_mm512_castsi256_si512(raw), _mm256_srli_epi16(raw, 4), 1), _mm512_set1_epi8(0x0F));
}

// 8-bit quantization of N floats (N a multiple of 32) with one scale, as llama.cpp's q8 blocks: scale amax/127,
// round to nearest even. Writes the values and the sum of each 32, returns the scale. Every result is exact, so it
// equals the scalar loop it replaced bit for bit.
template <int N>
inline float quantize_q8(const float * x, std::int8_t * q, std::int32_t * sums) {
    __m512 m = _mm512_setzero_ps();
    for (int j = 0; j < N / 16; ++j) m = _mm512_max_ps(m, _mm512_abs_ps(_mm512_loadu_ps(x + 16 * j)));
    const float d = _mm512_reduce_max_ps(m) / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
    const __m512 vid = _mm512_set1_ps(id);
    for (int k = 0; k < N / 32; ++k) {
        const __m512i a = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + 32 * k), vid));
        const __m512i b = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + 32 * k + 16), vid));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(q + 32 * k), _mm512_cvtepi32_epi8(a));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(q + 32 * k + 16), _mm512_cvtepi32_epi8(b));
        sums[k] = _mm512_reduce_add_epi32(_mm512_add_epi32(a, b));
    }
    return d;
}

// 16-bit quantization of N floats (N a multiple of 16) with one scale amax/32767, round to nearest even.
template <int N>
inline float quantize_q16(const float * x, std::int16_t * q) {
    __m512 m = _mm512_setzero_ps();
    for (int j = 0; j < N / 16; ++j) m = _mm512_max_ps(m, _mm512_abs_ps(_mm512_loadu_ps(x + 16 * j)));
    const float d = _mm512_reduce_max_ps(m) / 32767.0f, id = d > 0 ? 1.0f / d : 0.0f;
    const __m512 vid = _mm512_set1_ps(id);
    for (int j = 0; j < N / 16; ++j)
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(q + 16 * j), _mm512_cvtepi32_epi16(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + 16 * j), vid))));
    return d;
}

// exp(x) for 16 floats: Cephes' expf polynomial (about 1 ulp), 0 and +inf beyond the float range.
inline __m512 exp16(__m512 x) {
    x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-104.0f)), _mm512_set1_ps(89.0f));
    const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 p = _mm512_set1_ps(1.9875691500e-4f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.3981999507e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.3334519073e-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.1665795894e-2f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.6666665459e-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0000001201e-1f));
    p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), _mm512_add_ps(r, _mm512_set1_ps(1.0f)));
    return _mm512_scalef_ps(p, n);
}

// One logical processor (as a one-bit mask) of each physical core this process may use, in core order.
std::vector<std::uint64_t> core_affinities() {
    std::vector<std::uint64_t> cores;
#if defined(_WIN32)
    DWORD_PTR process = 0, system = 0;
    DWORD len = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process, &system)) return cores;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<std::uint8_t> buf(len);
    if (len == 0 || !GetLogicalProcessorInformationEx(RelationProcessorCore, reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data()), &len))
        return cores;
    for (DWORD off = 0; off < len;) {
        const auto * r = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data() + off);
        off += r->Size;
        if (r->Relationship != RelationProcessorCore || r->Processor.GroupMask[0].Group != 0) continue;
        const std::uint64_t m = std::uint64_t(r->Processor.GroupMask[0].Mask & process);
        if (m) cores.push_back(m & (~m + 1));
    }
#endif
    return cores;
}

// Pins the calling thread to `mask` for its lifetime and restores the previous affinity.
class AffinityScope {
public:
    explicit AffinityScope(std::uint64_t mask) {
#if defined(_WIN32)
        if (mask) old_ = SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(mask));
#else
        (void)mask;
#endif
    }
    ~AffinityScope() {
#if defined(_WIN32)
        if (old_) SetThreadAffinityMask(GetCurrentThread(), old_);
#endif
    }
    AffinityScope(const AffinityScope &) = delete;
    AffinityScope & operator=(const AffinityScope &) = delete;

private:
#if defined(_WIN32)
    DWORD_PTR old_ = 0;
#endif
};

inline void prefetch_for_write(const void * p) {
#if defined(_MSC_VER)
    _m_prefetchw(p);
#else
    __builtin_prefetch(p, 1, 3);
#endif
}

// h = silu(g) * u for 16 values. run() and run_batch() both use it, so they quantize identical h.
inline __m512 swiglu16(__m512 g, __m512 u) {
    const __m512 e = exp16(_mm512_sub_ps(_mm512_setzero_ps(), g));
    return _mm512_mul_ps(_mm512_div_ps(g, _mm512_add_ps(_mm512_set1_ps(1.0f), e)), u);
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
// Precise mode: the same blocks with 16-bit values (scale amax/32767). The kernels widen the weights to signed int16,
// so no offset correction is needed. run() stores the values of each 64-column chunk in "lane order" (see
// widen_lanes), run_batch() in column order.
struct alignas(64) XAct16 {
    std::int16_t q[2560];
    float d[10];
};
struct alignas(64) HAct16 {
    std::int16_t q[640];
    float d[20];
};

// Every expert's matrices are stored together, [gate | up | down] at e * stride, so that one expert (or a run of
// experts with consecutive ids) is one contiguous range: the GPU streams them with one copy each (prefill).
struct CpuExperts::Layer {
    GateUp gu{};
    Down dn{};
    Buffer blob;  // [kExperts][stride], page-aligned
    std::uint8_t *gate = nullptr, *up = nullptr, *down = nullptr;  // expert 0's matrices; expert e's at + e * stride
    std::size_t stride = 0;
    std::size_t gu_row_bytes = 0, gu_expert_bytes = 0, dn_row_bytes = 0, dn_expert_bytes = 0;
};

struct CpuExperts::Scratch {
    XAct x[kMaxTokens];
    HAct h[kMaxTokens * kUsed];
    XAct16 x16[kMaxTokens];
    HAct16 h16[kMaxTokens * kUsed];
    alignas(64) float g[kMaxTokens * kUsed][kFF];
    alignas(64) float u[kMaxTokens * kUsed][kFF];
};

namespace {

void quantize_x(const float * x, XAct & a) {
    for (int b = 0; b < 10; ++b) a.d[b] = quantize_q8<256>(x + 256 * b, a.q + 256 * b, a.sum32 + 8 * b);
}

void quantize_h(const float * h, HAct & a) {
    std::int32_t sums[20];
    for (int b = 0; b < 20; ++b) a.d[b] = quantize_q8<32>(h + 32 * b, a.q + 32 * b, sums + b);
    for (int c = 0; c < 10; ++c)
        a.corr[c] = _mm512_inserti64x4(_mm512_set1_epi32(16 * sums[2 * c]), _mm256_set1_epi32(16 * sums[2 * c + 1]), 1);
}

void quantize_x16(const float * x, XAct16 & a) {
    for (int b = 0; b < 10; ++b) a.d[b] = quantize_q16<256>(x + 256 * b, a.q + 256 * b);
}

// Column order -> lane order within each 64-column chunk: first the columns 16L..16L+7 of each 128-bit lane L, then
// 16L+8..16L+15, which is the order in which widen_lanes produces the weights.
void to_lane_order(std::int16_t * q, int n) {
    alignas(64) static const std::int16_t ia[32] = {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23,
                                                    32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55};
    const __m512i a = _mm512_load_si512(ia), b = _mm512_add_epi16(a, _mm512_set1_epi16(8));
    for (int c = 0; c < n; c += 64) {
        const __m512i lo = _mm512_loadu_si512(q + c), hi = _mm512_loadu_si512(q + c + 32);
        _mm512_storeu_si512(q + c, _mm512_permutex2var_epi16(lo, a, hi));
        _mm512_storeu_si512(q + c + 32, _mm512_permutex2var_epi16(lo, b, hi));
    }
}

void quantize_x16_lanes(const float * x, XAct16 & a) {
    quantize_x16(x, a);
    to_lane_order(a.q, 2560);
}

void quantize_h16_lanes(const float * h, HAct16 & a) {
    for (int b = 0; b < 20; ++b) a.d[b] = quantize_q16<32>(h + 32 * b, a.q + 32 * b);
    to_lane_order(a.q, 640);
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

// Precise mode (16-bit activations): 64 int8 weights of a chunk (columns in byte order) -> int16, without crossing
// 128-bit lanes: a holds columns 16L..16L+7 of each lane L, b columns 16L+8..16L+15. Both hold the chunk's first
// 32-column block in dword lanes 0-7 and the second in lanes 8-15; the activations are stored in the same order.
inline void widen_lanes(__m512i v, __m512i & a, __m512i & b) {
    a = _mm512_srai_epi16(_mm512_unpacklo_epi8(v, v), 8);
    b = _mm512_srai_epi16(_mm512_unpackhi_epi8(v, v), 8);
}

// Gate and up rows against NT tokens, int16 x int16. The sub-block scale is folded into the int16 weights (|w x s| <=
// 15 x 31 for Q4L, 127 x 32 for Q4X); each int32 lane sums 16 products per 256-block, at most 2.13e9, so the block sums
// are exact. They are converted to float once per block.
template <int NT, bool Q4X, class Block>
void gu_p(const Block * g, const Block * u, const XAct16 * const * xs, float * og, float * ou) {
    const __m512i lut = Q4X ? lut_iq4_signed() : lut_q4l_signed();
    __m512 ag[NT], au[NT];
    for (int t = 0; t < NT; ++t) ag[t] = au[t] = _mm512_setzero_ps();
    for (int b = 0; b < 10; ++b) {
        __m512i ig[NT], iu[NT];
        for (int t = 0; t < NT; ++t) ig[t] = iu[t] = _mm512_setzero_si512();
        for (int c = 0; c < 4; ++c) {
            short sg0, sg1, su0, su1;
            if constexpr (Q4X) {
                sg0 = g[b].scales[2 * c], sg1 = g[b].scales[2 * c + 1], su0 = u[b].scales[2 * c], su1 = u[b].scales[2 * c + 1];
            } else {
                sg0 = short(2 * (g[b].scales[c] & 0xF) + 1), sg1 = short(2 * (g[b].scales[c] >> 4) + 1);
                su0 = short(2 * (u[b].scales[c] & 0xF) + 1), su1 = short(2 * (u[b].scales[c] >> 4) + 1);
            }
            const __m512i sg = _mm512_inserti64x4(_mm512_set1_epi16(sg0), _mm256_set1_epi16(sg1), 1);
            const __m512i su = _mm512_inserti64x4(_mm512_set1_epi16(su0), _mm256_set1_epi16(su1), 1);
            __m512i g0, g1, u0, u1;
            widen_lanes(_mm512_shuffle_epi8(lut, chunk_codes(g[b].qs + 32 * c)), g0, g1);
            widen_lanes(_mm512_shuffle_epi8(lut, chunk_codes(u[b].qs + 32 * c)), u0, u1);
            g0 = _mm512_mullo_epi16(g0, sg);
            g1 = _mm512_mullo_epi16(g1, sg);
            u0 = _mm512_mullo_epi16(u0, su);
            u1 = _mm512_mullo_epi16(u1, su);
            for (int t = 0; t < NT; ++t) {
                const __m512i x0 = _mm512_load_si512(xs[t]->q + 256 * b + 64 * c), x1 = _mm512_load_si512(xs[t]->q + 256 * b + 64 * c + 32);
                ig[t] = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(ig[t], g0, x0), g1, x1);
                iu[t] = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(iu[t], u0, x0), u1, x1);
            }
        }
        const float dg = half_to_float(g[b].d), du = half_to_float(u[b].d);
        for (int t = 0; t < NT; ++t) {
            ag[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ig[t]), _mm512_set1_ps(dg * xs[t]->d[b]), ag[t]);
            au[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iu[t]), _mm512_set1_ps(du * xs[t]->d[b]), au[t]);
        }
    }
    for (int t = 0; t < NT; ++t) { og[t] = _mm512_reduce_add_ps(ag[t]); ou[t] = _mm512_reduce_add_ps(au[t]); }
}

// One down row against NT pairs' 16-bit h. Per chunk, lanes 0-7 sum the first 32-column block (one IQ4_NL / Q8_0
// block) exactly and lanes 8-15 the second; each half is scaled by its d(row) * d(h).
template <int NT, bool Q8>
void dn_p(const std::uint8_t * row, const HAct16 * const * hs, float * out) {
    __m512 acc[NT];
    for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_ps();
    for (int c = 0; c < 10; ++c) {
        __m512i w0, w1;
        float d0, d1;
        if constexpr (Q8) {
            const auto * r = reinterpret_cast<const BlockQ8_0 *>(row);
            widen_lanes(load_2x256(reinterpret_cast<const std::uint8_t *>(r[2 * c].qs), reinterpret_cast<const std::uint8_t *>(r[2 * c + 1].qs)), w0, w1);
            d0 = half_to_float(r[2 * c].d), d1 = half_to_float(r[2 * c + 1].d);
        } else {
            const auto * r = reinterpret_cast<const RowIQ4L *>(row);
            widen_lanes(_mm512_shuffle_epi8(lut_iq4_signed(), chunk_codes(r->qs + 32 * c)), w0, w1);
            d0 = half_to_float(r->d[2 * c]), d1 = half_to_float(r->d[2 * c + 1]);
        }
        for (int t = 0; t < NT; ++t) {
            const __m512i v = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(_mm512_setzero_si512(), w0, _mm512_load_si512(hs[t]->q + 64 * c)), w1,
                                                  _mm512_load_si512(hs[t]->q + 64 * c + 32));
            const __m512 sc = _mm512_insertf32x8(_mm512_set1_ps(d0 * hs[t]->d[2 * c]), _mm256_set1_ps(d1 * hs[t]->d[2 * c + 1]), 1);
            acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(v), sc, acc[t]);
        }
    }
    for (int t = 0; t < NT; ++t) out[t] = _mm512_reduce_add_ps(acc[t]);
}

// Gate and up rows in Q8_0 (80 blocks of 32 weights; the MTP head's experts) against NT tokens' 16-bit activations, as
// dn_p computes a Q8_0 down row: per 64-column chunk, lanes 0-7 sum the first block's products exactly in int32 and
// lanes 8-15 the second's; each half is scaled by its block's d times the activation scale of its 256-block.
template <int NT>
void gu_q8p(const BlockQ8_0 * g, const BlockQ8_0 * u, const XAct16 * const * xs, float * og, float * ou) {
    __m512 ag[NT], au[NT];
    for (int t = 0; t < NT; ++t) ag[t] = au[t] = _mm512_setzero_ps();
    for (int c = 0; c < 40; ++c) {
        __m512i g0, g1, u0, u1;
        widen_lanes(load_2x256(reinterpret_cast<const std::uint8_t *>(g[2 * c].qs), reinterpret_cast<const std::uint8_t *>(g[2 * c + 1].qs)), g0, g1);
        widen_lanes(load_2x256(reinterpret_cast<const std::uint8_t *>(u[2 * c].qs), reinterpret_cast<const std::uint8_t *>(u[2 * c + 1].qs)), u0, u1);
        const float dg0 = half_to_float(g[2 * c].d), dg1 = half_to_float(g[2 * c + 1].d);
        const float du0 = half_to_float(u[2 * c].d), du1 = half_to_float(u[2 * c + 1].d);
        const int b = c / 4;  // the activations' 256-block
        for (int t = 0; t < NT; ++t) {
            const __m512i x0 = _mm512_load_si512(xs[t]->q + 64 * c), x1 = _mm512_load_si512(xs[t]->q + 64 * c + 32);
            const __m512i vg = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(_mm512_setzero_si512(), g0, x0), g1, x1);
            const __m512i vu = _mm512_dpwssd_epi32(_mm512_dpwssd_epi32(_mm512_setzero_si512(), u0, x0), u1, x1);
            const float dx = xs[t]->d[b];
            ag[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vg), _mm512_insertf32x8(_mm512_set1_ps(dg0 * dx), _mm256_set1_ps(dg1 * dx), 1), ag[t]);
            au[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vu), _mm512_insertf32x8(_mm512_set1_ps(du0 * dx), _mm256_set1_ps(du1 * dx), 1), au[t]);
        }
    }
    for (int t = 0; t < NT; ++t) { og[t] = _mm512_reduce_add_ps(ag[t]); ou[t] = _mm512_reduce_add_ps(au[t]); }
}

}  // namespace

CpuExperts::CpuExperts(const GgufModel & model, const CpuExpertsConfig & config)
    : pool_(config.threads), precise_(config.precise_activations), scratch_(std::make_unique<Scratch>()) {
    if (config.pin_threads) {
        const std::vector<std::uint64_t> cores = core_affinities();
        if (int(cores.size()) >= pool_.size()) {
#if defined(_WIN32)
            pool_.run([&](int t) {
                if (t > 0) SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(cores[std::size_t(t)]));
            });
#endif
            caller_affinity_ = cores[0];
        }
    }
    const int n_layer = int(model.get_int("qwen4exp.block_count"));
    std::vector<int> want = config.layers;
    if (want.empty())
        for (int i = 0; i < n_layer; ++i) want.push_back(i);
    layers_.resize(std::size_t(n_layer));
    for (int il : want) load_layer(model, il);
}

void CpuExperts::add_layer(const GgufModel & model, int il) {
    if (il < 0) throw std::runtime_error("CpuExperts: negative layer");
    if (std::size_t(il) >= layers_.size()) layers_.resize(std::size_t(il) + 1);
    if (layers_[std::size_t(il)]) throw std::runtime_error("CpuExperts: layer " + std::to_string(il) + " is loaded already");
    load_layer(model, il);
}

void CpuExperts::load_layer(const GgufModel & model, int il) {
    {
        const std::string p = "blk." + std::to_string(il) + ".";
        const GgufTensor & tg = model.tensor(p + "ffn_gate_exps.weight");
        const GgufTensor & tu = model.tensor(p + "ffn_up_exps.weight");
        const GgufTensor & td = model.tensor(p + "ffn_down_exps.weight");
        if (tg.shape != std::vector<std::int64_t>{kEmbd, kFF, kExperts} || tu.shape != tg.shape || tu.type != tg.type ||
            td.shape != std::vector<std::int64_t>{kFF, kEmbd, kExperts})
            throw std::runtime_error("unexpected expert shapes in layer " + std::to_string(il));
        // The repack below reads exactly these sizes; the reader already bounds them to the file.
        if (tg.bytes != row_bytes(tg.type, kEmbd) * kFF * kExperts || tu.bytes != tg.bytes ||
            td.bytes != row_bytes(td.type, kFF) * std::size_t(kEmbd) * kExperts)
            throw std::runtime_error("unexpected expert tensor sizes in layer " + std::to_string(il));
        auto L = std::make_unique<Layer>();
        if (tg.type == GgufType::IQ3_S && tu.type == GgufType::IQ3_S) {
            L->gu = GateUp::Q4L;
            L->gu_row_bytes = 10 * sizeof(BlockQ4L);
        } else if (tg.type == GgufType::IQ4_XS && tu.type == GgufType::IQ4_XS) {
            L->gu = GateUp::Q4X;
            L->gu_row_bytes = 10 * sizeof(BlockQ4X);
        } else if (tg.type == GgufType::Q8_0 && tu.type == GgufType::Q8_0) {
            L->gu = GateUp::Q8_0;
            L->gu_row_bytes = 80 * sizeof(BlockQ8_0);
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
        L->stride = 2 * L->gu_expert_bytes + L->dn_expert_bytes;
        L->blob.alloc(L->stride * kExperts, 4096);
        L->gate = L->blob.p;
        L->up = L->blob.p + L->gu_expert_bytes;
        L->down = L->blob.p + 2 * L->gu_expert_bytes;
        Layer * Lp = L.get();
        const int nt = pool_.size();
        pool_.run([&](int t) {
            for (int e = t; e < kExperts; e += nt) {
                if (Lp->gu == GateUp::Q8_0) {
                    std::memcpy(Lp->gate + e * Lp->stride, tg.data + e * Lp->gu_expert_bytes, Lp->gu_expert_bytes);
                    std::memcpy(Lp->up + e * Lp->stride, tu.data + e * Lp->gu_expert_bytes, Lp->gu_expert_bytes);
                } else if (Lp->gu == GateUp::Q4L) {
                    const std::size_t src_bytes = std::size_t(kFF) * 10 * sizeof(BlockIQ3_S);
                    repack_iq3s_to_q4l(reinterpret_cast<const BlockIQ3_S *>(tg.data + e * src_bytes),
                                       reinterpret_cast<BlockQ4L *>(Lp->gate + e * Lp->stride), kFF * 10);
                    repack_iq3s_to_q4l(reinterpret_cast<const BlockIQ3_S *>(tu.data + e * src_bytes),
                                       reinterpret_cast<BlockQ4L *>(Lp->up + e * Lp->stride), kFF * 10);
                } else {
                    const std::size_t src_bytes = std::size_t(kFF) * 10 * sizeof(BlockIQ4_XS);
                    for (int i = 0; i < kFF * 10; ++i) {
                        repack_iq4xs_block(reinterpret_cast<const BlockIQ4_XS *>(tg.data + e * src_bytes)[i],
                                           reinterpret_cast<BlockQ4X *>(Lp->gate + e * Lp->stride)[i]);
                        repack_iq4xs_block(reinterpret_cast<const BlockIQ4_XS *>(tu.data + e * src_bytes)[i],
                                           reinterpret_cast<BlockQ4X *>(Lp->up + e * Lp->stride)[i]);
                    }
                }
                if (Lp->dn == Down::IQ4L) {
                    const std::size_t src_row = 20 * sizeof(BlockIQ4_NL);
                    for (int r = 0; r < kEmbd; ++r)
                        repack_iq4nl_row(reinterpret_cast<const BlockIQ4_NL *>(td.data + (std::size_t(e) * kEmbd + r) * src_row),
                                         reinterpret_cast<RowIQ4L *>(Lp->down + e * Lp->stride)[r]);
                } else {
                    std::memcpy(Lp->down + e * Lp->stride, td.data + e * Lp->dn_expert_bytes, Lp->dn_expert_bytes);
                }
            }
        });
        resident_bytes_ += L->blob.n;
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

CpuExperts::HostLayer CpuExperts::host_layer(int layer) const {
    if (!has_layer(layer)) throw std::runtime_error("CpuExperts: layer " + std::to_string(layer) + " is not loaded");
    const Layer & L = *layers_[std::size_t(layer)];
    HostLayer h;
    h.base = L.blob.p;
    h.stride = L.stride;
    h.gate_bytes = L.gu_expert_bytes;
    h.down_bytes = L.dn_expert_bytes;
    h.bytes = (L.stride * kExperts + 4095) / 4096 * 4096;
    h.gate_q4x = L.gu == GateUp::Q4X;
    h.down_q8 = L.dn == Down::Q8_0;
    return h;
}

const char * CpuExperts::format_name(int layer) const {
    const Layer & L = *layers_.at(std::size_t(layer));
    if (L.gu == GateUp::Q8_0) return L.dn == Down::IQ4L ? "q8_0/q8_0/iq4l" : "q8_0/q8_0/q8_0";
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

    // Q8_0 gate/up (the MTP head) has only the 16-bit activation kernels
    const bool precise = precise_ || L.gu == GateUp::Q8_0;
    for (int t = 0; t < n_tokens; ++t) {
        if (precise) quantize_x16_lanes(x + std::size_t(t) * kEmbd, S.x16[t]);
        else quantize_x(x + std::size_t(t) * kEmbd, S.x[t]);
    }

    const int nt = pool_.size();
    // Gate/up: rows of all selected experts, spread evenly over the threads.
    const int gu_rows = n_experts * kFF;
    pool_.run([&](int th) {
        const int r0 = gu_rows * th / nt, r1 = gu_rows * (th + 1) / nt;
        for (int r = r0; r < r1; ++r) {
            const int u = r / kFF, row = r % kFF;
            const std::uint8_t * gp = L.gate + std::size_t(experts[u]) * L.stride + std::size_t(row) * L.gu_row_bytes;
            const std::uint8_t * up = L.up + std::size_t(experts[u]) * L.stride + std::size_t(row) * L.gu_row_bytes;
            for (int p0 = pair_start[u]; p0 < pair_start[u + 1]; p0 += 4) {
                const int n = std::min(4, pair_start[u + 1] - p0);
                const XAct * xs[4];
                const XAct16 * xs16[4];
                float og[4], ou[4];
                for (int i = 0; i < n; ++i) {
                    xs[i] = &S.x[pair_token[p0 + i]];
                    xs16[i] = &S.x16[pair_token[p0 + i]];
                }
                if (L.gu == GateUp::Q8_0) {
                    const auto * g = reinterpret_cast<const BlockQ8_0 *>(gp);
                    const auto * uu = reinterpret_cast<const BlockQ8_0 *>(up);
                    switch (n) {
                    case 1: gu_q8p<1>(g, uu, xs16, og, ou); break;
                    case 2: gu_q8p<2>(g, uu, xs16, og, ou); break;
                    case 3: gu_q8p<3>(g, uu, xs16, og, ou); break;
                    default: gu_q8p<4>(g, uu, xs16, og, ou); break;
                    }
                } else if (precise && L.gu == GateUp::Q4L) {
                    const auto * g = reinterpret_cast<const BlockQ4L *>(gp);
                    const auto * uu = reinterpret_cast<const BlockQ4L *>(up);
                    switch (n) {
                    case 1: gu_p<1, false>(g, uu, xs16, og, ou); break;
                    case 2: gu_p<2, false>(g, uu, xs16, og, ou); break;
                    case 3: gu_p<3, false>(g, uu, xs16, og, ou); break;
                    default: gu_p<4, false>(g, uu, xs16, og, ou); break;
                    }
                } else if (precise) {
                    const auto * g = reinterpret_cast<const BlockQ4X *>(gp);
                    const auto * uu = reinterpret_cast<const BlockQ4X *>(up);
                    switch (n) {
                    case 1: gu_p<1, true>(g, uu, xs16, og, ou); break;
                    case 2: gu_p<2, true>(g, uu, xs16, og, ou); break;
                    case 3: gu_p<3, true>(g, uu, xs16, og, ou); break;
                    default: gu_p<4, true>(g, uu, xs16, og, ou); break;
                    }
                } else if (L.gu == GateUp::Q4L) {
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
        alignas(64) float h[kFF];
        for (int p = th; p < n_pairs; p += nt) {
            for (int j = 0; j < kFF; j += 16) _mm512_store_ps(h + j, swiglu16(_mm512_load_ps(S.g[p] + j), _mm512_load_ps(S.u[p] + j)));
            if (precise) quantize_h16_lanes(h, S.h16[p]);
            else quantize_h(h, S.h[p]);
        }
    });

    // Down: each thread owns a slice of output rows and runs every pair against it.
    pool_.run([&](int th) {
        const int r0 = kEmbd * th / nt, r1 = kEmbd * (th + 1) / nt;
        for (int r = r0; r < r1; ++r) {
            float acc[kMaxTokens] = {};
            for (int u = 0; u < n_experts; ++u) {
                const std::uint8_t * row = L.down + std::size_t(experts[u]) * L.stride + std::size_t(r) * L.dn_row_bytes;
                for (int p0 = pair_start[u]; p0 < pair_start[u + 1]; p0 += 4) {
                    const int n = std::min(4, pair_start[u + 1] - p0);
                    const HAct * hs[4];
                    const HAct16 * hs16[4];
                    float d[4];
                    for (int i = 0; i < n; ++i) {
                        hs[i] = &S.h[p0 + i];
                        hs16[i] = &S.h16[p0 + i];
                    }
                    if (precise && L.dn == Down::IQ4L) {
                        switch (n) {
                        case 1: dn_p<1, false>(row, hs16, d); break;
                        case 2: dn_p<2, false>(row, hs16, d); break;
                        case 3: dn_p<3, false>(row, hs16, d); break;
                        default: dn_p<4, false>(row, hs16, d); break;
                        }
                    } else if (precise) {
                        switch (n) {
                        case 1: dn_p<1, true>(row, hs16, d); break;
                        case 2: dn_p<2, true>(row, hs16, d); break;
                        case 3: dn_p<3, true>(row, hs16, d); break;
                        default: dn_p<4, true>(row, hs16, d); break;
                        }
                    } else if (L.dn == Down::IQ4L) {
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

// ---------------------------------------------------------------------------------------------------------------
// Batched mode (prefill).
//
// Weight tiles: a group of 16 rows is decoded into "row-interleaved" vectors. Each 64-byte vector holds 4
// consecutive weights (one dword) of each of the 16 rows, as unsigned bytes (value + offset), in natural column
// order. One vpdpbusd then multiplies 16 rows by 4 activations of one token broadcast to every lane, so a register
// tile of 2 row vectors x 6 tokens costs 2 vector loads and 6 broadcast loads per 12 vpdpbusd, and per-row scales
// stay in their lanes. Each 32-column sub-block is summed exactly in int32 starting from -offset x sum(activations),
// as run() does, so the gate/up results equal run()'s bit for bit; down differs only in float summation order.
// With 16-bit activations (precise mode) each vector holds an int16 pair of columns of the 16 rows and vpdpwssd does
// the multiplication; a row's sum over a 32-column sub-block is exact and is scaled in float.
//
// Phase 1 (gate/up): a task is (expert, 32 rows of gate and the same 32 rows of up), dynamically scheduled. It
// decodes its 64 rows (86 KB from RAM, 160 KB decoded, L2) and runs every token of the expert block by block, so a
// block's 16 KB of weights stay in L1 while the tokens stream past. The epilogue applies SwiGLU and quantizes the 32 h
// values, which are exactly one of h's quantization blocks.
// Phase 2 (down): each thread owns a fixed slice of 160 output rows, walks every expert in the same order, decodes
// its slice of the expert and adds every pair's contribution; no two threads write the same output, and all threads
// read each expert's h at about the same time, mostly from the L3.
// Both phases software-prefetch the weights they decode next while their kernels run, and visit experts alternately
// from the large and the small end of the size-sorted list, so that RAM keeps streaming while the cores compute.

namespace {

constexpr int kPairChunk = 256;                    // pairs per pass of a phase-1 task: bounds its float accumulators
constexpr std::size_t kGuW = 10 * 8 * 8 * 4 * 64;  // decoded gate/up weights of a task: [block][sub][quad][rv][64]
constexpr std::size_t kGuS = 10 * 8 * 4 * 16;      // int32 sub-block scales: [block][sub][rv][16]
constexpr std::size_t kGuD = 10 * 4 * 16;          // float block scales: [block][rv][16]
constexpr std::size_t kGuScratch = kGuW + 4 * kGuS + 4 * kGuD + 64 * 4 * kPairChunk;
constexpr std::size_t kDnW = 20 * 8 * 2 * 64;      // decoded down weights of 32 rows: [sub][quad][2][64]
constexpr std::size_t kDnD = 20 * 2 * 16;          // float scales of 32 rows: [sub][2][16]
constexpr int kDnGroups = CpuExperts::kEmbd / 32;  // 32-row groups of the down output
// Precise mode: each vector holds 2 int16 weights of 16 rows (a column pair), so tiles are twice as large.
constexpr std::size_t kGuW16 = 10 * 8 * 16 * 4 * 64;  // [block][sub][pair][rv][64]
constexpr std::size_t kGuSD = 10 * 8 * 4 * 16;        // float sub-block scale x block scale: [block][sub][rv][16]
constexpr std::size_t kGuScratch16 = kGuW16 + 4 * kGuSD + 64 * 4 * kPairChunk;
constexpr std::size_t kDnW16 = 20 * 16 * 2 * 64;      // [sub][pair][2][64]

}  // namespace

// Gate/up input of one token: as XAct, with the weight offset correction precomputed per 32 activations
// (-16 x sum for Q4L, whose weights are stored +16; -128 x sum for Q4X, stored +128).
struct alignas(64) XActB {
    std::int8_t q[2560];
    float d[10];
    std::int32_t off[80];
};

struct CpuExperts::BatchScratch {
    int cap_tokens = 0;
    std::size_t cap_pairs = 0;
    Buffer x;              // XActB (precise: XAct16) [cap_tokens]
    Buffer hq, hd, hoff;   // down inputs by sub-block, then pair: int8 (int16) [20][cap_pairs][32], float and int32 [20][cap_pairs]
    std::vector<std::unique_ptr<Buffer>> thread;
    std::vector<int> count, pos, start, experts, pair_token;
    std::vector<float> pair_weight;
    std::vector<std::uint8_t> used;
};

namespace {

// 32 bytes from each of 16 rows (p + r * stride) -> v[i], whose lane r is bytes 4i..4i+3 of row r. A 16 x 8 dword
// transpose as two 8 x 8 transposes side by side; rows 0-3 | 4-7 and 8-11 | 12-15 share vectors, so that the last
// step can collect the 128-bit pieces in row order.
inline void transpose_16x32(const std::uint8_t * p, std::size_t stride, __m512i v[8]) {
    __m512i z[8], t[8], u[8];
    for (int i = 0; i < 4; ++i) {
        z[i] = load_2x256(p + i * stride, p + (4 + i) * stride);
        z[4 + i] = load_2x256(p + (8 + i) * stride, p + (12 + i) * stride);
    }
    for (int h = 0; h < 8; h += 4) {
        t[h + 0] = _mm512_unpacklo_epi32(z[h], z[h + 1]);
        t[h + 1] = _mm512_unpackhi_epi32(z[h], z[h + 1]);
        t[h + 2] = _mm512_unpacklo_epi32(z[h + 2], z[h + 3]);
        t[h + 3] = _mm512_unpackhi_epi32(z[h + 2], z[h + 3]);
        u[h + 0] = _mm512_unpacklo_epi64(t[h], t[h + 2]);          // dwords 0 | 4 of 4 rows
        u[h + 1] = _mm512_unpackhi_epi64(t[h], t[h + 2]);          // 1 | 5
        u[h + 2] = _mm512_unpacklo_epi64(t[h + 1], t[h + 3]);      // 2 | 6
        u[h + 3] = _mm512_unpackhi_epi64(t[h + 1], t[h + 3]);      // 3 | 7
    }
    for (int i = 0; i < 4; ++i) {
        v[i] = _mm512_shuffle_i32x4(u[i], u[4 + i], 0x88);
        v[4 + i] = _mm512_shuffle_i32x4(u[i], u[4 + i], 0xDD);
    }
}

inline __m512i row_offsets(std::size_t row_bytes) {
    return _mm512_mullo_epi32(_mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), _mm512_set1_epi32(int(row_bytes)));
}

// One int32 from memory to all lanes. Written this way MSVC folds it into the consumer as an embedded broadcast (a load);
// _mm512_set1_epi32(*p) goes through a general register and a vector shuffle, which competes with vpdpbusd.
inline __m512i bcast32(const std::int32_t * p) { return _mm512_broadcastd_epi32(_mm_loadu_si32(p)); }

// fp16 in the low half of each dword -> float.
inline __m512 half16_to_float(__m512i v) { return _mm512_cvtph_ps(_mm512_cvtepi32_epi16(v)); }

// 16 gate or up rows (base + r * row_bytes, Q4L or Q4X) -> row vector rv of a phase-1 tile.
void decode_gu16(GateUp fmt, const std::uint8_t * base, std::size_t row_bytes, std::uint8_t * W, std::int32_t * S, float * D, int rv) {
    const bool q4l = fmt == GateUp::Q4L;
    const int bb = q4l ? int(sizeof(BlockQ4L)) : int(sizeof(BlockQ4X)), qo = q4l ? 6 : 10;
    const __m512i lut = q4l ? lut_q4l() : lut_iq4(), m = _mm512_set1_epi8(0x0F), rows = row_offsets(row_bytes);
    for (int b = 0; b < 10; ++b) {
        for (int c = 0; c < 4; ++c) {
            __m512i v[8];
            transpose_16x32(base + b * bb + qo + 32 * c, row_bytes, v);
            // low nibbles are sub-block 2c, high nibbles 2c+1; quads are 4 vectors apart
            std::uint8_t * lo = W + ((b * 8 + 2 * c) * 32 + rv) * 64;
            std::uint8_t * hi = lo + 32 * 64;
            for (int i = 0; i < 8; ++i) {
                _mm512_store_si512(lo + 256 * i, _mm512_shuffle_epi8(lut, _mm512_and_si512(v[i], m)));
                _mm512_store_si512(hi + 256 * i, _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(v[i], 4), m)));
            }
        }
        const __m512i at = _mm512_add_epi32(rows, _mm512_set1_epi32(b * bb));
        _mm512_store_ps(D + (b * 4 + rv) * 16, half16_to_float(_mm512_i32gather_epi32(at, base, 1)));
        std::int32_t * sb = S + (b * 32 + rv) * 16;
        if (q4l) {
            // 8 nibbles ls -> 2 ls + 1
            const __m512i sc = _mm512_i32gather_epi32(_mm512_add_epi32(at, _mm512_set1_epi32(2)), base, 1);
            for (int s = 0; s < 8; ++s)
                _mm512_store_si512(sb + 64 * s, _mm512_add_epi32(_mm512_slli_epi32(_mm512_and_si512(_mm512_srlv_epi32(sc, _mm512_set1_epi32(4 * s)), _mm512_set1_epi32(15)), 1),
                                                                  _mm512_set1_epi32(1)));
        } else {
            // 8 signed bytes
            const __m512i s0 = _mm512_i32gather_epi32(_mm512_add_epi32(at, _mm512_set1_epi32(2)), base, 1);
            const __m512i s1 = _mm512_i32gather_epi32(_mm512_add_epi32(at, _mm512_set1_epi32(6)), base, 1);
            for (int s = 0; s < 4; ++s) {
                _mm512_store_si512(sb + 64 * s, _mm512_srai_epi32(_mm512_sllv_epi32(s0, _mm512_set1_epi32(24 - 8 * s)), 24));
                _mm512_store_si512(sb + 64 * (4 + s), _mm512_srai_epi32(_mm512_sllv_epi32(s1, _mm512_set1_epi32(24 - 8 * s)), 24));
            }
        }
    }
}

// 16 down rows (base + r * row_bytes, IQ4L or Q8_0) -> half h (0, 1) of a 32-row phase-2 tile.
void decode_dn16(Down fmt, const std::uint8_t * base, std::size_t row_bytes, std::uint8_t * W, float * DW, int h) {
    const __m512i rows = row_offsets(row_bytes);
    if (fmt == Down::IQ4L) {
        const __m512i lut = lut_iq4(), m = _mm512_set1_epi8(0x0F);
        for (int c = 0; c < 10; ++c) {
            __m512i v[8];
            transpose_16x32(base + 40 + 32 * c, row_bytes, v);
            std::uint8_t * lo = W + (32 * c + h) * 64;  // sub-block 2c
            std::uint8_t * hi = lo + 16 * 64;            // 2c+1
            for (int i = 0; i < 8; ++i) {
                _mm512_store_si512(lo + 128 * i, _mm512_shuffle_epi8(lut, _mm512_and_si512(v[i], m)));
                _mm512_store_si512(hi + 128 * i, _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(v[i], 4), m)));
            }
        }
        for (int k = 0; k < 10; ++k) {
            const __m512i d = _mm512_i32gather_epi32(_mm512_add_epi32(rows, _mm512_set1_epi32(4 * k)), base, 1);
            _mm512_store_ps(DW + (4 * k + h) * 16, half16_to_float(d));
            _mm512_store_ps(DW + (4 * k + 2 + h) * 16, half16_to_float(_mm512_srli_epi32(d, 16)));
        }
    } else {
        const __m512i flip = _mm512_set1_epi8(char(0x80));  // int8 -> uint8 + 128
        for (int s = 0; s < 20; ++s) {
            __m512i v[8];
            transpose_16x32(base + 34 * s + 2, row_bytes, v);
            std::uint8_t * dst = W + (16 * s + h) * 64;
            for (int i = 0; i < 8; ++i) _mm512_store_si512(dst + 128 * i, _mm512_xor_si512(v[i], flip));
            _mm512_store_ps(DW + (2 * s + h) * 16,
                            half16_to_float(_mm512_i32gather_epi32(_mm512_add_epi32(rows, _mm512_set1_epi32(34 * s)), base, 1)));
        }
    }
}

// Phase-1 micro-kernel: 2 row vectors (32 rows) x NT tokens over one 256-column block b. W, S, D point at the first
// row vector of the block; F[t] at the token's float accumulators for those rows. Per 32 columns: exact int32 sums,
// times the sub-block scale; per block: float(sum) * d(row) * d(token) added to F, in run()'s order. (Written as one
// loop over tokens with the row vectors spelled out: MSVC unrolls that and keeps the arrays in registers, but not
// nested loops.)
template <int NT>
void gu_kernel(const std::uint8_t * W, const std::int32_t * S, const float * D, const XActB * const * xs, int b, float * const * F) {
    __m512i blk0[NT], blk1[NT];
    for (int t = 0; t < NT; ++t) blk0[t] = blk1[t] = _mm512_setzero_si512();
    for (int s = 0; s < 8; ++s) {
        __m512i acc0[NT], acc1[NT];
        for (int t = 0; t < NT; ++t) acc0[t] = acc1[t] = bcast32(xs[t]->off + 8 * b + s);
        const std::uint8_t * ws = W + 2048 * s;
        const int xo = 64 * b + 8 * s;
        for (int q = 0; q < 8; ++q) {
            const __m512i w0 = _mm512_load_si512(ws + 256 * q), w1 = _mm512_load_si512(ws + 256 * q + 64);
            for (int t = 0; t < NT; ++t) {
                const __m512i a = bcast32(reinterpret_cast<const std::int32_t *>(xs[t]->q) + xo + q);
                acc0[t] = _mm512_dpbusd_epi32(acc0[t], w0, a);
                acc1[t] = _mm512_dpbusd_epi32(acc1[t], w1, a);
            }
        }
        const __m512i sc0 = _mm512_load_si512(S + 64 * s), sc1 = _mm512_load_si512(S + 64 * s + 16);
        for (int t = 0; t < NT; ++t) {
            blk0[t] = _mm512_add_epi32(blk0[t], _mm512_mullo_epi32(acc0[t], sc0));
            blk1[t] = _mm512_add_epi32(blk1[t], _mm512_mullo_epi32(acc1[t], sc1));
        }
    }
    const __m512 d0 = _mm512_load_ps(D), d1 = _mm512_load_ps(D + 16);
    for (int t = 0; t < NT; ++t) {
        const __m512 xd = _mm512_set1_ps(xs[t]->d[b]);
        float * f = F[t];
        _mm512_store_ps(f, _mm512_add_ps(_mm512_load_ps(f), _mm512_mul_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(blk0[t]), d0), xd)));
        _mm512_store_ps(f + 16, _mm512_add_ps(_mm512_load_ps(f + 16), _mm512_mul_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(blk1[t]), d1), xd)));
    }
}

// Both halves (gate rows, up rows) of a phase-1 tile for NT tokens: 2 row vectors at a time keeps 2 x 2 x NT
// accumulators in registers.
template <int NT>
void gu_tile(const std::uint8_t * W, const std::int32_t * S, const float * D, const XActB * const * xs, int b, float * const * F) {
    float * F2[NT];
    for (int t = 0; t < NT; ++t) F2[t] = F[t] + 32;
    gu_kernel<NT>(W, S, D, xs, b, F);
    gu_kernel<NT>(W + 128, S + 32, D + 32, xs, b, F2);
}

constexpr int kNtGU = 6, kNtDN = 6;

void gu_tile_n(int n, const std::uint8_t * W, const std::int32_t * S, const float * D, const XActB * const * xs, int b, float * const * F) {
    switch (n) {
    case 1: gu_tile<1>(W, S, D, xs, b, F); break;
    case 2: gu_tile<2>(W, S, D, xs, b, F); break;
    case 3: gu_tile<3>(W, S, D, xs, b, F); break;
    case 4: gu_tile<4>(W, S, D, xs, b, F); break;
    case 5: gu_tile<5>(W, S, D, xs, b, F); break;
    default: gu_tile<6>(W, S, D, xs, b, F); break;
    }
}

// Down inputs of the pairs, stored by sub-block so that the phase-1 tasks of one expert write disjoint ranges.
struct HView {
    const std::int8_t * q;
    const float * d;
    const std::int32_t * off;
    std::size_t cap;
};

// Phase-2 micro-kernel: 32 rows (2 row vectors) x NT pairs over all 640 columns; o[t][0..31] = in[t][0..31] + w * result.
// in[t] is o[t], or zeros for a token's first contribution, which saves zeroing the output beforehand.
template <int NT>
void dn_kernel(const std::uint8_t * W, const float * DW, const HView & H, const int * pair, const float * const * in, float * const * o,
               const float * pw) {
    __m512 f0[NT], f1[NT];
    for (int t = 0; t < NT; ++t) f0[t] = f1[t] = _mm512_setzero_ps();
    for (int s = 0; s < 20; ++s) {
        const std::size_t so = std::size_t(s) * H.cap;
        const std::int32_t * hq = reinterpret_cast<const std::int32_t *>(H.q) + 8 * so;
        __m512i acc0[NT], acc1[NT];
        for (int t = 0; t < NT; ++t) acc0[t] = acc1[t] = bcast32(H.off + so + pair[t]);
        const std::uint8_t * ws = W + 1024 * s;
        for (int q = 0; q < 8; ++q) {
            const __m512i w0 = _mm512_load_si512(ws + 128 * q), w1 = _mm512_load_si512(ws + 128 * q + 64);
            for (int t = 0; t < NT; ++t) {
                const __m512i a = bcast32(hq + 8 * pair[t] + q);
                acc0[t] = _mm512_dpbusd_epi32(acc0[t], w0, a);
                acc1[t] = _mm512_dpbusd_epi32(acc1[t], w1, a);
            }
        }
        const __m512 dw0 = _mm512_load_ps(DW + 32 * s), dw1 = _mm512_load_ps(DW + 32 * s + 16);
        for (int t = 0; t < NT; ++t) {
            const __m512 hd = _mm512_set1_ps(H.d[so + pair[t]]);
            f0[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc0[t]), _mm512_mul_ps(dw0, hd), f0[t]);
            f1[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc1[t]), _mm512_mul_ps(dw1, hd), f1[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m512 w = _mm512_set1_ps(pw[t]);
        _mm512_storeu_ps(o[t], _mm512_fmadd_ps(f0[t], w, _mm512_loadu_ps(in[t])));
        _mm512_storeu_ps(o[t] + 16, _mm512_fmadd_ps(f1[t], w, _mm512_loadu_ps(in[t] + 16)));
    }
}

void dn_kernel_n(int n, const std::uint8_t * W, const float * DW, const HView & H, const int * pair, const float * const * in,
                 float * const * o, const float * pw) {
    switch (n) {
    case 1: dn_kernel<1>(W, DW, H, pair, in, o, pw); break;
    case 2: dn_kernel<2>(W, DW, H, pair, in, o, pw); break;
    case 3: dn_kernel<3>(W, DW, H, pair, in, o, pw); break;
    case 4: dn_kernel<4>(W, DW, H, pair, in, o, pw); break;
    case 5: dn_kernel<5>(W, DW, H, pair, in, o, pw); break;
    default: dn_kernel<6>(W, DW, H, pair, in, o, pw); break;
    }
}

// ---- Precise mode (16-bit activations) ----

// Lane r of v holds 4 int8 weights of row r -> int16 pairs (0, 1) and (2, 3) of each row, the operand layout of
// vpdpwssd: each byte goes to the high byte of a word, and an arithmetic shift sign-extends it.
inline void pairs_i16(__m512i v, __m512i & p01, __m512i & p23) {
    const __m512i s01 = _mm512_broadcast_i32x4(_mm_setr_epi32(0x01800080, 0x05800480, 0x09800880, 0x0D800C80));
    const __m512i s23 = _mm512_broadcast_i32x4(_mm_setr_epi32(0x03800280, 0x07800680, 0x0B800A80, 0x0F800E80));
    p01 = _mm512_srai_epi16(_mm512_shuffle_epi8(v, s01), 8);
    p23 = _mm512_srai_epi16(_mm512_shuffle_epi8(v, s23), 8);
}

// 16 gate or up rows -> row vector rv of a precise phase-1 tile: signed int16 weights by column pair, and per
// sub-block the float product of the sub-block scale and the block scale.
void decode_gu16_p(GateUp fmt, const std::uint8_t * base, std::size_t row_bytes, std::uint8_t * W, float * SD, int rv) {
    const bool q4l = fmt == GateUp::Q4L;
    const int bb = q4l ? int(sizeof(BlockQ4L)) : int(sizeof(BlockQ4X)), qo = q4l ? 6 : 10;
    const __m512i lut = q4l ? lut_q4l_signed() : lut_iq4_signed(), m = _mm512_set1_epi8(0x0F), rows = row_offsets(row_bytes);
    for (int b = 0; b < 10; ++b) {
        for (int c = 0; c < 4; ++c) {
            __m512i v[8];
            transpose_16x32(base + b * bb + qo + 32 * c, row_bytes, v);
            // sub-block 2c from the low nibbles, 2c+1 from the high ones; columns 4i..4i+3 are pairs 2i and 2i+1
            std::uint8_t * lo = W + (((b * 8 + 2 * c) * 16) * 4 + rv) * 64;
            std::uint8_t * hi = lo + 16 * 4 * 64;
            for (int i = 0; i < 8; ++i) {
                __m512i a, b2;
                pairs_i16(_mm512_shuffle_epi8(lut, _mm512_and_si512(v[i], m)), a, b2);
                _mm512_store_si512(lo + 512 * i, a);
                _mm512_store_si512(lo + 512 * i + 256, b2);
                pairs_i16(_mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(v[i], 4), m)), a, b2);
                _mm512_store_si512(hi + 512 * i, a);
                _mm512_store_si512(hi + 512 * i + 256, b2);
            }
        }
        const __m512i at = _mm512_add_epi32(rows, _mm512_set1_epi32(b * bb));
        const __m512 d = half16_to_float(_mm512_i32gather_epi32(at, base, 1));
        float * sd = SD + (b * 32 + rv) * 16;
        if (q4l) {
            const __m512i sc = _mm512_i32gather_epi32(_mm512_add_epi32(at, _mm512_set1_epi32(2)), base, 1);
            for (int s = 0; s < 8; ++s) {
                const __m512i ls = _mm512_and_si512(_mm512_srlv_epi32(sc, _mm512_set1_epi32(4 * s)), _mm512_set1_epi32(15));
                _mm512_store_ps(sd + 64 * s, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_slli_epi32(ls, 1), _mm512_set1_epi32(1))), d));
            }
        } else {
            const __m512i s0 = _mm512_i32gather_epi32(_mm512_add_epi32(at, _mm512_set1_epi32(2)), base, 1);
            const __m512i s1 = _mm512_i32gather_epi32(_mm512_add_epi32(at, _mm512_set1_epi32(6)), base, 1);
            for (int s = 0; s < 4; ++s) {
                _mm512_store_ps(sd + 64 * s, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_srai_epi32(_mm512_sllv_epi32(s0, _mm512_set1_epi32(24 - 8 * s)), 24)), d));
                _mm512_store_ps(sd + 64 * (4 + s), _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_srai_epi32(_mm512_sllv_epi32(s1, _mm512_set1_epi32(24 - 8 * s)), 24)), d));
            }
        }
    }
}

// 16 down rows -> half h of a precise 32-row phase-2 tile.
void decode_dn16_p(Down fmt, const std::uint8_t * base, std::size_t row_bytes, std::uint8_t * W, float * DW, int h) {
    const __m512i rows = row_offsets(row_bytes);
    if (fmt == Down::IQ4L) {
        const __m512i lut = lut_iq4_signed(), m = _mm512_set1_epi8(0x0F);
        for (int c = 0; c < 10; ++c) {
            __m512i v[8];
            transpose_16x32(base + 40 + 32 * c, row_bytes, v);
            std::uint8_t * lo = W + (2 * c * 16 * 2 + h) * 64;  // sub-block 2c
            std::uint8_t * hi = lo + 16 * 2 * 64;               // 2c+1
            for (int i = 0; i < 8; ++i) {
                __m512i a, b;
                pairs_i16(_mm512_shuffle_epi8(lut, _mm512_and_si512(v[i], m)), a, b);
                _mm512_store_si512(lo + 256 * i, a);
                _mm512_store_si512(lo + 256 * i + 128, b);
                pairs_i16(_mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(v[i], 4), m)), a, b);
                _mm512_store_si512(hi + 256 * i, a);
                _mm512_store_si512(hi + 256 * i + 128, b);
            }
        }
        for (int k = 0; k < 10; ++k) {
            const __m512i d = _mm512_i32gather_epi32(_mm512_add_epi32(rows, _mm512_set1_epi32(4 * k)), base, 1);
            _mm512_store_ps(DW + (4 * k + h) * 16, half16_to_float(d));
            _mm512_store_ps(DW + (4 * k + 2 + h) * 16, half16_to_float(_mm512_srli_epi32(d, 16)));
        }
    } else {
        for (int s = 0; s < 20; ++s) {
            __m512i v[8];
            transpose_16x32(base + 34 * s + 2, row_bytes, v);
            std::uint8_t * dst = W + (s * 16 * 2 + h) * 64;
            for (int i = 0; i < 8; ++i) {
                __m512i a, b;
                pairs_i16(v[i], a, b);
                _mm512_store_si512(dst + 256 * i, a);
                _mm512_store_si512(dst + 256 * i + 128, b);
            }
            _mm512_store_ps(DW + (2 * s + h) * 16,
                            half16_to_float(_mm512_i32gather_epi32(_mm512_add_epi32(rows, _mm512_set1_epi32(34 * s)), base, 1)));
        }
    }
}

// Precise phase-1 micro-kernel: 2 row vectors x NT tokens over block b, vpdpwssd on column pairs. A row's sum over
// one 32-column sub-block is exact in int32 (at most 32 x 127 x 32767); it is scaled in float, since a whole block
// could overflow.
template <int NT>
void gu_kernel_p(const std::uint8_t * W, const float * SD, const XAct16 * const * xs, int b, float * const * F) {
    __m512 f0[NT], f1[NT];
    for (int t = 0; t < NT; ++t) f0[t] = f1[t] = _mm512_setzero_ps();
    for (int s = 0; s < 8; ++s) {
        __m512i acc0[NT], acc1[NT];
        for (int t = 0; t < NT; ++t) acc0[t] = acc1[t] = _mm512_setzero_si512();
        const std::uint8_t * ws = W + 4096 * s;
        const int xo = 128 * b + 16 * s;
        for (int p = 0; p < 16; ++p) {
            const __m512i w0 = _mm512_load_si512(ws + 256 * p), w1 = _mm512_load_si512(ws + 256 * p + 64);
            for (int t = 0; t < NT; ++t) {
                const __m512i a = bcast32(reinterpret_cast<const std::int32_t *>(xs[t]->q) + xo + p);
                acc0[t] = _mm512_dpwssd_epi32(acc0[t], w0, a);
                acc1[t] = _mm512_dpwssd_epi32(acc1[t], w1, a);
            }
        }
        const __m512 sd0 = _mm512_load_ps(SD + 64 * s), sd1 = _mm512_load_ps(SD + 64 * s + 16);
        for (int t = 0; t < NT; ++t) {
            f0[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc0[t]), sd0, f0[t]);
            f1[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc1[t]), sd1, f1[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m512 xd = _mm512_set1_ps(xs[t]->d[b]);
        _mm512_store_ps(F[t], _mm512_fmadd_ps(f0[t], xd, _mm512_load_ps(F[t])));
        _mm512_store_ps(F[t] + 16, _mm512_fmadd_ps(f1[t], xd, _mm512_load_ps(F[t] + 16)));
    }
}

template <int NT>
void gu_tile_p(const std::uint8_t * W, const float * SD, const XAct16 * const * xs, int b, float * const * F) {
    float * F2[NT];
    for (int t = 0; t < NT; ++t) F2[t] = F[t] + 32;
    gu_kernel_p<NT>(W, SD, xs, b, F);
    gu_kernel_p<NT>(W + 128, SD + 32, xs, b, F2);
}

void gu_tile_p_n(int n, const std::uint8_t * W, const float * SD, const XAct16 * const * xs, int b, float * const * F) {
    switch (n) {
    case 1: gu_tile_p<1>(W, SD, xs, b, F); break;
    case 2: gu_tile_p<2>(W, SD, xs, b, F); break;
    case 3: gu_tile_p<3>(W, SD, xs, b, F); break;
    case 4: gu_tile_p<4>(W, SD, xs, b, F); break;
    case 5: gu_tile_p<5>(W, SD, xs, b, F); break;
    default: gu_tile_p<6>(W, SD, xs, b, F); break;
    }
}

struct HView16 {
    const std::int16_t * q;
    const float * d;
    std::size_t cap;
};

// Precise phase-2 micro-kernel: as dn_kernel, with vpdpwssd on column pairs.
template <int NT>
void dn_kernel_p(const std::uint8_t * W, const float * DW, const HView16 & H, const int * pair, const float * const * in,
                 float * const * o, const float * pw) {
    __m512 f0[NT], f1[NT];
    for (int t = 0; t < NT; ++t) f0[t] = f1[t] = _mm512_setzero_ps();
    for (int s = 0; s < 20; ++s) {
        const std::size_t so = std::size_t(s) * H.cap;
        const std::int32_t * hq = reinterpret_cast<const std::int32_t *>(H.q) + 16 * so;
        __m512i acc0[NT], acc1[NT];
        for (int t = 0; t < NT; ++t) acc0[t] = acc1[t] = _mm512_setzero_si512();
        const std::uint8_t * ws = W + 2048 * s;
        for (int p = 0; p < 16; ++p) {
            const __m512i w0 = _mm512_load_si512(ws + 128 * p), w1 = _mm512_load_si512(ws + 128 * p + 64);
            for (int t = 0; t < NT; ++t) {
                const __m512i a = bcast32(hq + 16 * pair[t] + p);
                acc0[t] = _mm512_dpwssd_epi32(acc0[t], w0, a);
                acc1[t] = _mm512_dpwssd_epi32(acc1[t], w1, a);
            }
        }
        const __m512 dw0 = _mm512_load_ps(DW + 32 * s), dw1 = _mm512_load_ps(DW + 32 * s + 16);
        for (int t = 0; t < NT; ++t) {
            const __m512 hd = _mm512_set1_ps(H.d[so + pair[t]]);
            f0[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc0[t]), _mm512_mul_ps(dw0, hd), f0[t]);
            f1[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc1[t]), _mm512_mul_ps(dw1, hd), f1[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        const __m512 w = _mm512_set1_ps(pw[t]);
        _mm512_storeu_ps(o[t], _mm512_fmadd_ps(f0[t], w, _mm512_loadu_ps(in[t])));
        _mm512_storeu_ps(o[t] + 16, _mm512_fmadd_ps(f1[t], w, _mm512_loadu_ps(in[t] + 16)));
    }
}

void dn_kernel_p_n(int n, const std::uint8_t * W, const float * DW, const HView16 & H, const int * pair, const float * const * in,
                   float * const * o, const float * pw) {
    switch (n) {
    case 1: dn_kernel_p<1>(W, DW, H, pair, in, o, pw); break;
    case 2: dn_kernel_p<2>(W, DW, H, pair, in, o, pw); break;
    case 3: dn_kernel_p<3>(W, DW, H, pair, in, o, pw); break;
    case 4: dn_kernel_p<4>(W, DW, H, pair, in, o, pw); break;
    case 5: dn_kernel_p<5>(W, DW, H, pair, in, o, pw); break;
    default: dn_kernel_p<6>(W, DW, H, pair, in, o, pw); break;
    }
}

// Software prefetch of the weights a thread decodes next, spread over the kernel calls of its current work so that
// RAM streams while the cores compute (decode itself would otherwise wait for every line, and all threads would
// alternate between using only RAM and using only the cores).
class Prefetcher {
public:
    void start(const std::uint8_t * a, std::size_t na, const std::uint8_t * b, std::size_t nb, int steps) {
        p_[0] = a;
        p_[1] = b;
        n_[0] = na;
        n_[1] = nb;
        k_ = 0;
        at_ = 0;
        per_step_ = ((na + nb) / 64 + std::size_t(steps) - 1) / std::size_t(std::max(steps, 1)) * 64;
    }
    void step() { issue(per_step_); }
    void finish() { issue(~std::size_t(0) / 2); }

private:
    void issue(std::size_t bytes) {
        while (bytes > 0 && k_ < 2) {
            if (at_ >= n_[k_]) {
                ++k_;
                at_ = 0;
                continue;
            }
            _mm_prefetch(reinterpret_cast<const char *>(p_[k_] + at_), _MM_HINT_T1);
            at_ += 64;
            bytes = bytes > 64 ? bytes - 64 : 0;
        }
    }
    const std::uint8_t * p_[2] = {};
    std::size_t n_[2] = {}, at_ = 0, per_step_ = 0;
    int k_ = 2;
};

}  // namespace

void CpuExperts::run_batch(int layer, int n_tokens, const float * x, const std::int32_t * ids, const float * weights,
                           const std::uint8_t * on_cpu, float * out) {
    if (has_layer(layer) && layers_[std::size_t(layer)]->gu == GateUp::Q8_0)
        throw std::runtime_error("CpuExperts: run_batch does not support Q8_0 gate/up (layer " + std::to_string(layer) + ")");
    if (precise_) run_batch_impl<true>(layer, n_tokens, x, ids, weights, on_cpu, out);
    else run_batch_impl<false>(layer, n_tokens, x, ids, weights, on_cpu, out);
}

template <bool P>
void CpuExperts::run_batch_impl(int layer, int n_tokens, const float * x, const std::int32_t * ids, const float * weights,
                                const std::uint8_t * on_cpu, float * out) {
    using XA = std::conditional_t<P, XAct16, XActB>;
    using HQ = std::conditional_t<P, std::int16_t, std::int8_t>;
    constexpr std::size_t kGuTile = P ? kGuW16 : kGuW, kDnTile = P ? kDnW16 : kDnW;
    if (!has_layer(layer)) throw std::runtime_error("CpuExperts: layer " + std::to_string(layer) + " is not loaded");
    if (n_tokens < 1 || n_tokens > kMaxBatchTokens) throw std::runtime_error("CpuExperts: n_tokens out of range");
    const Layer & L = *layers_[std::size_t(layer)];
    const int nt = pool_.size();
    const AffinityScope pin(caller_affinity_);
    if (!batch_) batch_ = std::make_unique<BatchScratch>();
    BatchScratch & B = *batch_;
    if (B.thread.empty()) {
        const std::size_t dn = std::size_t((kDnGroups + nt - 1) / nt) * (kDnTile + 4 * kDnD) + kMaxBatchTokens;
        for (int t = 0; t < nt; ++t) {
            B.thread.push_back(std::make_unique<Buffer>());
            B.thread.back()->alloc(std::max(P ? kGuScratch16 : kGuScratch, dn));
        }
        B.count.resize(kExperts);
        B.pos.resize(kExperts);
        B.start.resize(kExperts + 1);
        B.experts.resize(kExperts);
    }
    if (n_tokens > B.cap_tokens) {
        B.cap_tokens = n_tokens;
        B.cap_pairs = std::size_t(n_tokens) * kUsed;
        B.x.alloc(sizeof(XA) * std::size_t(n_tokens));
        B.hq.alloc(20 * 32 * sizeof(HQ) * B.cap_pairs);
        B.hd.alloc(20 * 4 * B.cap_pairs);
        if (!P) B.hoff.alloc(20 * 4 * B.cap_pairs);
        B.pair_token.resize(B.cap_pairs);
        B.pair_weight.resize(B.cap_pairs);
        B.used.resize(std::size_t(n_tokens));
    }

    // Group the (token, slot) pairs by expert, tokens ascending within an expert; experts sorted by pair count,
    // largest first (both phases take them from both ends of this list).
    int * count = B.count.data();
    std::fill(count, count + kExperts, 0);
    std::fill(B.used.begin(), B.used.begin() + n_tokens, std::uint8_t(0));
    for (int t = 0; t < n_tokens; ++t)
        for (int k = 0; k < kUsed; ++k) {
            if (on_cpu && !on_cpu[t * kUsed + k]) continue;
            const int e = ids[t * kUsed + k];
            if (e < 0 || e >= kExperts) throw std::runtime_error("CpuExperts: expert id out of range");
            ++count[e];
            B.used[std::size_t(t)] = 1;
        }
    int n_experts = 0;
    for (int e = 0; e < kExperts; ++e)
        if (count[e]) B.experts[std::size_t(n_experts++)] = e;
    std::sort(B.experts.begin(), B.experts.begin() + n_experts, [&](int a, int b) { return count[a] != count[b] ? count[a] > count[b] : a < b; });
    int * pos = B.pos.data();  // next free pair slot of each expert
    int n_pairs = 0;
    for (int u = 0; u < n_experts; ++u) {
        B.start[std::size_t(u)] = n_pairs;
        pos[B.experts[std::size_t(u)]] = n_pairs;
        n_pairs += count[B.experts[std::size_t(u)]];
    }
    B.start[std::size_t(n_experts)] = n_pairs;
    for (int t = 0; t < n_tokens; ++t)
        for (int k = 0; k < kUsed; ++k) {
            if (on_cpu && !on_cpu[t * kUsed + k]) continue;
            const int p = pos[ids[t * kUsed + k]]++;
            B.pair_token[std::size_t(p)] = t;
            B.pair_weight[std::size_t(p)] = weights[t * kUsed + k];
        }
    if (n_pairs == 0) {
        std::memset(out, 0, sizeof(float) * std::size_t(n_tokens) * kEmbd);
        return;
    }

    // Quantize the activations of the tokens that have pairs, exactly as run() does.
    XA * xb = reinterpret_cast<XA *>(B.x.p);
    const int off_scale = L.gu == GateUp::Q4L ? -16 : -128;
    pool_.run([&](int th) {
        for (int t = th; t < n_tokens; t += nt) {
            if (!B.used[std::size_t(t)]) continue;
            XA & a = xb[t];
            if constexpr (P) {
                quantize_x16(x + std::size_t(t) * kEmbd, a);
            } else {
                std::int32_t sums[80];
                for (int b = 0; b < 10; ++b) a.d[b] = quantize_q8<256>(x + std::size_t(t) * kEmbd + 256 * b, a.q + 256 * b, sums + 8 * b);
                for (int k = 0; k < 80; ++k) a.off[k] = off_scale * sums[k];
            }
        }
    });

    // Phase 1: gate/up, SwiGLU and quantization of h.
    const std::size_t cap = B.cap_pairs;
    HQ * hq = reinterpret_cast<HQ *>(B.hq.p);
    float * hd = reinterpret_cast<float *>(B.hd.p);
    std::int32_t * hoff = reinterpret_cast<std::int32_t *>(B.hoff.p);
    const int n_tasks = n_experts * (kFF / 32);
    const std::size_t panel = 32 * L.gu_row_bytes;  // 32 rows of gate (or up): one task's share of a matrix
    // Tasks alternate between the two ends of the size-sorted list (expert-major within each end), so that threads
    // run large (compute-bound) and small (RAM-bound) experts side by side and the last tasks are medium-sized.
    auto slot = [&](int task) { return task % 2 == 0 ? task / 2 : n_tasks - 1 - task / 2; };
    auto panel_at = [&](const std::uint8_t * mat, int task) {
        const int k = slot(task);
        return mat + std::size_t(B.experts[std::size_t(k / (kFF / 32))]) * L.stride + std::size_t(k % (kFF / 32)) * panel;
    };
    std::atomic<int> next_task{0};
    pool_.run([&](int th) {
        // int8: W [kGuW] | S int32 [kGuS] | D [kGuD] | F; precise: W [kGuW16] | SD [kGuSD] | F
        std::uint8_t * W = B.thread[std::size_t(th)]->p;
        std::int32_t * S = reinterpret_cast<std::int32_t *>(W + kGuTile);
        float * D = reinterpret_cast<float *>(W + kGuW + 4 * kGuS);
        float * SD = reinterpret_cast<float *>(W + kGuTile);
        float * F = P ? SD + kGuSD : D + kGuD;
        Prefetcher pf;
        int task = next_task.fetch_add(1, std::memory_order_relaxed);
        while (task < n_tasks) {
            // Claim the next task now: its weights stream in over this task's decode and kernel calls.
            const int next = next_task.fetch_add(1, std::memory_order_relaxed);
            const int u = slot(task) / (kFF / 32), j = slot(task) % (kFF / 32);
            const int pb = B.start[std::size_t(u)], pe = B.start[std::size_t(u) + 1];
            if (next < n_tasks) pf.start(panel_at(L.gate, next), panel, panel_at(L.up, next), panel, 4 + 10 * ((pe - pb + kNtGU - 1) / kNtGU));
            for (int rv = 0; rv < 4; ++rv) {
                pf.step();
                const std::uint8_t * rows = panel_at(rv < 2 ? L.gate : L.up, task) + std::size_t(16 * (rv & 1)) * L.gu_row_bytes;
                if constexpr (P) decode_gu16_p(L.gu, rows, L.gu_row_bytes, W, SD, rv);
                else decode_gu16(L.gu, rows, L.gu_row_bytes, W, S, D, rv);
            }
            for (int p0 = pb; p0 < pe; p0 += kPairChunk) {
                const int n = std::min(kPairChunk, pe - p0);
                std::memset(F, 0, sizeof(float) * 64 * std::size_t(n));
                for (int b = 0; b < 10; ++b)
                    for (int i = 0; i < n; i += kNtGU) {
                        const int m = std::min(kNtGU, n - i);
                        const XA * xs[kNtGU];
                        float * fs[kNtGU];
                        for (int k = 0; k < m; ++k) {
                            xs[k] = xb + B.pair_token[std::size_t(p0 + i + k)];
                            fs[k] = F + 64 * (i + k);
                        }
                        pf.step();
                        if constexpr (P) gu_tile_p_n(m, W + 32768 * b, SD + 512 * b, xs, b, fs);
                        else gu_tile_n(m, W + 16384 * b, S + 512 * b, D + 64 * b, xs, b, fs);
                    }
                // F[i]: gate rows 0-31 then up rows 0-31 of this task
                for (int i = 0; i < n; ++i) {
                    const float * f = F + 64 * i;
                    alignas(64) float h[32];
                    _mm512_store_ps(h, swiglu16(_mm512_load_ps(f), _mm512_load_ps(f + 32)));
                    _mm512_store_ps(h + 16, swiglu16(_mm512_load_ps(f + 16), _mm512_load_ps(f + 48)));
                    const std::size_t at = std::size_t(j) * cap + std::size_t(p0 + i);
                    if constexpr (P) {
                        hd[at] = quantize_q16<32>(h, hq + 32 * at);
                    } else {
                        std::int32_t sum;
                        hd[at] = quantize_q8<32>(h, hq + 32 * at, &sum);
                        hoff[at] = -128 * sum;  // IQ4L and Q8_0 weights are both stored +128
                    }
                }
            }
            if (next < n_tasks) pf.finish();
            task = next;
        }
    });

    // Phase 2: down. Thread th owns the 32-row groups [g0, g1) of every token's output.
    const auto H = [&] {
        if constexpr (P) return HView16{hq, hd, cap};
        else return HView{hq, hd, hoff, cap};
    }();
    alignas(64) static const float kZeros[32] = {};
    pool_.run([&](int th) {
        const int g0 = kDnGroups * th / nt, g1 = kDnGroups * (th + 1) / nt, ng = g1 - g0;
        if (ng == 0) return;
        const int r0 = 32 * g0;
        std::uint8_t * W = B.thread[std::size_t(th)]->p;
        float * DW = reinterpret_cast<float *>(W + kDnTile * std::size_t(ng));
        std::uint8_t * first = reinterpret_cast<std::uint8_t *>(DW + kDnD * std::size_t(ng));  // no contribution stored yet
        std::memset(first, 1, std::size_t(n_tokens));
        const std::size_t slice = std::size_t(32 * ng) * L.dn_row_bytes;
        // Experts alternately from both ends of the size-sorted list, so that a large expert's kernels cover the
        // streaming of the small ones around it. All threads use the same order, so each expert's h is read by all of
        // them at about the same time, mostly from the L3.
        auto rank = [&](int u) { return u % 2 == 0 ? u / 2 : n_experts - 1 - u / 2; };
        auto slice_at = [&](int u) { return L.down + std::size_t(B.experts[std::size_t(rank(u))]) * L.stride + std::size_t(r0) * L.dn_row_bytes; };
        // weights two experts ahead: one expert's kernels can be shorter than the time its successor takes to stream in
        Prefetcher pf;
        pf.start(slice_at(0), slice, n_experts > 1 ? slice_at(1) : nullptr, n_experts > 1 ? slice : 0, 1);
        pf.finish();
        for (int u = 0; u < n_experts; ++u) {
            const std::uint8_t * base = slice_at(u);
            const int pb = B.start[std::size_t(rank(u))], pe = B.start[std::size_t(rank(u)) + 1];
            // expert u + 2 streams in over this expert's decode and kernel calls
            if (u + 2 < n_experts) pf.start(slice_at(u + 2), slice, nullptr, 0, 2 * ng + (pe - pb + kNtDN - 1) / kNtDN);
            for (int g = 0; g < ng; ++g)
                for (int h = 0; h < 2; ++h) {
                    pf.step();
                    const std::uint8_t * rows = base + std::size_t(32 * g + 16 * h) * L.dn_row_bytes;
                    if constexpr (P) decode_dn16_p(L.dn, rows, L.dn_row_bytes, W + kDnTile * g, DW + kDnD * g, h);
                    else decode_dn16(L.dn, rows, L.dn_row_bytes, W + kDnTile * g, DW + kDnD * g, h);
                }
            for (int i = pb; i < pe;) {
                // up to kNtDN pairs, never one token twice (its first contribution must be stored before the next adds)
                int m = 1;
                while (m < kNtDN && i + m < pe && B.pair_token[std::size_t(i + m)] != B.pair_token[std::size_t(i + m - 1)]) ++m;
                int pair[kNtDN];
                float * o[kNtDN];
                const float * in[kNtDN];
                for (int k = 0; k < m; ++k) {
                    pair[k] = i + k;
                    o[k] = out + std::size_t(B.pair_token[std::size_t(i + k)]) * kEmbd + r0;
                }
                pf.step();
                // the next pairs' outputs and inputs, which are rarely still in cache
                const int n0 = i + m, n1 = std::min(i + m + kNtDN, pe);
                for (int k = n0; k < n1; ++k) {
                    const float * on = out + std::size_t(B.pair_token[std::size_t(k)]) * kEmbd + r0;
                    for (int c = 0; c < 32 * ng; c += 16) prefetch_for_write(on + c);
                }
                if (n0 < n1)
                    for (std::size_t so = 0; so < 20 * cap; so += cap) {
                        const char * q = reinterpret_cast<const char *>(hq + 32 * (so + std::size_t(n0)));
                        for (int c = 0; c < int(32 * sizeof(HQ)) * (n1 - n0); c += 64) _mm_prefetch(q + c, _MM_HINT_T0);
                        _mm_prefetch(reinterpret_cast<const char *>(hd + so + std::size_t(n0)), _MM_HINT_T0);
                        if (!P) _mm_prefetch(reinterpret_cast<const char *>(hoff + so + std::size_t(n0)), _MM_HINT_T0);
                    }
                for (int g = 0; g < ng; ++g) {
                    for (int k = 0; k < m; ++k) in[k] = first[B.pair_token[std::size_t(i + k)]] ? kZeros : o[k];
                    if constexpr (P) dn_kernel_p_n(m, W + kDnTile * g, DW + kDnD * g, H, pair, in, o, B.pair_weight.data() + i);
                    else dn_kernel_n(m, W + kDnTile * g, DW + kDnD * g, H, pair, in, o, B.pair_weight.data() + i);
                    for (int k = 0; k < m; ++k) o[k] += 32;
                }
                for (int k = 0; k < m; ++k) first[B.pair_token[std::size_t(i + k)]] = 0;
                i += m;
            }
            if (u + 2 < n_experts) pf.finish();
        }
        // tokens without any pair on the CPU
        for (int t = 0; t < n_tokens; ++t)
            if (first[t]) std::memset(out + std::size_t(t) * kEmbd + r0, 0, sizeof(float) * 32 * std::size_t(ng));
    });
}

// ---------------------------------------------------------------------------------------------------------------
// Export: one expert in the GGUF's own encoding, rebuilt bit for bit from the resident copy.

namespace {

// Reverse of k_iq3s_grid: the 3-bit magnitude codes of 4 weights, c0 | c1 << 3 | c2 << 6 | c3 << 9 -> grid index. The
// 512 grid entries are distinct, so the index is unique.
const std::int32_t * iq3s_grid_index() {
    static const std::vector<std::int32_t> table = [] {
        std::vector<std::int32_t> t(4096, 0);
        for (int i = 0; i < 512; ++i) {
            int key = 0;
            for (int j = 0; j < 4; ++j) key |= int((((k_iq3s_grid[i] >> (8 * j)) & 0xFF) - 1) / 2) << (3 * j);
            t[std::size_t(key)] = i;
        }
        return t;
    }();
    return table.data();
}

// 32 codes (one per byte, weight order) -> 16 bytes, byte j = code j | code j+16 << 4: an IQ4_XS sub-block or IQ4_NL block.
inline __m128i pack_nibbles(__m256i codes) {
    return _mm_or_si128(_mm256_castsi256_si128(codes), _mm_slli_epi16(_mm256_extracti128_si256(codes, 1), 4));
}

// Q4L -> IQ3_S. Per 32 weights: the sign bits are bit 3 of the codes; each group of 4 magnitudes is a grid entry whose
// index splits into qs (low 8 bits) and one qh bit.
void export_iq3s_block(const BlockQ4L & s, BlockIQ3_S & d, const std::int32_t * index) {
    d.d = s.d;
    std::memcpy(d.scales, s.scales, 4);
    const __m256i m = _mm256_set1_epi8(0x0F), m7 = _mm256_set1_epi8(7);
    for (int c = 0; c < 4; ++c) {
        const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s.qs + 32 * c));
        for (int half = 0; half < 2; ++half) {
            const int sub = 2 * c + half;
            const __m256i codes = _mm256_and_si256(half ? _mm256_srli_epi16(raw, 4) : raw, m);
            const std::uint32_t sg = std::uint32_t(_mm256_movemask_epi8(_mm256_slli_epi16(codes, 4)));
            std::memcpy(d.signs + 4 * sub, &sg, 4);
            const __m256i key = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_and_si256(codes, m7), _mm256_set1_epi16(0x0801)),
                                                  _mm256_set1_epi32(0x00400001));
            const __m256i idx = _mm256_i32gather_epi32(index, key, 4);
            _mm_storel_epi64(reinterpret_cast<__m128i *>(d.qs + 8 * sub), _mm256_cvtepi32_epi8(idx));
            d.qh[sub] = std::uint8_t(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_slli_epi32(idx, 23))));
        }
    }
}

// Q4X -> IQ4_XS: 6-bit scales back into 4 + 2 bit fields, nibbles back into ggml's order.
void export_iq4xs_block(const BlockQ4X & s, BlockIQ4_XS & d) {
    d.d = s.d;
    std::uint16_t sh = 0;
    std::uint8_t sl[4] = {};
    for (int ib = 0; ib < 8; ++ib) {
        const int ls = s.scales[ib] + 32;
        sl[ib / 2] = std::uint8_t(sl[ib / 2] | (ls & 0xF) << 4 * (ib % 2));
        sh = std::uint16_t(sh | (ls >> 4) << 2 * ib);
    }
    d.scales_h = sh;
    std::memcpy(d.scales_l, sl, 4);
    const __m256i m = _mm256_set1_epi8(0x0F);
    for (int c = 0; c < 4; ++c) {
        const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s.qs + 32 * c));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(d.qs + 32 * c), pack_nibbles(_mm256_and_si256(raw, m)));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(d.qs + 32 * c + 16), pack_nibbles(_mm256_and_si256(_mm256_srli_epi16(raw, 4), m)));
    }
}

// IQ4L -> 20 IQ4_NL blocks.
void export_iq4nl_row(const RowIQ4L & s, BlockIQ4_NL * d) {
    const __m256i m = _mm256_set1_epi8(0x0F);
    for (int c = 0; c < 10; ++c) {
        const __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s.qs + 32 * c));
        d[2 * c].d = s.d[2 * c];
        d[2 * c + 1].d = s.d[2 * c + 1];
        _mm_storeu_si128(reinterpret_cast<__m128i *>(d[2 * c].qs), pack_nibbles(_mm256_and_si256(raw, m)));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(d[2 * c + 1].qs), pack_nibbles(_mm256_and_si256(_mm256_srli_epi16(raw, 4), m)));
    }
}

}  // namespace

CpuExperts::ExportSizes CpuExperts::export_sizes(int layer) const {
    if (!has_layer(layer)) throw std::runtime_error("CpuExperts: layer " + std::to_string(layer) + " is not loaded");
    const Layer & L = *layers_[std::size_t(layer)];
    return {row_bytes(L.gu == GateUp::Q4L ? GgufType::IQ3_S : L.gu == GateUp::Q4X ? GgufType::IQ4_XS : GgufType::Q8_0, kEmbd) * kFF,
            row_bytes(L.dn == Down::IQ4L ? GgufType::IQ4_NL : GgufType::Q8_0, kFF) * kEmbd};
}

void CpuExperts::export_expert(int layer, int e, std::uint8_t * gate, std::uint8_t * up, std::uint8_t * down) const {
    if (!has_layer(layer)) throw std::runtime_error("CpuExperts: layer " + std::to_string(layer) + " is not loaded");
    if (e < 0 || e >= kExperts) throw std::runtime_error("CpuExperts: expert id out of range");
    const Layer & L = *layers_[std::size_t(layer)];
    const std::uint8_t * src[2] = {L.gate + std::size_t(e) * L.stride, L.up + std::size_t(e) * L.stride};
    std::uint8_t * dst[2] = {gate, up};
    for (int k = 0; k < 2; ++k) {
        if (!dst[k]) continue;
        if (L.gu == GateUp::Q8_0) {
            std::memcpy(dst[k], src[k], L.gu_expert_bytes);
        } else if (L.gu == GateUp::Q4L) {
            const std::int32_t * index = iq3s_grid_index();
            const auto * sb = reinterpret_cast<const BlockQ4L *>(src[k]);
            auto * db = reinterpret_cast<BlockIQ3_S *>(dst[k]);
            for (int i = 0; i < kFF * 10; ++i) export_iq3s_block(sb[i], db[i], index);
        } else {
            const auto * sb = reinterpret_cast<const BlockQ4X *>(src[k]);
            auto * db = reinterpret_cast<BlockIQ4_XS *>(dst[k]);
            for (int i = 0; i < kFF * 10; ++i) export_iq4xs_block(sb[i], db[i]);
        }
    }
    if (!down) return;
    const std::uint8_t * ds = L.down + std::size_t(e) * L.stride;
    if (L.dn == Down::IQ4L) {
        for (int r = 0; r < kEmbd; ++r) export_iq4nl_row(reinterpret_cast<const RowIQ4L *>(ds)[r], reinterpret_cast<BlockIQ4_NL *>(down) + 20 * r);
    } else {
        std::memcpy(down, ds, L.dn_expert_bytes);
    }
}

}  // namespace ninfer::flashnext
