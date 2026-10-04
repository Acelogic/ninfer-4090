// CPU expert kernels for Qwen3.8-Flash-Next on Zen 4 (AVX-512 BW + VNNI).
//
// Q4L: a lossless repack of IQ3_S. Every IQ3_S weight is +/-{1,3,...,15} times a 4-bit block scale
// (2*ls+1) times a per-256 fp16 d, so it fits a 4-bit code that a 16-entry byte shuffle decodes,
// 64 weights at a time. Same values, about 22% more bytes, and no lattice-table lookups.
//
// Block of 256 weights (134 bytes):
//   d       fp16, as in IQ3_S
//   scales  4 bytes, two 4-bit block scales per byte, as in IQ3_S (scale = 2*ls+1)
//   qs      128 bytes, four chunks of 64 weights; in each chunk byte j holds weight j (low
//           nibble) and weight j+32 (high nibble). Code c means sign (c & 8) and magnitude 2*(c&7)+1.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

#include "iq3s_grid.h"

struct block_q4l {
    uint16_t d;
    uint8_t scales[4];
    uint8_t qs[128];
};
static_assert(sizeof(block_q4l) == 134, "q4l block is 134 bytes");

// IQ3_S block layout (ggml block_iq3_s).
struct block_iq3s_src {
    uint16_t d;
    uint8_t qs[64];
    uint8_t qh[8];
    uint8_t signs[32];
    uint8_t scales[4];
};
static_assert(sizeof(block_iq3s_src) == 110, "iq3_s block is 110 bytes");

// Decode one IQ3_S block to signed magnitudes (without scales), following ggml's dequantize_row_iq3_s.
inline void iq3s_signed_mags(const block_iq3s_src & b, int8_t out[256]) {
    static const uint8_t kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    const uint8_t * qs = b.qs;
    const uint8_t * qh = b.qh;
    const uint8_t * signs = b.signs;
    int8_t * y = out;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        for (int half = 0; half < 2; ++half) {
            const uint8_t h = qh[half];
            for (int l = 0; l < 4; ++l) {
                const uint8_t * g1 = (const uint8_t *) (k_iq3s_grid + (qs[2 * l + 0] | ((h << (8 - 2 * l)) & 256)));
                const uint8_t * g2 = (const uint8_t *) (k_iq3s_grid + (qs[2 * l + 1] | ((h << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = int8_t((signs[l] & kmask[j + 0]) ? -g1[j] : g1[j]);
                    y[j + 4] = int8_t((signs[l] & kmask[j + 4]) ? -g2[j] : g2[j]);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
}

inline uint8_t q4l_code(int8_t v) {  // +/-{1..15, odd} -> 4-bit code
    const int mag = v < 0 ? -v : v;
    return uint8_t(((mag - 1) >> 1) | (v < 0 ? 8 : 0));
}

inline void repack_iq3s_to_q4l(const block_iq3s_src * src, block_q4l * dst, int64_t nblocks) {
    int8_t w[256];
    for (int64_t i = 0; i < nblocks; ++i) {
        iq3s_signed_mags(src[i], w);
        dst[i].d = src[i].d;
        memcpy(dst[i].scales, src[i].scales, 4);
        for (int c = 0; c < 4; ++c) {
            const int8_t * wc = w + 64 * c;
            for (int j = 0; j < 32; ++j) dst[i].qs[32 * c + j] = uint8_t(q4l_code(wc[j]) | (q4l_code(wc[j + 32]) << 4));
        }
    }
}

// Activations for Q4L rows: int8 per 256-block with a float scale, plus per-32 sums for the offset fix.
struct ActQ8 {
    int8_t q[2560];
    float d[10];
    int32_t sum32[80];
};

inline float fp16_to_f32(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }

inline void quantize_act(const float * x, int n, ActQ8 & a) {
    for (int b = 0; b < n / 256; ++b) {
        float amax = 0;
        for (int j = 0; j < 256; ++j) amax = std::max(amax, std::fabs(x[256 * b + j]));
        const float d = amax / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
        a.d[b] = d;
        for (int j = 0; j < 256; ++j) a.q[256 * b + j] = int8_t(std::lrintf(x[256 * b + j] * id));
    }
    for (int k = 0; k < n / 32; ++k) {
        int s = 0;
        for (int j = 0; j < 32; ++j) s += a.q[32 * k + j];
        a.sum32[k] = s;
    }
}

// Code -> weight + 16 (unsigned, so maddubs can multiply it by signed activations).
inline __m512i q4l_lut() {
    alignas(16) static const int8_t lut[16] = {17, 19, 21, 23, 25, 27, 29, 31, 15, 13, 11, 9, 7, 5, 3, 1};
    return _mm512_broadcast_i32x4(_mm_load_si128((const __m128i *) lut));
}

// Dot products of two Q4L rows (gate and up) with the same activations; n is a multiple of 256.
inline void q4l_dot2(const block_q4l * g, const block_q4l * u, const ActQ8 & a, int nblocks, float & out_g, float & out_u) {
    const __m512i lut = q4l_lut();
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    float acc_g = 0, acc_u = 0;
    for (int b = 0; b < nblocks; ++b) {
        __m512i ig = _mm512_setzero_si512(), iu = _mm512_setzero_si512();
        int corr_g = 0, corr_u = 0;
        for (int c = 0; c < 4; ++c) {
            const __m512i x = _mm512_loadu_si512((const void *) (a.q + 256 * b + 64 * c));
            const int sg0 = 2 * (g[b].scales[c] & 0xF) + 1, sg1 = 2 * (g[b].scales[c] >> 4) + 1;
            const int su0 = 2 * (u[b].scales[c] & 0xF) + 1, su1 = 2 * (u[b].scales[c] >> 4) + 1;
            // each chunk of 64 spans two 32-blocks: lanes 0..15 (int16 pairs) are the first, 16..31 the second
            const __m512i scg = _mm512_inserti64x4(_mm512_set1_epi16(short(sg0)), _mm256_set1_epi16(short(sg1)), 1);
            const __m512i scu = _mm512_inserti64x4(_mm512_set1_epi16(short(su0)), _mm256_set1_epi16(short(su1)), 1);
            const __m256i rg = _mm256_loadu_si256((const __m256i *) (g[b].qs + 32 * c));
            const __m256i ru = _mm256_loadu_si256((const __m256i *) (u[b].qs + 32 * c));
            const __m512i cg = _mm512_and_si512(_mm512_inserti64x4(_mm512_castsi256_si512(rg), _mm256_srli_epi16(rg, 4), 1), m4);
            const __m512i cu = _mm512_and_si512(_mm512_inserti64x4(_mm512_castsi256_si512(ru), _mm256_srli_epi16(ru, 4), 1), m4);
            const __m512i wg = _mm512_shuffle_epi8(lut, cg);
            const __m512i wu = _mm512_shuffle_epi8(lut, cu);
            ig = _mm512_dpwssd_epi32(ig, _mm512_maddubs_epi16(wg, x), scg);
            iu = _mm512_dpwssd_epi32(iu, _mm512_maddubs_epi16(wu, x), scu);
            const int k = 8 * b + 2 * c;
            corr_g += sg0 * a.sum32[k] + sg1 * a.sum32[k + 1];
            corr_u += su0 * a.sum32[k] + su1 * a.sum32[k + 1];
        }
        const float da = a.d[b];
        acc_g += float(_mm512_reduce_add_epi32(ig) - 16 * corr_g) * fp16_to_f32(g[b].d) * da;
        acc_u += float(_mm512_reduce_add_epi32(iu) - 16 * corr_u) * fp16_to_f32(u[b].d) * da;
    }
    out_g = acc_g;
    out_u = acc_u;
}
