#include "flashnext/quants.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "flashnext/iq3s_grid.h"

namespace ninfer::flashnext {

const std::int8_t kIQ4NLValues[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

float fp16_to_f32(std::uint16_t h) {
    const std::uint32_t sign = std::uint32_t(h & 0x8000) << 16;
    std::uint32_t exp = (h >> 10) & 0x1F, mant = h & 0x3FF, bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {  // subnormal: normalise
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
            bits = sign | (exp << 23) | ((mant & 0x3FF) << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

float bf16_to_f32(std::uint16_t h) {
    const std::uint32_t bits = std::uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

std::uint16_t f32_to_fp16(float f) {  // round to nearest even; finite inputs
    std::uint32_t x;
    std::memcpy(&x, &f, 4);
    const std::uint32_t sign = (x >> 16) & 0x8000;
    const int exp = int((x >> 23) & 0xFF) - 127 + 15;
    std::uint32_t mant = x & 0x7FFFFF;
    if (exp >= 31) return std::uint16_t(sign | 0x7C00);
    if (exp <= 0) {
        if (exp < -10) return std::uint16_t(sign);
        mant |= 0x800000;
        const int shift = 14 - exp;
        std::uint32_t half = mant >> shift;
        const std::uint32_t rem = mant & ((1u << shift) - 1), mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1))) ++half;
        return std::uint16_t(sign | half);
    }
    std::uint32_t half = (std::uint32_t(exp) << 10) | (mant >> 13);
    const std::uint32_t rem = mant & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;
    return std::uint16_t(sign | half);
}

namespace {

// IQ3_S weights of one block as signed magnitudes +/-{1,...,15}, without scales (ggml order).
void iq3s_signed(const BlockIQ3_S & b, std::int8_t out[256]) {
    static const std::uint8_t kmask[8] = {1, 2, 4, 8, 16, 32, 64, 128};
    const std::uint8_t * qs = b.qs;
    const std::uint8_t * qh = b.qh;
    const std::uint8_t * signs = b.signs;
    std::int8_t * y = out;
    for (int ib32 = 0; ib32 < 8; ib32 += 2) {
        for (int half = 0; half < 2; ++half) {
            const std::uint8_t h = qh[half];
            for (int l = 0; l < 4; ++l) {
                const auto * g1 = reinterpret_cast<const std::uint8_t *>(k_iq3s_grid + (qs[2 * l + 0] | ((h << (8 - 2 * l)) & 256)));
                const auto * g2 = reinterpret_cast<const std::uint8_t *>(k_iq3s_grid + (qs[2 * l + 1] | ((h << (7 - 2 * l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = std::int8_t((signs[l] & kmask[j + 0]) ? -g1[j] : g1[j]);
                    y[j + 4] = std::int8_t((signs[l] & kmask[j + 4]) ? -g2[j] : g2[j]);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
        }
        qh += 2;
    }
}

std::uint8_t q4l_code(std::int8_t v) {
    const int mag = v < 0 ? -v : v;
    return std::uint8_t(((mag - 1) >> 1) | (v < 0 ? 8 : 0));
}

int q4l_value(int code) {
    const int mag = 2 * (code & 7) + 1;
    return (code & 8) ? -mag : mag;
}

}  // namespace

void repack_iq3s_to_q4l(const BlockIQ3_S * src, BlockQ4L * dst, std::int64_t nblocks) {
    std::int8_t w[256];
    for (std::int64_t i = 0; i < nblocks; ++i) {
        iq3s_signed(src[i], w);
        dst[i].d = src[i].d;
        std::memcpy(dst[i].scales, src[i].scales, 4);
        for (int c = 0; c < 4; ++c) {
            const std::int8_t * wc = w + 64 * c;
            for (int j = 0; j < 32; ++j) dst[i].qs[32 * c + j] = std::uint8_t(q4l_code(wc[j]) | (q4l_code(wc[j + 32]) << 4));
        }
    }
}

void dequantize_row_q4l(const BlockQ4L * src, float * dst, std::int64_t n) {
    for (std::int64_t i = 0; i < n / 256; ++i) {
        const float d = fp16_to_f32(src[i].d);
        for (int c = 0; c < 4; ++c) {
            const float s0 = d * float(2 * (src[i].scales[c] & 0xF) + 1), s1 = d * float(2 * (src[i].scales[c] >> 4) + 1);
            for (int j = 0; j < 32; ++j) {
                const std::uint8_t b = src[i].qs[32 * c + j];
                dst[256 * i + 64 * c + j] = s0 * float(q4l_value(b & 0xF));
                dst[256 * i + 64 * c + 32 + j] = s1 * float(q4l_value(b >> 4));
            }
        }
    }
}

void dequantize_row(GgufType type, const void * src, float * y, std::int64_t n) {
    switch (type) {
    case GgufType::F32:
        std::memcpy(y, src, std::size_t(n) * 4);
        return;
    case GgufType::F16: {
        const auto * x = static_cast<const std::uint16_t *>(src);
        for (std::int64_t i = 0; i < n; ++i) y[i] = fp16_to_f32(x[i]);
        return;
    }
    case GgufType::BF16: {
        const auto * x = static_cast<const std::uint16_t *>(src);
        for (std::int64_t i = 0; i < n; ++i) y[i] = bf16_to_f32(x[i]);
        return;
    }
    case GgufType::Q8_0: {
        const auto * x = static_cast<const BlockQ8_0 *>(src);
        for (std::int64_t i = 0; i < n / 32; ++i) {
            const float d = fp16_to_f32(x[i].d);
            for (int j = 0; j < 32; ++j) y[32 * i + j] = float(x[i].qs[j]) * d;
        }
        return;
    }
    case GgufType::IQ4_NL: {
        const auto * x = static_cast<const BlockIQ4_NL *>(src);
        for (std::int64_t i = 0; i < n / 32; ++i) {
            const float d = fp16_to_f32(x[i].d);
            for (int j = 0; j < 16; ++j) {
                y[32 * i + j] = d * float(kIQ4NLValues[x[i].qs[j] & 0xF]);
                y[32 * i + j + 16] = d * float(kIQ4NLValues[x[i].qs[j] >> 4]);
            }
        }
        return;
    }
    case GgufType::IQ4_XS: {
        const auto * x = static_cast<const BlockIQ4_XS *>(src);
        for (std::int64_t i = 0; i < n / 256; ++i) {
            const float d = fp16_to_f32(x[i].d);
            const std::uint8_t * qs = x[i].qs;
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = ((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xF) | (((x[i].scales_h >> 2 * ib) & 3) << 4);
                const float dl = d * float(ls - 32);
                for (int j = 0; j < 16; ++j) {
                    y[256 * i + 32 * ib + j] = dl * float(kIQ4NLValues[qs[j] & 0xF]);
                    y[256 * i + 32 * ib + j + 16] = dl * float(kIQ4NLValues[qs[j] >> 4]);
                }
                qs += 16;
            }
        }
        return;
    }
    case GgufType::IQ3_S: {
        const auto * x = static_cast<const BlockIQ3_S *>(src);
        std::int8_t w[256];
        for (std::int64_t i = 0; i < n / 256; ++i) {
            iq3s_signed(x[i], w);
            const float d = fp16_to_f32(x[i].d);
            for (int ib32 = 0; ib32 < 8; ++ib32) {
                const int ls = (x[i].scales[ib32 / 2] >> (4 * (ib32 % 2))) & 0xF;
                const float db = d * float(1 + 2 * ls);
                for (int j = 0; j < 32; ++j) y[256 * i + 32 * ib32 + j] = db * float(w[32 * ib32 + j]);
            }
        }
        return;
    }
    case GgufType::Q6_K: {
        const auto * x = static_cast<const BlockQ6_K *>(src);
        for (std::int64_t i = 0; i < n / 256; ++i) {
            const float d = fp16_to_f32(x[i].d);
            const std::uint8_t * ql = x[i].ql;
            const std::uint8_t * qh = x[i].qh;
            const std::int8_t * sc = x[i].scales;
            float * yy = y + 256 * i;
            for (int half = 0; half < 2; ++half) {
                for (int l = 0; l < 32; ++l) {
                    const int is = l / 16;
                    const int q1 = int((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                    const int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    const int q3 = int((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    yy[l + 0] = d * float(sc[is + 0]) * float(q1);
                    yy[l + 32] = d * float(sc[is + 2]) * float(q2);
                    yy[l + 64] = d * float(sc[is + 4]) * float(q3);
                    yy[l + 96] = d * float(sc[is + 6]) * float(q4);
                }
                yy += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
        return;
    }
    }
    throw std::runtime_error("dequantize_row: unsupported type");
}

}  // namespace ninfer::flashnext
