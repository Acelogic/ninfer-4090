// Weight formats of the Flash-Next GGUFs: block layouts and exact reference dequantization.
//
// The block layouts and reconstruction formulas follow ggml (MIT, ggml-org/ggml); these are the
// ground truth that the fast CPU and GPU kernels are tested against. Q4L is NInfer's lossless
// repack of IQ3_S for fast decoding (see q4l_kernels.h).
#pragma once
#include <cstddef>
#include <cstdint>

#include "flashnext/gguf.h"

namespace ninfer::flashnext {

float fp16_to_f32(std::uint16_t h);
float bf16_to_f32(std::uint16_t h);
std::uint16_t f32_to_fp16(float f);

#pragma pack(push, 1)
struct BlockQ8_0 {
    std::uint16_t d;
    std::int8_t qs[32];
};
struct BlockQ6_K {
    std::uint8_t ql[128];
    std::uint8_t qh[64];
    std::int8_t scales[16];
    std::uint16_t d;
};
struct BlockIQ4_NL {
    std::uint16_t d;
    std::uint8_t qs[16];
};
struct BlockIQ4_XS {
    std::uint16_t d;
    std::uint16_t scales_h;
    std::uint8_t scales_l[4];
    std::uint8_t qs[128];
};
struct BlockIQ3_S {
    std::uint16_t d;
    std::uint8_t qs[64];
    std::uint8_t qh[8];
    std::uint8_t signs[32];
    std::uint8_t scales[4];
};
// Q4L: 256 weights, each +/-{1,3,...,15} x (2*ls+1) x d, as a 4-bit code. In each 64-weight chunk
// c, byte j of qs[32c..32c+31] holds weight j (low nibble) and weight j+32 (high nibble); code k
// means sign (k & 8) and magnitude 2*(k & 7) + 1. scales and d are IQ3_S's, unchanged.
struct BlockQ4L {
    std::uint16_t d;
    std::uint8_t scales[4];
    std::uint8_t qs[128];
};
// Q4X: an IQ4_XS block in the Q4L nibble layout with plain 8-bit block scales (ls - 32) (CpuExperts' gate/up
// format for IQ4_XS layers).
struct BlockQ4X {
    std::uint16_t d;
    std::int8_t scales[8];
    std::uint8_t qs[128];
};
// IQ4L: one 640-wide IQ4_NL row (20 blocks): the 20 fp16 scales, then ten 64-weight chunks of nibbles in the Q4L
// layout (CpuExperts' down format for IQ4_NL layers).
struct RowIQ4L {
    std::uint16_t d[20];
    std::uint8_t qs[320];
};
#pragma pack(pop)
static_assert(sizeof(BlockQ8_0) == 34 && sizeof(BlockQ6_K) == 210 && sizeof(BlockIQ4_NL) == 18 &&
              sizeof(BlockIQ4_XS) == 136 && sizeof(BlockIQ3_S) == 110 && sizeof(BlockQ4L) == 134 && sizeof(BlockQ4X) == 138 &&
                  sizeof(RowIQ4L) == 360,
              "block sizes match ggml");

extern const std::int8_t kIQ4NLValues[16];

// Dequantizes `n` consecutive elements of a row (n a multiple of the type's block) to FP32.
void dequantize_row(GgufType type, const void * src, float * dst, std::int64_t n);

// Lossless IQ3_S -> Q4L for `nblocks` consecutive blocks.
void repack_iq3s_to_q4l(const BlockIQ3_S * src, BlockQ4L * dst, std::int64_t nblocks);
void dequantize_row_q4l(const BlockQ4L * src, float * dst, std::int64_t n);

}  // namespace ninfer::flashnext
