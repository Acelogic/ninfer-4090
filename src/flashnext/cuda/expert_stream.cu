#include "flashnext/cuda/expert_stream.h"

#include <cstring>
#include <mutex>
#include <stdexcept>

#include "flashnext/cuda/device.h"
#include "flashnext/cuda/ops.h"
#include "flashnext/iq3s_grid.h"
#include "flashnext/quants.h"

namespace ninfer::flashnext::cuda {

namespace {

// IQ3_S grid index of four magnitude codes (each (m - 1) / 2 in 0..7): key = c0 | c1 << 3 | c2 << 6 | c3 << 9.
// The 512 grid entries are distinct, so the index is unique (CpuExperts' iq3s_grid_index, on the GPU).
__device__ std::uint16_t d_grid_index[4096];

constexpr int kGateRows = kExpertFF;       // 640 rows of 2560 weights
constexpr int kBlocksPerRow = kEmbd / 256;  // 10 blocks of 256 weights
constexpr int kThreads = 256;
// A thread block converts one unit: 8 rows of gate or up, or 32 rows of down. It loads the unit's source
// bytes into shared memory with 16-byte loads, converts there, and stores the result with 16-byte stores
// (every unit starts 16-byte aligned in both layouts).
constexpr int kGuRows = 8, kDnRows = 32;
constexpr int kGuUnits = kGateRows / kGuRows;  // per matrix
constexpr int kDnUnits = kEmbd / kDnRows;
constexpr int kUnits = 2 * kGuUnits + kDnUnits;
constexpr int kSmemIn = kDnRows * 20 * 34;                    // the largest source unit: 32 Q8_0 rows (21760 bytes)
constexpr int kSmemOut = kDnRows * (kExpertFF + 40);          // the largest result: 32 Q8_0 code rows and their scales
static_assert(kGuRows * kBlocksPerRow * 138 <= kSmemIn && kDnRows * 360 <= kSmemIn, "conversion staging");
static_assert(kGuRows * kBlocksPerRow * 136 <= kSmemOut, "conversion staging");

__device__ __forceinline__ std::uint32_t ld16(const std::uint8_t * p) { return *reinterpret_cast<const std::uint16_t *>(p); }
// 4 bytes from a 2-byte aligned address
__device__ __forceinline__ std::uint32_t ld32_a2(const std::uint8_t * p) { return ld16(p) | (ld16(p + 2) << 16); }
__device__ __forceinline__ void st16(std::uint8_t * p, std::uint32_t v) { *reinterpret_cast<std::uint16_t *>(p) = std::uint16_t(v); }
__device__ __forceinline__ void st32_a2(std::uint8_t * p, std::uint32_t v) {
    st16(p, v & 0xFFFF);
    st16(p + 2, v >> 16);
}

__device__ __forceinline__ void copy16(std::uint8_t * dst, const std::uint8_t * src, int bytes) {
    for (int i = threadIdx.x; i < bytes / 16; i += kThreads) reinterpret_cast<uint4 *>(dst)[i] = reinterpret_cast<const uint4 *>(src)[i];
}

// The 16-byte IQ4_NL / IQ4_XS sub-block (byte j = code j | code j+16 << 4) from 32 bytes of the Q4L chunk
// layout (byte j holds code j of the low sub-block and code j of the high one): o[0..3] for the low
// sub-block (half 0) or the high one (half 1).
__device__ __forceinline__ void chunk_to_iq4(const std::uint32_t (&w)[8], int half, std::uint32_t (&o)[4]) {
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const std::uint32_t lo = (w[k] >> (4 * half)) & 0x0F0F0F0Fu, hi = (w[k + 4] >> (4 * half)) & 0x0F0F0F0Fu;
        o[k] = lo | (hi << 4);
    }
}

// One IQ3_S sub-block (32 weights) from its Q4L codes: s points at the Q4L block, d at the IQ3_S block.
__device__ __forceinline__ void q4l_sub_to_iq3s(const std::uint8_t * s, std::uint8_t * d, int sub) {
    const int c = sub >> 1, half = sub & 1;
    std::uint32_t sign = 0, qs[2] = {0u, 0u}, qh = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {  // group k: weights 4k .. 4k+3 of the sub-block
        const std::uint32_t cw = (ld32_a2(s + offsetof(BlockQ4L, qs) + 32 * c + 4 * k) >> (4 * half)) & 0x0F0F0F0Fu;
        sign |= ((((cw & 0x08080808u) >> 3) * 0x00204081u) >> 21 & 0xFu) << (4 * k);
        const std::uint32_t m = cw & 0x07070707u;
        const std::uint32_t key = (m & 7u) | ((m >> 5) & 0x38u) | ((m >> 10) & 0x1C0u) | ((m >> 15) & 0xE00u);
        const std::uint32_t idx = d_grid_index[key];
        qs[k >> 2] |= (idx & 0xFFu) << (8 * (k & 3));
        qh |= ((idx >> 8) & 1u) << k;
    }
    st32_a2(d + offsetof(BlockIQ3_S, qs) + 8 * sub, qs[0]);
    st32_a2(d + offsetof(BlockIQ3_S, qs) + 8 * sub + 4, qs[1]);
    st32_a2(d + offsetof(BlockIQ3_S, signs) + 4 * sub, sign);
    d[offsetof(BlockIQ3_S, qh) + sub] = std::uint8_t(qh);
    if (sub == 0) {
        st16(d + offsetof(BlockIQ3_S, d), ld16(s + offsetof(BlockQ4L, d)));
        st32_a2(d + offsetof(BlockIQ3_S, scales), ld32_a2(s + offsetof(BlockQ4L, scales)));
    }
}

// One 64-weight chunk of a Q4X block into the IQ4_XS block (and the scales with chunk 0).
__device__ __forceinline__ void q4x_chunk_to_iq4xs(const std::uint8_t * s, std::uint8_t * d, int c) {
    std::uint32_t w[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) w[i] = ld32_a2(s + offsetof(BlockQ4X, qs) + 32 * c + 4 * i);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        std::uint32_t o[4];
        chunk_to_iq4(w, half, o);
        uint2 * q = reinterpret_cast<uint2 *>(d + offsetof(BlockIQ4_XS, qs) + 32 * c + 16 * half);  // 8-byte aligned
        q[0] = make_uint2(o[0], o[1]);
        q[1] = make_uint2(o[2], o[3]);
    }
    if (c == 0) {
        std::uint32_t sh = 0, sl = 0;
#pragma unroll
        for (int ib = 0; ib < 8; ++ib) {
            const int ls = int(std::int8_t(s[offsetof(BlockQ4X, scales) + ib])) + 32;
            sl |= std::uint32_t(ls & 0xF) << (4 * ib);
            sh |= std::uint32_t(ls >> 4) << (2 * ib);
        }
        *reinterpret_cast<std::uint32_t *>(d) = ld16(s + offsetof(BlockQ4X, d)) | (sh << 16);
        *reinterpret_cast<std::uint32_t *>(d + offsetof(BlockIQ4_XS, scales_l)) = sl;
    }
}

struct ConvertParams {
    ConvertBatch b;
    std::size_t gate_bytes, up_off, down_off, scale_off, down_src_off;
};

template <bool kQ4X, bool kQ8>
__global__ void __launch_bounds__(kThreads) k_convert(const ConvertParams P) {
    __shared__ __align__(16) std::uint8_t in[kSmemIn];
    __shared__ __align__(16) std::uint8_t out[kSmemOut];
    const int unit = blockIdx.x, e = blockIdx.y;
    const std::uint8_t * src = P.b.src[e];
    std::uint8_t * dst = P.b.dst[e];
    if (unit < 2 * kGuUnits) {
        const bool up = unit >= kGuUnits;
        const int r0 = (up ? unit - kGuUnits : unit) * kGuRows;
        constexpr int in_row = kBlocksPerRow * (kQ4X ? sizeof(BlockQ4X) : sizeof(BlockQ4L));
        constexpr int out_row = kBlocksPerRow * (kQ4X ? sizeof(BlockIQ4_XS) : sizeof(BlockIQ3_S));
        copy16(in, src + (up ? P.gate_bytes : 0) + std::size_t(r0) * in_row, kGuRows * in_row);
        __syncthreads();
        constexpr int blocks = kGuRows * kBlocksPerRow;
        if constexpr (kQ4X) {
            for (int i = threadIdx.x; i < blocks * 4; i += kThreads)
                q4x_chunk_to_iq4xs(in + (i >> 2) * sizeof(BlockQ4X), out + (i >> 2) * sizeof(BlockIQ4_XS), i & 3);
        } else {
            for (int i = threadIdx.x; i < blocks * 8; i += kThreads)
                q4l_sub_to_iq3s(in + (i >> 3) * sizeof(BlockQ4L), out + (i >> 3) * sizeof(BlockIQ3_S), i & 7);
        }
        __syncthreads();
        copy16(dst + (up ? P.up_off : 0) + std::size_t(r0) * out_row, out, kGuRows * out_row);
    } else {
        const int r0 = (unit - 2 * kGuUnits) * kDnRows;
        constexpr int in_row = kQ8 ? 20 * sizeof(BlockQ8_0) : sizeof(RowIQ4L);
        constexpr int code_row = kQ8 ? kExpertFF : kExpertFF / 2;
        copy16(in, src + P.down_src_off + std::size_t(r0) * in_row, kDnRows * in_row);
        __syncthreads();
        std::uint8_t * codes = out;
        std::uint8_t * scales = out + kDnRows * code_row;
        if constexpr (kQ8) {
            for (int i = threadIdx.x; i < kDnRows * 20; i += kThreads) {
                const int r = i / 20, b = i % 20;
                const std::uint8_t * blk = in + r * in_row + 34 * b;
                st16(scales + 40 * r + 2 * b, ld16(blk));
                std::uint32_t q[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) q[k] = ld32_a2(blk + 2 + 4 * k);
                uint4 * o = reinterpret_cast<uint4 *>(codes + r * code_row + 32 * b);
                o[0] = make_uint4(q[0], q[1], q[2], q[3]);
                o[1] = make_uint4(q[4], q[5], q[6], q[7]);
            }
        } else {
            for (int i = threadIdx.x; i < kDnRows * 10; i += kThreads) {
                const int r = i / 10, c = i % 10;
                const std::uint8_t * row = in + r * in_row;
                std::uint32_t w[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) w[k] = reinterpret_cast<const std::uint32_t *>(row + offsetof(RowIQ4L, qs) + 32 * c)[k];
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    std::uint32_t o[4];
                    chunk_to_iq4(w, half, o);
                    reinterpret_cast<uint4 *>(codes + r * code_row)[2 * c + half] = make_uint4(o[0], o[1], o[2], o[3]);
                }
                if (c < 5) reinterpret_cast<uint2 *>(scales + 40 * r)[c] = reinterpret_cast<const uint2 *>(row)[c];
            }
        }
        __syncthreads();
        copy16(dst + P.down_off + std::size_t(r0) * code_row, codes, kDnRows * code_row);
        copy16(dst + P.scale_off + std::size_t(r0) * 40, scales, kDnRows * 40);
    }
}

std::once_flag init_once;

}  // namespace

ExpertLayout stream_layout(const HostExpertFormat & f) {
    return expert_layout(f.gate_q4x ? GgufType::IQ4_XS : GgufType::IQ3_S, f.down_q8 ? GgufType::Q8_0 : GgufType::IQ4_NL);
}

void expert_stream_init() {
    std::call_once(init_once, [] {
        std::uint16_t t[4096] = {};
        for (int i = 0; i < 512; ++i) {
            int key = 0;
            for (int j = 0; j < 4; ++j) key |= int((((k_iq3s_grid[i] >> (8 * j)) & 0xFF) - 1) / 2) << (3 * j);
            t[key] = std::uint16_t(i);
        }
        check(cudaMemcpyToSymbol(d_grid_index, t, sizeof(t)), "expert_stream grid index");
    });
}

void convert_experts(const HostExpertFormat & f, const ExpertLayout & l, const ConvertBatch & b, cudaStream_t stream) {
    if (b.n <= 0) return;
    if (b.n > kMaxConvert) throw std::runtime_error("convert_experts: too many experts in one batch");
    const std::size_t want_gate = std::size_t(kGateRows) * kBlocksPerRow * (f.gate_q4x ? sizeof(BlockQ4X) : sizeof(BlockQ4L));
    const std::size_t want_down = std::size_t(kEmbd) * (f.down_q8 ? 20 * sizeof(BlockQ8_0) : sizeof(RowIQ4L));
    if (f.gate_bytes != want_gate || f.down_bytes != want_down) throw std::runtime_error("convert_experts: unexpected host format sizes");
    if (l.gate_type != (f.gate_q4x ? GgufType::IQ4_XS : GgufType::IQ3_S) || l.down_type != (f.down_q8 ? GgufType::Q8_0 : GgufType::IQ4_NL))
        throw std::runtime_error("convert_experts: layout does not match the host format");
    expert_stream_init();
    ConvertParams P{b, f.gate_bytes, l.up_off, l.down_off, l.scale_off, 2 * f.gate_bytes};
    const dim3 grid(kUnits, unsigned(b.n));
    if (f.gate_q4x) {
        if (f.down_q8) k_convert<true, true><<<grid, kThreads, 0, stream>>>(P);
        else k_convert<true, false><<<grid, kThreads, 0, stream>>>(P);
    } else {
        if (f.down_q8) k_convert<false, true><<<grid, kThreads, 0, stream>>>(P);
        else k_convert<false, false><<<grid, kThreads, 0, stream>>>(P);
    }
    check(cudaGetLastError(), "convert_experts");
}

}  // namespace ninfer::flashnext::cuda
