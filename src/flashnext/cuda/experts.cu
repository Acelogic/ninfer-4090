#include "flashnext/cuda/experts.h"

#include <cuda_fp16.h>

#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include "flashnext/cuda/device.h"
#include "flashnext/cuda/ops.h"
#include "flashnext/iq3s_grid.h"
#include "flashnext/quants.h"

namespace ninfer::flashnext::cuda {

namespace {

constexpr int kRowsPerBlock = 8;  // one warp per row

__device__ std::uint32_t d_iq3s_grid[512];
__constant__ std::int8_t c_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t(255); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Dot of an IQ3_S row of 2560 weights with x. Lane l takes 8 weights of every block: sub-block l/4,
// group l%4 (the two grid entries of ggml's inner loop), in ggml's weight order.
__device__ __forceinline__ float dot_iq3s(const BlockIQ3_S * __restrict__ row, const float * __restrict__ x, const std::uint32_t * grid,
                                          int lane) {
    const int sb = lane >> 2, q = lane & 3;
    float acc = 0.0f;
#pragma unroll 2
    for (int b = 0; b < kEmbd / 256; ++b) {
        const BlockIQ3_S & blk = row[b];
        const std::uint16_t qq = *reinterpret_cast<const std::uint16_t *>(blk.qs + 8 * sb + 2 * q);
        const int qh = blk.qh[sb];
        const std::uint32_t g1 = grid[(qq & 0xFF) | ((qh << (8 - 2 * q)) & 256)];
        const std::uint32_t g2 = grid[(qq >> 8) | ((qh << (7 - 2 * q)) & 256)];
        const int sgn = blk.signs[4 * sb + q];
        const int ls = (blk.scales[sb >> 1] >> (4 * (sb & 1))) & 0xF;
        const float db = __half2float(*reinterpret_cast<const __half *>(&blk.d)) * float(1 + 2 * ls);
        const float4 xa = *reinterpret_cast<const float4 *>(x + 256 * b + 32 * sb + 8 * q);
        const float4 xb = *reinterpret_cast<const float4 *>(x + 256 * b + 32 * sb + 8 * q + 4);
        auto w = [&](std::uint32_t g, int j, int bit) { const float m = float((g >> (8 * j)) & 0xFF); return (sgn >> bit) & 1 ? -m : m; };
        const float s = w(g1, 0, 0) * xa.x + w(g1, 1, 1) * xa.y + w(g1, 2, 2) * xa.z + w(g1, 3, 3) * xa.w + w(g2, 0, 4) * xb.x +
                        w(g2, 1, 5) * xb.y + w(g2, 2, 6) * xb.z + w(g2, 3, 7) * xb.w;
        acc += db * s;
    }
    return acc;
}

// Dot of an IQ4_XS row of 2560 weights with x. Lane l takes sub-block l/4, bytes 4*(l%4)..+3 of its
// 16: low nibbles are weights 32*ib + 4q + j, high nibbles 32*ib + 16 + 4q + j.
__device__ __forceinline__ float dot_iq4xs(const BlockIQ4_XS * __restrict__ row, const float * __restrict__ x, const float * lut, int lane) {
    const int ib = lane >> 2, q = lane & 3;
    float acc = 0.0f;
#pragma unroll 2
    for (int b = 0; b < kEmbd / 256; ++b) {
        const BlockIQ4_XS & blk = row[b];
        const std::uint32_t qs = *reinterpret_cast<const std::uint32_t *>(blk.qs + 16 * ib + 4 * q);
        const int ls = ((blk.scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xF) | (((blk.scales_h >> (2 * ib)) & 3) << 4);
        const float dl = __half2float(*reinterpret_cast<const __half *>(&blk.d)) * float(ls - 32);
        const float4 xa = *reinterpret_cast<const float4 *>(x + 256 * b + 32 * ib + 4 * q);
        const float4 xb = *reinterpret_cast<const float4 *>(x + 256 * b + 32 * ib + 16 + 4 * q);
        const float s = lut[qs & 0xF] * xa.x + lut[(qs >> 8) & 0xF] * xa.y + lut[(qs >> 16) & 0xF] * xa.z + lut[(qs >> 24) & 0xF] * xa.w +
                        lut[(qs >> 4) & 0xF] * xb.x + lut[(qs >> 12) & 0xF] * xb.y + lut[(qs >> 20) & 0xF] * xb.z + lut[(qs >> 28) & 0xF] * xb.w;
        acc += dl * s;
    }
    return acc;
}

template <GgufType G>
__global__ void __launch_bounds__(32 * kRowsPerBlock) k_gate_up(const std::uint8_t * __restrict__ pool, std::size_t slot_bytes,
                                                                std::size_t gate_row, std::size_t up_off, const std::int32_t * __restrict__ slots,
                                                                const float * __restrict__ x, float * __restrict__ h) {
    __shared__ std::uint32_t grid[G == GgufType::IQ3_S ? 512 : 1];
    __shared__ float lut[16];
    const int p = blockIdx.x, slot = slots[p];
    if (slot < 0) return;
    if constexpr (G == GgufType::IQ3_S) {
        for (int i = threadIdx.x; i < 512; i += blockDim.x) grid[i] = d_iq3s_grid[i];
    } else {
        if (threadIdx.x < 16) lut[threadIdx.x] = float(c_iq4nl[threadIdx.x]);
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, r = blockIdx.y * kRowsPerBlock + (threadIdx.x >> 5);
    const std::uint8_t * base = pool + std::size_t(slot) * slot_bytes;
    const float * xt = x + std::size_t(p / kUsed) * kEmbd;
    float g, u;
    if constexpr (G == GgufType::IQ3_S) {
        g = dot_iq3s(reinterpret_cast<const BlockIQ3_S *>(base + r * gate_row), xt, grid, lane);
        u = dot_iq3s(reinterpret_cast<const BlockIQ3_S *>(base + up_off + r * gate_row), xt, grid, lane);
    } else {
        g = dot_iq4xs(reinterpret_cast<const BlockIQ4_XS *>(base + r * gate_row), xt, lut, lane);
        u = dot_iq4xs(reinterpret_cast<const BlockIQ4_XS *>(base + up_off + r * gate_row), xt, lut, lane);
    }
    g = warp_sum(g);
    u = warp_sum(u);
    if (lane == 0) h[std::size_t(p) * kExpertFF + r] = g / (1.0f + expf(-g)) * u;
}

// down: 2560 rows of 640. IQ4_NL: codes [row][320] (16 bytes per 32-weight block, ggml's nibble
// order), scales [row][20]. Q8_0: codes [row][640], scales [row][20].
template <GgufType D>
__global__ void __launch_bounds__(32 * kRowsPerBlock) k_down(const std::uint8_t * __restrict__ pool, std::size_t slot_bytes,
                                                             std::size_t down_off, std::size_t scale_off, const std::int32_t * __restrict__ slots,
                                                             const float * __restrict__ weights, const float * __restrict__ h,
                                                             float * __restrict__ y) {
    __shared__ float lut[16];
    const int p = blockIdx.x, slot = slots[p];
    const int lane = threadIdx.x & 31, r = blockIdx.y * kRowsPerBlock + (threadIdx.x >> 5);
    if (slot < 0) {
        if (lane == 0) y[std::size_t(p) * kEmbd + r] = 0.0f;
        return;
    }
    if (threadIdx.x < 16) lut[threadIdx.x] = float(c_iq4nl[threadIdx.x]);
    __syncthreads();
    const std::uint8_t * base = pool + std::size_t(slot) * slot_bytes;
    const __half * sc = reinterpret_cast<const __half *>(base + scale_off) + std::size_t(r) * (kExpertFF / 32);
    const float * hp = h + std::size_t(p) * kExpertFF;
    float acc = 0.0f;
    if constexpr (D == GgufType::IQ4_NL) {
        if (lane < kExpertFF / 32) {
            const uint4 c = *reinterpret_cast<const uint4 *>(base + down_off + std::size_t(r) * (kExpertFF / 2) + 16 * lane);
            const float d = __half2float(sc[lane]);
            const float * hb = hp + 32 * lane;
            const std::uint32_t words[4] = {c.x, c.y, c.z, c.w};
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const float4 lo = *reinterpret_cast<const float4 *>(hb + 4 * w);
                const float4 hi = *reinterpret_cast<const float4 *>(hb + 16 + 4 * w);
                const std::uint32_t v = words[w];
                s += lut[v & 0xF] * lo.x + lut[(v >> 8) & 0xF] * lo.y + lut[(v >> 16) & 0xF] * lo.z + lut[(v >> 24) & 0xF] * lo.w;
                s += lut[(v >> 4) & 0xF] * hi.x + lut[(v >> 12) & 0xF] * hi.y + lut[(v >> 20) & 0xF] * hi.z + lut[(v >> 28) & 0xF] * hi.w;
            }
            acc = d * s;
        }
    } else {
        const std::int8_t * codes = reinterpret_cast<const std::int8_t *>(base + down_off) + std::size_t(r) * kExpertFF;
        for (int c = 4 * lane; c < kExpertFF; c += 128) {
            const int packed = *reinterpret_cast<const int *>(codes + c);
            const float4 hv = *reinterpret_cast<const float4 *>(hp + c);
            acc += __half2float(sc[c / 32]) *
                   (float(std::int8_t(packed)) * hv.x + float(std::int8_t(packed >> 8)) * hv.y + float(std::int8_t(packed >> 16)) * hv.z +
                    float(packed >> 24) * hv.w);
        }
    }
    acc = warp_sum(acc);
    if (lane == 0) y[std::size_t(p) * kEmbd + r] = weights[p] * acc;
}

std::once_flag grid_once;

}  // namespace

ExpertLayout expert_layout(GgufType gate_up, GgufType down) {
    if (gate_up != GgufType::IQ3_S && gate_up != GgufType::IQ4_XS)
        throw std::runtime_error(std::string("experts: no GPU kernel for gate/up ") + type_name(gate_up));
    if (down != GgufType::IQ4_NL && down != GgufType::Q8_0) throw std::runtime_error(std::string("experts: no GPU kernel for down ") + type_name(down));
    ExpertLayout l;
    l.gate_type = gate_up;
    l.down_type = down;
    l.gate_row = row_bytes(gate_up, kEmbd);
    l.up_off = align256(kExpertFF * l.gate_row);
    l.down_off = l.up_off + align256(kExpertFF * l.gate_row);
    const std::size_t code_row = down == GgufType::IQ4_NL ? kExpertFF / 2 : kExpertFF;
    l.scale_off = l.down_off + align256(std::size_t(kEmbd) * code_row);
    l.slot_bytes = l.scale_off + align256(std::size_t(kEmbd) * (kExpertFF / 32) * 2);
    return l;
}

void pack_expert(const ExpertLayout & l, const GgufTensor & gate, const GgufTensor & up, const GgufTensor & down, int e, std::uint8_t * dst) {
    if (gate.type != l.gate_type || up.type != l.gate_type || down.type != l.down_type) throw std::runtime_error("pack_expert: type mismatch");
    if (e < 0 || e >= gate.shape[2]) throw std::runtime_error("pack_expert: expert out of range");
    std::memset(dst, 0, l.slot_bytes);
    const std::size_t gbytes = kExpertFF * l.gate_row;
    std::memcpy(dst, gate.data + std::size_t(e) * gbytes, gbytes);
    std::memcpy(dst + l.up_off, up.data + std::size_t(e) * gbytes, gbytes);
    const std::size_t drow = row_bytes(down.type, kExpertFF);
    const std::uint8_t * src = down.data + std::size_t(e) * kEmbd * drow;
    auto * scales = reinterpret_cast<std::uint16_t *>(dst + l.scale_off);
    for (int r = 0; r < kEmbd; ++r) {
        for (int b = 0; b < kExpertFF / 32; ++b) {
            if (l.down_type == GgufType::IQ4_NL) {
                const auto & blk = reinterpret_cast<const BlockIQ4_NL *>(src + std::size_t(r) * drow)[b];
                std::memcpy(dst + l.down_off + std::size_t(r) * (kExpertFF / 2) + 16 * b, blk.qs, 16);
                scales[r * (kExpertFF / 32) + b] = blk.d;
            } else {
                const auto & blk = reinterpret_cast<const BlockQ8_0 *>(src + std::size_t(r) * drow)[b];
                std::memcpy(dst + l.down_off + std::size_t(r) * kExpertFF + 32 * b, blk.qs, 32);
                scales[r * (kExpertFF / 32) + b] = blk.d;
            }
        }
    }
}

void experts_init() {
    std::call_once(grid_once, [] { check(cudaMemcpyToSymbol(d_iq3s_grid, k_iq3s_grid, sizeof(k_iq3s_grid)), "iq3s grid"); });
}

void experts_gpu(const ExpertLayout & l, const std::uint8_t * pool, const std::int32_t * slots, const float * weights, const float * x,
                 float * h, float * y, int T, cudaStream_t stream) {
    experts_init();
    const dim3 block(32 * kRowsPerBlock);
    const dim3 gu(T * kUsed, kExpertFF / kRowsPerBlock), dn(T * kUsed, kEmbd / kRowsPerBlock);
    if (l.gate_type == GgufType::IQ3_S)
        k_gate_up<GgufType::IQ3_S><<<gu, block, 0, stream>>>(pool, l.slot_bytes, l.gate_row, l.up_off, slots, x, h);
    else
        k_gate_up<GgufType::IQ4_XS><<<gu, block, 0, stream>>>(pool, l.slot_bytes, l.gate_row, l.up_off, slots, x, h);
    check(cudaGetLastError(), "experts gate/up");
    if (l.down_type == GgufType::IQ4_NL)
        k_down<GgufType::IQ4_NL><<<dn, block, 0, stream>>>(pool, l.slot_bytes, l.down_off, l.scale_off, slots, weights, h, y);
    else
        k_down<GgufType::Q8_0><<<dn, block, 0, stream>>>(pool, l.slot_bytes, l.down_off, l.scale_off, slots, weights, h, y);
    check(cudaGetLastError(), "experts down");
}

}  // namespace ninfer::flashnext::cuda
