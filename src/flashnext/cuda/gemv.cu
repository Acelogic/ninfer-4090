#include "flashnext/cuda/gemv.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstring>
#include <vector>

#include "flashnext/quants.h"

namespace ninfer::flashnext::cuda {

namespace {

// A block is 8 warps. WPR warps share one row (split-K), so a block covers 8 / WPR rows: matrices
// with few rows (the hyper-connection projections have 320, the router 512) still fill the GPU.
constexpr int kWarps = 8;

int split_of(const GpuWeight & w);

template <int T, int WPR>
__device__ __forceinline__ void finish(float (&acc)[T], float * __restrict__ y, int n, int row) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], o);
    if constexpr (WPR == 1) {
        if (lane == 0 && row < n)
#pragma unroll
            for (int t = 0; t < T; ++t) y[std::size_t(t) * n + row] = acc[t];
    } else {
        __shared__ float part[kWarps][T];
        if (lane == 0)
#pragma unroll
            for (int t = 0; t < T; ++t) part[warp][t] = acc[t];
        __syncthreads();
        if (warp % WPR == 0 && lane < T && row < n) {
            float s = 0.0f;
#pragma unroll
            for (int w = 0; w < WPR; ++w) s += part[warp + w][lane];
            y[std::size_t(lane) * n + row] = s;
        }
    }
}

// Q8 split: per step a warp reads 128 contiguous codes (4 per lane) and the matching activations;
// the 8 lanes of each 32-code group share its scale.
template <int T, int WPR>
__global__ void __launch_bounds__(32 * kWarps) gemv_q8(const std::int8_t * __restrict__ codes, const __half * __restrict__ scales, int n,
                                                       int k, const float * __restrict__ x, float * __restrict__ y) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x * (kWarps / WPR) + warp / WPR;
    float acc[T] = {};
    if (row < n) {
        const std::int8_t * wr = codes + std::size_t(row) * k;
        const __half * sr = scales + std::size_t(row) * (k / 32);
#pragma unroll 4
        for (int c = 128 * (warp % WPR) + 4 * lane; c < k; c += 128 * WPR) {
            const int packed = __ldg(reinterpret_cast<const int *>(wr + c));
            const float s = __half2float(__ldg(sr + c / 32));
            const float w0 = float(std::int8_t(packed)), w1 = float(std::int8_t(packed >> 8));
            const float w2 = float(std::int8_t(packed >> 16)), w3 = float(packed >> 24);
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const float4 xv = __ldg(reinterpret_cast<const float4 *>(x + std::size_t(t) * k + c));
                acc[t] += s * (w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w);
            }
        }
    }
    finish<T, WPR>(acc, y, n, row);
}

template <int T, int WPR>
__global__ void __launch_bounds__(32 * kWarps) gemv_f32(const float * __restrict__ w, int n, int k, const float * __restrict__ x,
                                                        float * __restrict__ y) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x * (kWarps / WPR) + warp / WPR;
    float acc[T] = {};
    if (row < n) {
        const float * wr = w + std::size_t(row) * k;
#pragma unroll 4
        for (int c = 128 * (warp % WPR) + 4 * lane; c < k; c += 128 * WPR) {
            const float4 wv = __ldg(reinterpret_cast<const float4 *>(wr + c));
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const float4 xv = __ldg(reinterpret_cast<const float4 *>(x + std::size_t(t) * k + c));
                acc[t] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
            }
        }
    }
    finish<T, WPR>(acc, y, n, row);
}

template <int T, int WPR>
__global__ void __launch_bounds__(32 * kWarps) gemv_bf16(const __nv_bfloat16 * __restrict__ w, int n, int k, const float * __restrict__ x,
                                                         float * __restrict__ y) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x * (kWarps / WPR) + warp / WPR;
    float acc[T] = {};
    if (row < n) {
        const __nv_bfloat16 * wr = w + std::size_t(row) * k;
#pragma unroll 4
        for (int c = 128 * (warp % WPR) + 4 * lane; c < k; c += 128 * WPR) {
            const uint2 raw = __ldg(reinterpret_cast<const uint2 *>(wr + c));
            const float w0 = __uint_as_float(raw.x << 16), w1 = __uint_as_float(raw.x & 0xffff0000u);
            const float w2 = __uint_as_float(raw.y << 16), w3 = __uint_as_float(raw.y & 0xffff0000u);
#pragma unroll
            for (int t = 0; t < T; ++t) {
                const float4 xv = __ldg(reinterpret_cast<const float4 *>(x + std::size_t(t) * k + c));
                acc[t] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
            }
        }
    }
    finish<T, WPR>(acc, y, n, row);
}

// Q6_K: one 256-weight block per step; lane l dequantizes positions 128*half + l + {0,32,64,96}
// exactly as ggml's dequantize_row_q6_K, so the activations it reads are contiguous across the warp.
struct BlockQ6K {
    std::uint8_t ql[128];
    std::uint8_t qh[64];
    std::int8_t scales[16];
    __half d;
};
static_assert(sizeof(BlockQ6K) == 210, "q6_K block");

template <int T, int WPR>
__global__ void __launch_bounds__(32 * kWarps) gemv_q6k(const BlockQ6K * __restrict__ w, int n, int k, const float * __restrict__ x,
                                                        float * __restrict__ y) {
    const int l = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x * (kWarps / WPR) + warp / WPR;
    float acc[T] = {};
    if (row < n) {
        const int nb = k / 256;
        const BlockQ6K * br = w + std::size_t(row) * nb;
        for (int b = warp % WPR; b < nb; b += WPR) {
            const BlockQ6K & blk = br[b];
            const float d = __half2float(blk.d);
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int ql0 = blk.ql[64 * half + l], ql1 = blk.ql[64 * half + l + 32], qh = blk.qh[32 * half + l];
                const std::int8_t * sc = blk.scales + 8 * half + l / 16;
                const float v0 = d * sc[0] * float(((ql0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
                const float v1 = d * sc[2] * float(((ql1 & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
                const float v2 = d * sc[4] * float(((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32);
                const float v3 = d * sc[6] * float(((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32);
                const int base = 256 * b + 128 * half + l;
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float * xt = x + std::size_t(t) * k + base;
                    acc[t] += v0 * __ldg(xt) + v1 * __ldg(xt + 32) + v2 * __ldg(xt + 64) + v3 * __ldg(xt + 96);
                }
            }
        }
    }
    finish<T, WPR>(acc, y, n, row);
}

template <int T, int WPR>
void launch(const GpuWeight & w, const float * x, float * y, cudaStream_t stream) {
    const dim3 grid((w.n + kWarps / WPR - 1) / (kWarps / WPR)), block(32 * kWarps);
    switch (w.format) {
    case WeightFormat::Q8_SPLIT:
        gemv_q8<T, WPR><<<grid, block, 0, stream>>>(static_cast<const std::int8_t *>(w.data), static_cast<const __half *>(w.scales), w.n,
                                                    w.k, x, y);
        break;
    case WeightFormat::F32:
        gemv_f32<T, WPR><<<grid, block, 0, stream>>>(static_cast<const float *>(w.data), w.n, w.k, x, y);
        break;
    case WeightFormat::BF16:
        gemv_bf16<T, WPR><<<grid, block, 0, stream>>>(static_cast<const __nv_bfloat16 *>(w.data), w.n, w.k, x, y);
        break;
    case WeightFormat::Q6_K:
        gemv_q6k<T, WPR><<<grid, block, 0, stream>>>(static_cast<const BlockQ6K *>(w.data), w.n, w.k, x, y);
        break;
    }
}

template <int T>
void launch_split(const GpuWeight & w, const float * x, float * y, cudaStream_t stream) {
    switch (split_of(w)) {
    case 1: launch<T, 1>(w, x, y, stream); break;
    case 2: launch<T, 2>(w, x, y, stream); break;
    case 4: launch<T, 4>(w, x, y, stream); break;
    default: launch<T, 8>(w, x, y, stream); break;
    }
}

int split_of(const GpuWeight & w) {
    // Split K across warps until there are enough warps in flight, keeping at least 512 weights per warp.
    int wpr = 1;
    while (wpr < kWarps && std::int64_t(w.n) * wpr < 8192 && w.k >= 512 * wpr) wpr *= 2;
    return wpr;
}

struct MultiSeg {
    WeightFormat format;
    int n, k, wpr, block_begin;
    const void * data;
    const void * scales;
    float * y;
};
struct MultiArgs {
    MultiSeg seg[kMaxMulti];
    int count;
};

// One launch over several matrices; each block belongs to one matrix and follows the same loops as
// the single-matrix kernels above, so the sums are formed in the same order.
template <int T>
__global__ void __launch_bounds__(32 * kWarps) gemv_multi_kernel(const MultiArgs a, const float * __restrict__ x) {
    int s = 0;
    while (s + 1 < a.count && int(blockIdx.x) >= a.seg[s + 1].block_begin) ++s;
    const MultiSeg & g = a.seg[s];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wpr = g.wpr, part = warp % wpr;
    const int row = (int(blockIdx.x) - g.block_begin) * (kWarps / wpr) + warp / wpr;
    const int n = g.n, k = g.k;
    float acc[T] = {};
    if (row < n) {
        switch (g.format) {
        case WeightFormat::Q8_SPLIT: {
            const std::int8_t * wr = static_cast<const std::int8_t *>(g.data) + std::size_t(row) * k;
            const __half * sr = static_cast<const __half *>(g.scales) + std::size_t(row) * (k / 32);
#pragma unroll 4
            for (int c = 128 * part + 4 * lane; c < k; c += 128 * wpr) {
                const int packed = __ldg(reinterpret_cast<const int *>(wr + c));
                const float sc = __half2float(__ldg(sr + c / 32));
                const float w0 = float(std::int8_t(packed)), w1 = float(std::int8_t(packed >> 8));
                const float w2 = float(std::int8_t(packed >> 16)), w3 = float(packed >> 24);
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float4 xv = __ldg(reinterpret_cast<const float4 *>(x + std::size_t(t) * k + c));
                    acc[t] += sc * (w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w);
                }
            }
            break;
        }
        case WeightFormat::F32: {
            const float * wr = static_cast<const float *>(g.data) + std::size_t(row) * k;
#pragma unroll 4
            for (int c = 128 * part + 4 * lane; c < k; c += 128 * wpr) {
                const float4 wv = __ldg(reinterpret_cast<const float4 *>(wr + c));
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float4 xv = __ldg(reinterpret_cast<const float4 *>(x + std::size_t(t) * k + c));
                    acc[t] += wv.x * xv.x + wv.y * xv.y + wv.z * xv.z + wv.w * xv.w;
                }
            }
            break;
        }
        case WeightFormat::BF16: {
            const __nv_bfloat16 * wr = static_cast<const __nv_bfloat16 *>(g.data) + std::size_t(row) * k;
#pragma unroll 4
            for (int c = 128 * part + 4 * lane; c < k; c += 128 * wpr) {
                const uint2 raw = __ldg(reinterpret_cast<const uint2 *>(wr + c));
                const float w0 = __uint_as_float(raw.x << 16), w1 = __uint_as_float(raw.x & 0xffff0000u);
                const float w2 = __uint_as_float(raw.y << 16), w3 = __uint_as_float(raw.y & 0xffff0000u);
#pragma unroll
                for (int t = 0; t < T; ++t) {
                    const float4 xv = __ldg(reinterpret_cast<const float4 *>(x + std::size_t(t) * k + c));
                    acc[t] += w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
                }
            }
            break;
        }
        case WeightFormat::Q6_K: {
            const BlockQ6K * br = static_cast<const BlockQ6K *>(g.data) + std::size_t(row) * (k / 256);
            for (int b = part; b < k / 256; b += wpr) {
                const BlockQ6K & blk = br[b];
                const float d = __half2float(blk.d);
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int ql0 = blk.ql[64 * half + lane], ql1 = blk.ql[64 * half + lane + 32], qh = blk.qh[32 * half + lane];
                    const std::int8_t * sc = blk.scales + 8 * half + lane / 16;
                    const float v0 = d * sc[0] * float(((ql0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
                    const float v1 = d * sc[2] * float(((ql1 & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
                    const float v2 = d * sc[4] * float(((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32);
                    const float v3 = d * sc[6] * float(((ql1 >> 4) | (((qh >> 6) & 3) << 4)) - 32);
                    const int base = 256 * b + 128 * half + lane;
#pragma unroll
                    for (int t = 0; t < T; ++t) {
                        const float * xt = x + std::size_t(t) * k + base;
                        acc[t] += v0 * __ldg(xt) + v1 * __ldg(xt + 32) + v2 * __ldg(xt + 64) + v3 * __ldg(xt + 96);
                    }
                }
            }
            break;
        }
        }
    }
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], o);
    if (wpr == 1) {
        if (lane == 0 && row < n)
#pragma unroll
            for (int t = 0; t < T; ++t) g.y[std::size_t(t) * n + row] = acc[t];
        return;
    }
    __shared__ float red[kWarps][T];
    if (lane == 0)
#pragma unroll
        for (int t = 0; t < T; ++t) red[warp][t] = acc[t];
    __syncthreads();
    if (part == 0 && lane < T && row < n) {
        float sum = 0.0f;
        for (int w = 0; w < wpr; ++w) sum += red[warp + w][lane];
        g.y[std::size_t(lane) * n + row] = sum;
    }
}

}  // namespace

DeviceWeight upload_gemv_weight(const GgufTensor & t) {
    for (std::size_t i = 2; i < t.shape.size(); ++i)
        if (t.shape[i] != 1) throw std::runtime_error(t.name + ": not a 2-D tensor");
    const std::int64_t k = t.shape[0], n = t.shape.size() > 1 ? t.shape[1] : 1;
    if (k > (1 << 30) || n > (1 << 30)) throw std::runtime_error(t.name + ": too large for a GEMV");
    DeviceWeight dw;
    dw.view.n = int(n);
    dw.view.k = int(k);
    switch (t.type) {
    case GgufType::Q8_0: {
        dw.view.format = WeightFormat::Q8_SPLIT;
        const std::int64_t nb = k / 32;
        const std::size_t code_bytes = std::size_t(n) * k, scale_off = (code_bytes + 255) & ~std::size_t(255);
        std::vector<std::uint8_t> host(scale_off + std::size_t(n) * nb * 2);
        const auto * blocks = reinterpret_cast<const BlockQ8_0 *>(t.data);
        auto * scales = reinterpret_cast<std::uint16_t *>(host.data() + scale_off);
        for (std::int64_t r = 0; r < n; ++r)
            for (std::int64_t b = 0; b < nb; ++b) {
                const BlockQ8_0 & blk = blocks[r * nb + b];
                std::memcpy(host.data() + r * k + 32 * b, blk.qs, 32);
                scales[r * nb + b] = blk.d;
            }
        dw.buffer = DeviceBuffer(host.size());
        check(cudaMemcpy(dw.buffer.get(), host.data(), host.size(), cudaMemcpyHostToDevice), "upload");
        dw.view.data = dw.buffer.get();
        dw.view.scales = dw.buffer.as<std::uint8_t>() + scale_off;
        break;
    }
    case GgufType::F32:
    case GgufType::BF16:
    case GgufType::Q6_K:
        dw.view.format = t.type == GgufType::F32 ? WeightFormat::F32 : t.type == GgufType::BF16 ? WeightFormat::BF16 : WeightFormat::Q6_K;
        dw.buffer = DeviceBuffer(t.bytes);
        check(cudaMemcpy(dw.buffer.get(), t.data, t.bytes, cudaMemcpyHostToDevice), "upload");
        dw.view.data = dw.buffer.get();
        break;
    default:
        throw std::runtime_error(t.name + ": no GEMV kernel for " + type_name(t.type));
    }
    return dw;
}

void gemv(const GpuWeight & w, const float * x, float * y, int tokens, cudaStream_t stream) {
    const int align = w.format == WeightFormat::Q8_SPLIT ? 32 : w.format == WeightFormat::Q6_K ? 256 : 4;
    if (w.n <= 0 || w.k <= 0 || w.k % align != 0)
        throw std::runtime_error("gemv: K=" + std::to_string(w.k) + " is not a multiple of " + std::to_string(align));
    switch (tokens) {
    case 1: launch_split<1>(w, x, y, stream); break;
    case 2: launch_split<2>(w, x, y, stream); break;
    case 3: launch_split<3>(w, x, y, stream); break;
    case 4: launch_split<4>(w, x, y, stream); break;
    default: throw std::runtime_error("gemv: tokens must be 1..4");
    }
}

void gemv_multi(const GemvTarget * targets, int count, const float * x, int tokens, cudaStream_t stream) {
    if (count < 1 || count > kMaxMulti) throw std::runtime_error("gemv_multi: 1 to 4 matrices");
    MultiArgs a{};
    a.count = count;
    int blocks = 0;
    for (int i = 0; i < count; ++i) {
        const GpuWeight & w = *targets[i].w;
        const int align = w.format == WeightFormat::Q8_SPLIT ? 32 : w.format == WeightFormat::Q6_K ? 256 : 4;
        if (w.n <= 0 || w.k <= 0 || w.k % align != 0) throw std::runtime_error("gemv_multi: bad matrix shape");
        if (i > 0 && w.k != targets[0].w->k) throw std::runtime_error("gemv_multi: matrices must share the input length");
        MultiSeg & g = a.seg[i];
        g.format = w.format;
        g.n = w.n;
        g.k = w.k;
        g.wpr = split_of(w);
        g.block_begin = blocks;
        g.data = w.data;
        g.scales = w.scales;
        g.y = targets[i].y;
        blocks += (w.n + kWarps / g.wpr - 1) / (kWarps / g.wpr);
    }
    switch (tokens) {
    case 1: gemv_multi_kernel<1><<<blocks, 32 * kWarps, 0, stream>>>(a, x); break;
    case 2: gemv_multi_kernel<2><<<blocks, 32 * kWarps, 0, stream>>>(a, x); break;
    case 3: gemv_multi_kernel<3><<<blocks, 32 * kWarps, 0, stream>>>(a, x); break;
    case 4: gemv_multi_kernel<4><<<blocks, 32 * kWarps, 0, stream>>>(a, x); break;
    default: throw std::runtime_error("gemv_multi: tokens must be 1..4");
    }
    check(cudaGetLastError(), "gemv_multi");
}

}  // namespace ninfer::flashnext::cuda
