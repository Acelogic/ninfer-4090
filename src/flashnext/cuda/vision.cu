#include "flashnext/cuda/vision.h"

#include <cmath>
#include <stdexcept>

#include "flashnext/cuda/device.h"

namespace ninfer::flashnext::cuda {

namespace {

inline void launched(const char * what) { check(cudaGetLastError(), what); }

constexpr int kRopePairs = kVisHeadDim / 2;  // 36: dims (i, i + 36) rotate together
constexpr int kRopeFreqs = kRopePairs / 2;   // 18 frequencies, for rows and for columns
__constant__ double c_vis_freq[kRopeFreqs];  // 10000^(-j/18)

// grid cell of patch p (merge order) in a grid gw patches wide
__device__ __forceinline__ void patch_cell(int p, int gw, int & row, int & col) {
    const int block = p >> 2, w = p & 3, bw = gw >> 1;
    row = 2 * (block / bw) + (w >> 1);
    col = 2 * (block % bw) + (w & 1);
}

__global__ void k_f16_to_f32(const half * __restrict__ src, float * __restrict__ dst, std::size_t n) {
    const std::size_t i = (std::size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(src + i));
        const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(src + i + 2));
        *reinterpret_cast<float4 *>(dst + i) = make_float4(a.x, a.y, b.x, b.y);
    } else {
        for (std::size_t j = i; j < n; ++j) dst[j] = __half2float(src[j]);
    }
}

// one block per patch
__global__ void k_embed_add(float * __restrict__ x, const float * __restrict__ bias, const float * __restrict__ table, int side,
                            int gh, int gw) {
    const int p = blockIdx.x;
    int row = 0, col = 0;
    patch_cell(p, gw, row, col);
    // ggml_interpolate, bilinear with aligned corners: source coordinate i / ((dst - 1) / (src - 1)); a 1-wide
    // destination keeps the plain scale (dst / src) and samples coordinate 0
    const float sf1 = gh > 1 ? float(gh - 1) / float(side - 1) : float(gh) / float(side);
    const float sf0 = gw > 1 ? float(gw - 1) / float(side - 1) : float(gw) / float(side);
    const float y = float(row) / sf1, xf = float(col) / sf0;
    const int y0 = min(max(int(floorf(y)), 0), side - 1), y1 = min(max(y0 + 1, 0), side - 1);
    const int x0 = min(max(int(floorf(xf)), 0), side - 1), x1 = min(max(x0 + 1, 0), side - 1);
    const float dy = fminf(fmaxf(y - float(y0), 0.0f), 1.0f), dx = fminf(fmaxf(xf - float(x0), 0.0f), 1.0f);
    const float * a = table + (std::size_t(y0) * side + x0) * kVisHidden;
    const float * b = table + (std::size_t(y0) * side + x1) * kVisHidden;
    const float * c = table + (std::size_t(y1) * side + x0) * kVisHidden;
    const float * d = table + (std::size_t(y1) * side + x1) * kVisHidden;
    float * out = x + std::size_t(p) * kVisHidden;
    for (int i = threadIdx.x; i < kVisHidden; i += blockDim.x) {
        const float pos = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(__fmul_rn(a[i], 1.0f - dx), 1.0f - dy), __fmul_rn(__fmul_rn(b[i], dx), 1.0f - dy)),
                                              __fmul_rn(__fmul_rn(c[i], 1.0f - dx), dy)),
                                    __fmul_rn(__fmul_rn(d[i], dx), dy));
        out[i] = (out[i] + bias[i]) + pos;
    }
}

template <int N>
__device__ __forceinline__ double block_sum_d(double v, double * sh) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    double t = 0.0;
    for (int i = 0; i < N / 32; ++i) t += sh[i];
    return t;
}

// one block of 256 threads per row
__global__ void __launch_bounds__(256) k_layer_norm(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ b,
                                                    float * __restrict__ y, int n, float eps) {
    __shared__ double sh[8];
    const float * row = x + std::size_t(blockIdx.x) * n;
    float * out = y + std::size_t(blockIdx.x) * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) s += row[i];
    const float mean = float(block_sum_d<256>(double(s), sh) / double(n));
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) {
        const float d = row[i] - mean;
        v += d * d;
    }
    const float var = float(block_sum_d<256>(double(v), sh) / double(n));
    const float inv = 1.0f / sqrtf(var + eps);
    for (int i = threadIdx.x; i < n; i += 256) out[i] = (row[i] - mean) * inv * w[i] + b[i];
}

// one block of 576 threads per patch: q and k bias + 2-D rope (thread j: pair j % 36 of head j / 36), v bias
__global__ void __launch_bounds__(kVisHeads * kRopePairs) k_qkv_rope(float * __restrict__ qkv, const float * __restrict__ bias, int gw) {
    const int p = blockIdx.x, j = threadIdx.x;
    int row = 0, col = 0;
    patch_cell(p, gw, row, col);
    const int head = j / kRopePairs, i = j % kRopePairs;
    const double theta = double(i < kRopeFreqs ? row : col) * c_vis_freq[i % kRopeFreqs];
    const float c = float(cos(theta)), s = float(sin(theta));
    float * base = qkv + std::size_t(p) * 3 * kVisHidden;
#pragma unroll
    for (int part = 0; part < 2; ++part) {  // q, then k
        const int o0 = part * kVisHidden + head * kVisHeadDim + i, o1 = o0 + kRopePairs;
        const float x0 = base[o0] + bias[o0], x1 = base[o1] + bias[o1];
        base[o0] = x0 * c - x1 * s;
        base[o1] = x0 * s + x1 * c;
    }
    for (int k = j; k < kVisHidden; k += kVisHeads * kRopePairs) base[2 * kVisHidden + k] += bias[2 * kVisHidden + k];
}

// Bidirectional softmax attention, FP32, one block of 256 threads per (64 queries, head). Thread (ty, tx) = (tid / 16,
// tid % 16) owns query rows 4ty .. 4ty+3 and, of each 64-key tile, keys tx + 16j (j < 4); its output columns are
// tx + 16c (c < 5, below 72). Online softmax (running max and sum per row) across the key tiles.
constexpr int kAttQ = 64, kAttK = 64, kAttThreads = 256, kKsStride = kVisHeadDim + 1;
constexpr int kAttSmemFloats = kVisHeadDim * kAttQ + kAttK * kKsStride + kAttK * kVisHeadDim;

__global__ void __launch_bounds__(kAttThreads) k_vis_attention(const float * __restrict__ qkv, float * __restrict__ out, int P, float scale) {
    extern __shared__ float sm[];
    float * Qt = sm;                            // [72][64]: q transposed
    float * Ks = Qt + kVisHeadDim * kAttQ;      // [64][73]: k; then p transposed [64 keys][64 queries]
    float * Vs = Ks + kAttK * kKsStride;        // [64][72]
    float * Pt = Ks;
    const int h = blockIdx.y, q0 = blockIdx.x * kAttQ, tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const std::size_t stride = 3 * kVisHidden;
    for (int e = tid; e < kAttQ * kVisHeadDim; e += kAttThreads) {
        const int r = e / kVisHeadDim, d = e % kVisHeadDim, p = q0 + r;
        Qt[d * kAttQ + r] = p < P ? qkv[std::size_t(p) * stride + h * kVisHeadDim + d] : 0.0f;
    }
    float m[4], l[4], o[4][5];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
#pragma unroll
        for (int c = 0; c < 5; ++c) o[i][c] = 0.0f;
    }
    for (int k0 = 0; k0 < P; k0 += kAttK) {
        __syncthreads();  // the previous tile's p and v are consumed
        for (int e = tid; e < kAttK * kVisHeadDim; e += kAttThreads) {
            const int r = e / kVisHeadDim, d = e % kVisHeadDim, p = k0 + r;
            const float * src = qkv + std::size_t(p) * stride + h * kVisHeadDim + d;
            Ks[r * kKsStride + d] = p < P ? src[kVisHidden] : 0.0f;
            Vs[r * kVisHeadDim + d] = p < P ? src[2 * kVisHidden] : 0.0f;
        }
        __syncthreads();
        float s[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) s[i][j] = 0.0f;
#pragma unroll 4
        for (int d = 0; d < kVisHeadDim; ++d) {
            const float4 a = *reinterpret_cast<const float4 *>(Qt + d * kAttQ + ty * 4);
            float b[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) b[j] = Ks[(tx + 16 * j) * kKsStride + d];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                s[0][j] = fmaf(a.x, b[j], s[0][j]);
                s[1][j] = fmaf(a.y, b[j], s[1][j]);
                s[2][j] = fmaf(a.z, b[j], s[2][j]);
                s[3][j] = fmaf(a.w, b[j], s[3][j]);
            }
        }
        float pr[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float mx = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                s[i][j] = k0 + tx + 16 * j < P ? s[i][j] * scale : -INFINITY;
                mx = fmaxf(mx, s[i][j]);
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const float mn = fmaxf(m[i], mx);
            const float alpha = expf(m[i] - mn);  // 0 on the first tile (m = -inf)
            float sum = 0.0f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                pr[i][j] = expf(s[i][j] - mn);
                sum += pr[i][j];
            }
#pragma unroll
            for (int off = 1; off < 16; off <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
            l[i] = l[i] * alpha + sum;
            m[i] = mn;
#pragma unroll
            for (int c = 0; c < 5; ++c) o[i][c] *= alpha;
        }
        __syncthreads();  // every thread is done reading k: p overwrites it
#pragma unroll
        for (int j = 0; j < 4; ++j)
            *reinterpret_cast<float4 *>(Pt + (tx + 16 * j) * kAttQ + ty * 4) = make_float4(pr[0][j], pr[1][j], pr[2][j], pr[3][j]);
        __syncthreads();
        const int kn = min(kAttK, P - k0);
        for (int j = 0; j < kn; ++j) {
            const float4 pv = *reinterpret_cast<const float4 *>(Pt + j * kAttQ + ty * 4);
            const float * vr = Vs + j * kVisHeadDim + tx;
#pragma unroll
            for (int c = 0; c < 5; ++c) {
                if (c == 4 && tx >= kVisHeadDim - 64) break;
                const float v = vr[16 * c];
                o[0][c] = fmaf(pv.x, v, o[0][c]);
                o[1][c] = fmaf(pv.y, v, o[1][c]);
                o[2][c] = fmaf(pv.z, v, o[2][c]);
                o[3][c] = fmaf(pv.w, v, o[3][c]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int p = q0 + ty * 4 + i;
        if (p >= P) continue;
        const float inv = 1.0f / l[i];
        float * dst = out + std::size_t(p) * kVisHidden + h * kVisHeadDim;
#pragma unroll
        for (int c = 0; c < 5; ++c) {
            const int col = tx + 16 * c;
            if (col < kVisHeadDim) dst[col] = o[i][c] * inv;
        }
    }
}

__global__ void k_bias_add(float * __restrict__ x, const float * __restrict__ bias, std::size_t total, int n) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total) x[i] += bias[i % std::size_t(n)];
}

__global__ void k_bias_gelu(float * __restrict__ x, const float * __restrict__ bias, std::size_t total, int n, bool exact) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float v = x[i] + bias[i % std::size_t(n)];
    if (exact) {
        x[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    } else {
        const float k = 0.79788456080286535588f;  // sqrt(2 / pi)
        x[i] = 0.5f * v * (1.0f + tanhf(k * (v + 0.044715f * v * v * v)));
    }
}

}  // namespace

void vision_init() {
    double f[kRopeFreqs];
    for (int j = 0; j < kRopeFreqs; ++j) f[j] = std::pow(10000.0, -double(j) / double(kRopeFreqs));
    check(cudaMemcpyToSymbol(c_vis_freq, f, sizeof(f)), "vision rope frequencies");
    check(cudaFuncSetAttribute(k_vis_attention, cudaFuncAttributeMaxDynamicSharedMemorySize, kAttSmemFloats * int(sizeof(float))),
          "vision attention shared memory");
}

void f16_to_f32(const half * src, float * dst, std::size_t n, cudaStream_t s) {
    const std::size_t threads = (n + 3) / 4;
    k_f16_to_f32<<<unsigned((threads + 255) / 256), 256, 0, s>>>(src, dst, n);
    launched("f16_to_f32");
}

void vis_embed_add(float * x, const float * bias, const float * pos_table, int side, int gh, int gw, cudaStream_t s) {
    k_embed_add<<<gh * gw, 256, 0, s>>>(x, bias, pos_table, side, gh, gw);
    launched("vis_embed_add");
}

void vis_layer_norm(const float * x, const float * w, const float * b, float * y, int P, int n, float eps, cudaStream_t s) {
    k_layer_norm<<<P, 256, 0, s>>>(x, w, b, y, n, eps);
    launched("vis_layer_norm");
}

void vis_qkv_rope(float * qkv, const float * bias, int gw, int P, cudaStream_t s) {
    k_qkv_rope<<<P, kVisHeads * kRopePairs, 0, s>>>(qkv, bias, gw);
    launched("vis_qkv_rope");
}

void vis_attention(const float * qkv, float * out, int P, cudaStream_t s) {
    const float scale = 1.0f / std::sqrt(float(kVisHeadDim));
    k_vis_attention<<<dim3((P + kAttQ - 1) / kAttQ, kVisHeads), kAttThreads, kAttSmemFloats * sizeof(float), s>>>(qkv, out, P, scale);
    launched("vis_attention");
}

void vis_bias_add(float * x, const float * bias, int rows, int n, cudaStream_t s) {
    const std::size_t total = std::size_t(rows) * n;
    k_bias_add<<<unsigned((total + 255) / 256), 256, 0, s>>>(x, bias, total, n);
    launched("vis_bias_add");
}

void vis_bias_gelu(float * x, const float * bias, int rows, int n, bool exact, cudaStream_t s) {
    const std::size_t total = std::size_t(rows) * n;
    k_bias_gelu<<<unsigned((total + 255) / 256), 256, 0, s>>>(x, bias, total, n, exact);
    launched("vis_bias_gelu");
}

}  // namespace ninfer::flashnext::cuda
