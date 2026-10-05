#include "flashnext/cuda/ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>

#include "flashnext/cuda/device.h"
#include "flashnext/cuda/experts.h"

namespace ninfer::flashnext::cuda {

namespace {

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplusf_(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Sum over the whole block, returned to every thread. sh: 32 floats of shared memory.
__device__ float block_sum(float v, float * sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_sum(v);
    __syncthreads();  // sh may still be read by the previous call
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    return warp_sum(lane < nw ? sh[lane] : 0.0f);
}
__device__ float block_max(float v, float * sh) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) sh[warp] = v;
    __syncthreads();
    return warp_max(lane < nw ? sh[lane] : -INFINITY);
}

inline unsigned blocks(std::size_t n, unsigned per) { return unsigned((n + per - 1) / per); }

inline void launched(const char * what) { check(cudaGetLastError(), what); }

// ------------------------------------------------------------------------------------------------
// hyper-connections

__global__ void k_hc_expand(const float * __restrict__ x, float * __restrict__ res, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) res[i] = x[(i / kHcd) * kEmbd + i % kEmbd];
}

__global__ void __launch_bounds__(256) k_hc_norm(const float * __restrict__ res, const float * __restrict__ w, float * __restrict__ xn,
                                                 float eps) {
    __shared__ float sh[32];
    const float * x = res + std::size_t(blockIdx.x) * kEmbd;
    const float * wc = w + (blockIdx.x % kHc) * kEmbd;
    float * y = xn + std::size_t(blockIdx.x) * kEmbd;
    constexpr int kPer = kEmbd / 256;  // the row stays in registers: one read
    float v[kPer];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
        v[j] = x[threadIdx.x + 256 * j];
        ss += v[j] * v[j];
    }
    ss = block_sum(ss, sh);
    const float scale = 1.0f / sqrtf(ss / float(kEmbd) + eps);
#pragma unroll
    for (int j = 0; j < kPer; ++j) y[threadIdx.x + 256 * j] = (v[j] * scale) * wc[threadIdx.x + 256 * j];
}

__global__ void k_hc_lowrank_act(float * lo, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) lo[i] = siluf_(lo[i] * (1.0f / kHc));
}

__global__ void k_hc_gate_mean(const float * __restrict__ xn, const float * __restrict__ gate, float * __restrict__ mixed, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::size_t base = std::size_t(i / kEmbd) * kHcd + i % kEmbd;
    float m = xn[base] * sigmoidf_(gate[base]);
#pragma unroll
    for (int c = 1; c < kHc; ++c) m = m + xn[base + c * kEmbd] * sigmoidf_(gate[base + c * kEmbd]);
    mixed[i] = m * (1.0f / kHc);
}

__global__ void k_hc_combine(float * __restrict__ res, const float * __restrict__ out, const float * __restrict__ inject, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int t = i / kHcd, c = (i / kEmbd) % kHc;
    const float w = 2.0f * sigmoidf_(inject[t * kHc + c] * (1.0f / kHc));
    res[i] = res[i] + out[t * kEmbd + i % kEmbd] * w;
}

// ------------------------------------------------------------------------------------------------
// PLE

__global__ void __launch_bounds__(256) k_ple_gate(const float * __restrict__ key, const float * __restrict__ value, const float * __restrict__ res,
                                                  const float * __restrict__ nk, const float * __restrict__ nq, const float * __restrict__ nc,
                                                  float * __restrict__ gated, float * __restrict__ normalized, float * __restrict__ gates,
                                                  float eps) {
    __shared__ float sh[32];
    const int t = blockIdx.x / kHc, c = blockIdx.x % kHc;
    const std::size_t off = std::size_t(t) * kHcd + c * kEmbd;
    const float * kc = key + off;
    const float * rc = res + off;
    const float * val = value + std::size_t(t) * kEmbd;
    float sk = 0.0f, sq = 0.0f;
    for (int i = threadIdx.x; i < kEmbd; i += 256) {
        sk += kc[i] * kc[i];
        sq += rc[i] * rc[i];
    }
    sk = 1.0f / sqrtf(block_sum(sk, sh) / float(kEmbd) + eps);
    sq = 1.0f / sqrtf(block_sum(sq, sh) / float(kEmbd) + eps);
    float dot = 0.0f;
    for (int i = threadIdx.x; i < kEmbd; i += 256) dot += ((kc[i] * sk) * nk[c * kEmbd + i]) * ((rc[i] * sq) * nq[c * kEmbd + i]);
    const float s = block_sum(dot, sh) * (1.0f / sqrtf(float(kEmbd)));
    const float mag = sqrtf(fminf(fmaxf(fabsf(s), 1e-6f), INFINITY));
    const float sgn = s > 0.0f ? 1.0f : (s < 0.0f ? -1.0f : 0.0f);
    const float g = sigmoidf_(sgn * mag);
    if (threadIdx.x == 0) gates[t * kHc + c] = g;
    float sg = 0.0f;
    for (int i = threadIdx.x; i < kEmbd; i += 256) {
        const float v = val[i] * g;
        gated[off + i] = v;
        sg += v * v;
    }
    sg = 1.0f / sqrtf(block_sum(sg, sh) / float(kEmbd) + eps);
    for (int i = threadIdx.x; i < kEmbd; i += 256) normalized[off + i] = (gated[off + i] * sg) * nc[c * kEmbd + i];
}

__global__ void k_ple_conv_add(float * __restrict__ res, const float * __restrict__ gated, const float * __restrict__ normalized,
                               const float * __restrict__ conv_w, float * __restrict__ hist, int T) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= kHcd) return;
    float w[kPleKernel];
    for (int k = 0; k < kPleKernel; ++k) w[k] = conv_w[ch * kPleKernel + k];
    // input at token r of this step: the history row for r < 0, else the new normalized row
    auto in = [&](int r) { return r < 0 ? hist[std::size_t(kPleHist + r) * kHcd + ch] : normalized[std::size_t(r) * kHcd + ch]; };
    for (int t = 0; t < T; ++t) {
        float acc = 0.0f;
        for (int k = 0; k < kPleKernel; ++k) {
            const float term = in(t - (kPleKernel - 1 - k) * kPleDilation) * w[k];
            acc = k == 0 ? term : acc + term;
        }
        const std::size_t i = std::size_t(t) * kHcd + ch;
        res[i] = res[i] + (gated[i] + siluf_(acc));
    }
    // the last 9 inputs become the history, oldest first (reads run ahead of the writes)
    for (int j = 0; j < kPleHist; ++j) hist[std::size_t(j) * kHcd + ch] = in(T - kPleHist + j);
}

// ------------------------------------------------------------------------------------------------
// Gated DeltaNet

// One block per head of 128 channels (32 q/k heads, then 48 v heads).
__global__ void __launch_bounds__(kDnState) k_dn_conv(const float * __restrict__ qkv, float * __restrict__ state, const float * __restrict__ w,
                                                      float * __restrict__ out, int T, float eps, float * __restrict__ snap) {
    __shared__ float sh[32];
    const int c = blockIdx.x * kDnState + threadIdx.x;
    const bool qk = blockIdx.x < 2 * kDnKHeads;
    float s0 = state[c], s1 = state[kDnConvDim + c], s2 = state[2 * kDnConvDim + c];
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    for (int t = 0; t < T; ++t) {
        const float x = qkv[std::size_t(t) * kDnConvDim + c];
        float sum = 0.0f;
        sum += s0 * w0;
        sum += s1 * w1;
        sum += s2 * w2;
        sum += x * w3;
        float y = siluf_(sum);
        if (qk) y *= 1.0f / fmaxf(sqrtf(block_sum(y * y, sh)), eps);
        out[std::size_t(t) * kDnConvDim + c] = y;
        s0 = s1;
        s1 = s2;
        s2 = x;
        if (snap && t + 1 < T) {
            float * sp = snap + std::size_t(t) * 3 * kDnConvDim;
            sp[c] = s0;
            sp[kDnConvDim + c] = s1;
            sp[2 * kDnConvDim + c] = s2;
        }
    }
    state[c] = s0;
    state[kDnConvDim + c] = s1;
    state[2 * kDnConvDim + c] = s2;
}

// One block per v head: 16 warps x 8 state rows; lane l holds columns 4l..4l+3 of its rows.
constexpr int kDnRowsPerWarp = 8;
constexpr int kDnThreads = 32 * (kDnState / kDnRowsPerWarp);  // 512

__global__ void __launch_bounds__(kDnThreads) k_dn_recurrence(const float * __restrict__ conv, const float * __restrict__ z,
                                                              const float * __restrict__ beta, const float * __restrict__ alpha,
                                                              const float * __restrict__ dt, const float * __restrict__ a,
                                                              const float * __restrict__ norm_w, float * __restrict__ S,
                                                              float * __restrict__ out, int T, float eps, float * __restrict__ snap) {
    __shared__ float o_sh[kDnState];
    __shared__ float sh[32];
    const int h = blockIdx.x, hk = h % kDnKHeads;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float4 * Sh = reinterpret_cast<float4 *>(S + std::size_t(h) * kDnState * kDnState);
    float4 st[kDnRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kDnRowsPerWarp; ++r) st[r] = Sh[(warp * kDnRowsPerWarp + r) * (kDnState / 4) + lane];
    const float scale = 1.0f / sqrtf(float(kDnState));
    for (int t = 0; t < T; ++t) {
        const float * row = conv + std::size_t(t) * kDnConvDim;
        const float4 q = reinterpret_cast<const float4 *>(row + hk * kDnState)[lane];
        const float4 k = reinterpret_cast<const float4 *>(row + kDnKeyDim + hk * kDnState)[lane];
        const float * v = row + 2 * kDnKeyDim + h * kDnState;
        const float b = sigmoidf_(beta[t * kDnVHeads + h]);
        const float decay = expf(softplusf_(alpha[t * kDnVHeads + h] + dt[h]) * a[h]);
#pragma unroll
        for (int r = 0; r < kDnRowsPerWarp; ++r) {
            float4 & s = st[r];
            s.x *= decay; s.y *= decay; s.z *= decay; s.w *= decay;
            const float sk = warp_sum(s.x * k.x + s.y * k.y + s.z * k.z + s.w * k.w);
            const int j = warp * kDnRowsPerWarp + r;
            const float d = (v[j] - sk) * b;
            s.x += k.x * d; s.y += k.y * d; s.z += k.z * d; s.w += k.w * d;
            const float o = warp_sum(s.x * q.x + s.y * q.y + s.z * q.z + s.w * q.w) * scale;
            if (lane == 0) o_sh[j] = o;
        }
        __syncthreads();
        const float o = threadIdx.x < kDnState ? o_sh[threadIdx.x] : 0.0f;
        const float rs = 1.0f / sqrtf(block_sum(o * o, sh) / float(kDnState) + eps);
        if (threadIdx.x < kDnState) {
            const std::size_t i = std::size_t(t) * kDnVDim + h * kDnState + threadIdx.x;
            out[i] = ((o * rs) * norm_w[threadIdx.x]) * sigmoidf_(z[i]);
        }
        if (snap && t + 1 < T) {
            float4 * sp = reinterpret_cast<float4 *>(snap + (std::size_t(t) * kDnVHeads + h) * kDnState * kDnState);
#pragma unroll
            for (int r = 0; r < kDnRowsPerWarp; ++r) sp[(warp * kDnRowsPerWarp + r) * (kDnState / 4) + lane] = st[r];
        }
    }
#pragma unroll
    for (int r = 0; r < kDnRowsPerWarp; ++r) Sh[(warp * kDnRowsPerWarp + r) * (kDnState / 4) + lane] = st[r];
}

// Prompt chunks (no per-token snapshots): the same arithmetic as k_dn_conv and k_dn_recurrence, bitwise,
// with more of the GPU at work. The conv has no recurrence (each output reads the last 4 inputs), so tiles of
// tokens run in parallel; the delta rule is independent per state row, so a v head's 128 rows are spread
// over 16 blocks (a warp per 2 rows) and the RMSNorm of the head's outputs runs afterwards per token, over
// the same 128 values with the same reduction tree (the 512-thread block of k_dn_recurrence only adds zeros).
constexpr int kDnConvTile = 32;  // tokens per block
// NINFER_FN_SERIAL_DN=1 keeps the one-block-per-head kernels for prompts too (to check that both agree bitwise)
bool serial_dn() {
    static const bool v = std::getenv("NINFER_FN_SERIAL_DN") != nullptr;
    return v;
}

__global__ void __launch_bounds__(kDnState) k_dn_conv_tiles(const float * __restrict__ qkv, const float * __restrict__ state,
                                                            const float * __restrict__ w, float * __restrict__ out, int T, float eps) {
    __shared__ float sh[32];
    const int c = blockIdx.x * kDnState + threadIdx.x;
    const bool qk = blockIdx.x < 2 * kDnKHeads;
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    auto in = [&](int t) { return t >= 0 ? qkv[std::size_t(t) * kDnConvDim + c] : state[std::size_t(3 + t) * kDnConvDim + c]; };
    const int t0 = blockIdx.y * kDnConvTile, t1 = min(T, t0 + kDnConvTile);
    float s0 = in(t0 - 3), s1 = in(t0 - 2), s2 = in(t0 - 1);
    for (int t = t0; t < t1; ++t) {
        const float x = qkv[std::size_t(t) * kDnConvDim + c];
        float sum = 0.0f;
        sum += s0 * w0;
        sum += s1 * w1;
        sum += s2 * w2;
        sum += x * w3;
        float y = siluf_(sum);
        if (qk) y *= 1.0f / fmaxf(sqrtf(block_sum(y * y, sh)), eps);
        out[std::size_t(t) * kDnConvDim + c] = y;
        s0 = s1;
        s1 = s2;
        s2 = x;
    }
}

// the conv state after the chunk: its last three inputs (from the old state when T < 3); after every reader
__global__ void k_dn_conv_state(const float * __restrict__ qkv, float * __restrict__ state, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kDnConvDim) return;
    float v[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int t = T - 3 + j;
        v[j] = t >= 0 ? qkv[std::size_t(t) * kDnConvDim + c] : state[std::size_t(3 + t) * kDnConvDim + c];
    }
#pragma unroll
    for (int j = 0; j < 3; ++j) state[std::size_t(j) * kDnConvDim + c] = v[j];
}

constexpr int kDnRowWarpRows = 2, kDnRowWarps = 4;
constexpr int kDnRowBlocks = kDnState / (kDnRowWarpRows * kDnRowWarps);  // per v head

// out[t][h][j] = the raw output row j of head h (normalized by k_dn_norm_gate)
__global__ void __launch_bounds__(32 * kDnRowWarps) k_dn_rows(const float * __restrict__ conv, const float * __restrict__ beta,
                                                              const float * __restrict__ alpha, const float * __restrict__ dt,
                                                              const float * __restrict__ a, float * __restrict__ S, float * __restrict__ out, int T) {
    const int h = blockIdx.x, hk = h % kDnKHeads;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int j0 = (blockIdx.y * kDnRowWarps + warp) * kDnRowWarpRows;
    float4 * Sh = reinterpret_cast<float4 *>(S + std::size_t(h) * kDnState * kDnState);
    float4 st[kDnRowWarpRows];
#pragma unroll
    for (int r = 0; r < kDnRowWarpRows; ++r) st[r] = Sh[(j0 + r) * (kDnState / 4) + lane];
    const float scale = 1.0f / sqrtf(float(kDnState));
    const float dth = dt[h], ah = a[h];
    // the next token's inputs load while this one's dependent chain runs
    auto load = [&](int t, float4 & q, float4 & k, float (&v)[kDnRowWarpRows], float & be, float & al) {
        const float * row = conv + std::size_t(t) * kDnConvDim;
        q = reinterpret_cast<const float4 *>(row + hk * kDnState)[lane];
        k = reinterpret_cast<const float4 *>(row + kDnKeyDim + hk * kDnState)[lane];
#pragma unroll
        for (int r = 0; r < kDnRowWarpRows; ++r) v[r] = row[2 * kDnKeyDim + h * kDnState + j0 + r];
        be = beta[t * kDnVHeads + h];
        al = alpha[t * kDnVHeads + h];
    };
    float4 q, k, qn, kn;
    float v[kDnRowWarpRows], vn[kDnRowWarpRows], be, al, ben = 0.f, aln = 0.f;
    load(0, q, k, v, be, al);
    for (int t = 0; t < T; ++t) {
        if (t + 1 < T) load(t + 1, qn, kn, vn, ben, aln);
        const float b = sigmoidf_(be);
        const float decay = expf(softplusf_(al + dth) * ah);
#pragma unroll
        for (int r = 0; r < kDnRowWarpRows; ++r) {
            float4 & s = st[r];
            s.x *= decay; s.y *= decay; s.z *= decay; s.w *= decay;
            const float sk = warp_sum(s.x * k.x + s.y * k.y + s.z * k.z + s.w * k.w);
            const int j = j0 + r;
            const float d = (v[r] - sk) * b;
            s.x += k.x * d; s.y += k.y * d; s.z += k.z * d; s.w += k.w * d;
            const float o = warp_sum(s.x * q.x + s.y * q.y + s.z * q.z + s.w * q.w) * scale;
            if (lane == 0) out[std::size_t(t) * kDnVDim + h * kDnState + j] = o;
        }
        q = qn;
        k = kn;
#pragma unroll
        for (int r = 0; r < kDnRowWarpRows; ++r) v[r] = vn[r];
        be = ben;
        al = aln;
    }
#pragma unroll
    for (int r = 0; r < kDnRowWarpRows; ++r) Sh[(j0 + r) * (kDnState / 4) + lane] = st[r];
}

// out = RMSNorm(o) * norm_w * sigmoid(z) in place, per (token, v head)
__global__ void __launch_bounds__(kDnState) k_dn_norm_gate(const float * __restrict__ z, const float * __restrict__ norm_w,
                                                           float * __restrict__ out, float eps) {
    __shared__ float sh[32];
    const std::size_t i = std::size_t(blockIdx.x) * kDnVDim + blockIdx.y * kDnState + threadIdx.x;
    const float o = out[i];
    const float rs = 1.0f / sqrtf(block_sum(o * o, sh) / float(kDnState) + eps);
    out[i] = ((o * rs) * norm_w[threadIdx.x]) * sigmoidf_(z[i]);
}

// ------------------------------------------------------------------------------------------------
// full attention

// grid (T, 24 q heads + 2 kv heads), one thread per dim
__global__ void __launch_bounds__(kHeadDim) k_attn_prep(const float * __restrict__ q_full, const float * __restrict__ k,
                                                        const float * __restrict__ v, const float * __restrict__ q_norm,
                                                        const float * __restrict__ k_norm, const double * __restrict__ inv_freq,
                                                        float * __restrict__ q, float * __restrict__ gate, half * __restrict__ k_cache,
                                                        half * __restrict__ v_cache, const std::int64_t * __restrict__ pos0p, float eps) {
    __shared__ float ys[kHeadDim];
    __shared__ float sh[32];
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    const std::int64_t pos = *pos0p + t;
    const bool is_q = hh < kHeads;
    const int hk = hh - kHeads;
    const float x = is_q ? q_full[(std::size_t(t) * kHeads + hh) * 2 * kHeadDim + d] : k[(std::size_t(t) * kKvHeads + hk) * kHeadDim + d];
    const float scale = 1.0f / sqrtf(block_sum(x * x, sh) / float(kHeadDim) + eps);
    ys[d] = (x * scale) * (is_q ? q_norm : k_norm)[d];
    __syncthreads();
    float y0 = 0.0f, y1 = 0.0f;
    if (d < kRot / 2) {
        const double theta = double(pos) * inv_freq[d];
        const float c = float(cos(theta)), s = float(sin(theta));
        const float x0 = ys[d], x1 = ys[d + kRot / 2];
        y0 = x0 * c - x1 * s;
        y1 = x0 * s + x1 * c;
    }
    __syncthreads();
    if (d < kRot / 2) {
        ys[d] = y0;
        ys[d + kRot / 2] = y1;
    }
    __syncthreads();
    if (is_q) {
        const std::size_t o = (std::size_t(t) * kHeads + hh) * kHeadDim + d;
        q[o] = ys[d];
        gate[o] = q_full[(std::size_t(t) * kHeads + hh) * 2 * kHeadDim + kHeadDim + d];
    } else {
        const std::size_t o = (std::size_t(pos) * kKvHeads + hk) * kHeadDim + d;
        k_cache[o] = __float2half_rn(ys[d]);
        v_cache[o] = __float2half_rn(v[(std::size_t(t) * kKvHeads + hk) * kHeadDim + d]);
    }
}

constexpr int kAttnChunk = 256;                // cells per block
constexpr int kAttnStride = kHeadDim + 2;      // per (chunk, kv head, query): max, sum, acc[256]

// Partial attention over one chunk of cells for one kv head and its 12 x T queries.
template <int T>
__global__ void __launch_bounds__(kAttnChunk) k_attn_partial(const float * __restrict__ q, const half * __restrict__ k_cache,
                                                             const half * __restrict__ v_cache, const std::int64_t * __restrict__ pos0p,
                                                             float scale, float * __restrict__ work) {
    constexpr int Q = kGroup * T;
    extern __shared__ float smem[];
    float * qs = smem;                   // [Q][256]
    float * sc = smem + Q * kHeadDim;    // [256 cells][Q]
    const int ch = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const std::int64_t pos0 = *pos0p, n_kv = pos0 + T;
    if (std::int64_t(ch) * kAttnChunk >= n_kv) return;
    for (int i = tid; i < Q * kHeadDim; i += kAttnChunk) {
        const int qi = i / kHeadDim, tt = qi / kGroup, g = qi % kGroup;
        qs[i] = q[(std::size_t(tt) * kHeads + hk * kGroup + g) * kHeadDim + i % kHeadDim];
    }
    __syncthreads();

    const std::int64_t cell = std::int64_t(ch) * kAttnChunk + tid;
    float acc[Q];
#pragma unroll
    for (int i = 0; i < Q; ++i) acc[i] = 0.0f;
    if (cell < n_kv) {
        const uint4 * kr = reinterpret_cast<const uint4 *>(k_cache + (std::size_t(cell) * kKvHeads + hk) * kHeadDim);
        for (int d8 = 0; d8 < kHeadDim / 8; ++d8) {
            const uint4 raw = kr[d8];
            const half2 * hp = reinterpret_cast<const half2 *>(&raw);
            float kf[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 f = __half22float2(hp[j]);
                kf[2 * j] = f.x;
                kf[2 * j + 1] = f.y;
            }
#pragma unroll
            for (int i = 0; i < Q; ++i) {
                const float4 a = *reinterpret_cast<const float4 *>(qs + i * kHeadDim + d8 * 8);
                const float4 b = *reinterpret_cast<const float4 *>(qs + i * kHeadDim + d8 * 8 + 4);
                acc[i] += kf[0] * a.x + kf[1] * a.y + kf[2] * a.z + kf[3] * a.w + kf[4] * b.x + kf[5] * b.y + kf[6] * b.z + kf[7] * b.w;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < Q; ++i) {
        const bool visible = cell < n_kv && cell <= pos0 + i / kGroup;
        sc[tid * Q + i] = visible ? acc[i] * scale : -INFINITY;
    }
    __syncthreads();

    float * wbase = work + (std::size_t(ch) * kKvHeads + hk) * Q * kAttnStride;
    for (int i = warp; i < Q; i += kAttnChunk / 32) {
        float mx = -INFINITY;
        for (int c = lane; c < kAttnChunk; c += 32) mx = fmaxf(mx, sc[c * Q + i]);
        mx = warp_max(mx);
        float sum = 0.0f;
        for (int c = lane; c < kAttnChunk; c += 32) {
            const float p = mx == -INFINITY ? 0.0f : expf(sc[c * Q + i] - mx);
            sc[c * Q + i] = p;
            sum += p;
        }
        sum = warp_sum(sum);
        if (lane == 0) {
            wbase[i * kAttnStride] = mx;
            wbase[i * kAttnStride + 1] = sum;
        }
    }
    __syncthreads();

    float o[Q];
#pragma unroll
    for (int i = 0; i < Q; ++i) o[i] = 0.0f;
    const std::int64_t left = n_kv - std::int64_t(ch) * kAttnChunk;
    const int n_cells = left < kAttnChunk ? int(left) : kAttnChunk;
    for (int c = 0; c < n_cells; ++c) {
        const float vv = __half2float(v_cache[((std::size_t(ch) * kAttnChunk + c) * kKvHeads + hk) * kHeadDim + tid]);
#pragma unroll
        for (int i = 0; i < Q; ++i) o[i] += sc[c * Q + i] * vv;
    }
#pragma unroll
    for (int i = 0; i < Q; ++i) wbase[i * kAttnStride + 2 + tid] = o[i];
}

__global__ void __launch_bounds__(kHeadDim) k_attn_combine(const float * __restrict__ work, const float * __restrict__ gate,
                                                           const std::int64_t * __restrict__ pos0p, int T, float * __restrict__ out) {
    const int tt = blockIdx.x / kHeads, h = blockIdx.x % kHeads, d = threadIdx.x;
    const int n_chunks = int((*pos0p + T + kAttnChunk - 1) / kAttnChunk);
    const int hk = h / kGroup, Q = kGroup * T, qi = tt * kGroup + h % kGroup;
    float M = -INFINITY;
    for (int ch = 0; ch < n_chunks; ++ch) M = fmaxf(M, work[((std::size_t(ch) * kKvHeads + hk) * Q + qi) * kAttnStride]);
    float L = 0.0f, acc = 0.0f;
    for (int ch = 0; ch < n_chunks; ++ch) {
        const float * w = work + ((std::size_t(ch) * kKvHeads + hk) * Q + qi) * kAttnStride;
        if (w[0] == -INFINITY) continue;
        const float f = expf(w[0] - M);
        L += w[1] * f;
        acc += w[2 + d] * f;
    }
    const std::size_t o = (std::size_t(tt) * kHeads + h) * kHeadDim + d;
    out[o] = (acc / L) * sigmoidf_(gate[o]);
}

// ------------------------------------------------------------------------------------------------
// MoE

__global__ void __launch_bounds__(kExperts) k_router_topk(const float * __restrict__ logits, std::int32_t * __restrict__ ids,
                                                          float * __restrict__ weights) {
    __shared__ float sh[32];
    __shared__ float best_v[32];
    __shared__ int best_i[32];
    __shared__ float sel_p[kUsed];
    const int t = blockIdx.x, e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const float l = logits[t * kExperts + e];
    const float mx = block_max(l, sh);
    float p = expf(l - mx);
    const float sum = block_sum(p, sh);
    p *= 1.0f / sum;
    float cand = p;
    for (int k = 0; k < kUsed; ++k) {
        // argmax with ties to the lower expert id
        float v = cand;
        int idx = e;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (lane == 0) { best_v[warp] = v; best_i[warp] = idx; }
        __syncthreads();
        if (warp == 0) {
            v = lane < kExperts / 32 ? best_v[lane] : -INFINITY;
            idx = lane < kExperts / 32 ? best_i[lane] : kExperts;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, o);
                const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
                if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
            }
            if (lane == 0) {
                ids[t * kUsed + k] = idx;
                sel_p[k] = v;
                best_i[0] = idx;
            }
        }
        __syncthreads();
        if (e == best_i[0]) cand = -INFINITY;
        __syncthreads();
    }
    if (e == 0) {
        float wsum = 0.0f;
        for (int k = 0; k < kUsed; ++k) wsum += sel_p[k];
        wsum = fmaxf(wsum, 6.103515625e-5f);
        for (int k = 0; k < kUsed; ++k) weights[t * kUsed + k] = sel_p[k] / wsum;
    }
}

__global__ void k_swiglu(const float * __restrict__ g, const float * __restrict__ u, float * __restrict__ h, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = siluf_(g[i]) * u[i];
}

__global__ void k_ffn_combine(const float * __restrict__ moe, const float * __restrict__ shared, const float * __restrict__ sg,
                              float * __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = moe[i] + shared[i] * sigmoidf_(sg[i / kEmbd]);
}

__global__ void k_ple_hist_rebuild(const float * __restrict__ prev, const float * __restrict__ normalized, float * __restrict__ hist,
                                   int n_keep) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= kHcd) return;
    for (int j = 0; j < kPleHist; ++j) {
        const int r = n_keep - kPleHist + j;  // row of [prev | normalized] relative to the step's first token
        hist[std::size_t(j) * kHcd + ch] = r < 0 ? prev[std::size_t(kPleHist + r) * kHcd + ch] : normalized[std::size_t(r) * kHcd + ch];
    }
}

__global__ void __launch_bounds__(256) k_rms_norm_rows(const float * __restrict__ x, const float * __restrict__ w, float * __restrict__ y,
                                                       int n, float eps) {
    __shared__ float sh[32];
    const float * xr = x + std::size_t(blockIdx.x) * n;
    float * yr = y + std::size_t(blockIdx.x) * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += 256) ss += xr[i] * xr[i];
    const float scale = 1.0f / sqrtf(block_sum(ss, sh) / float(n) + eps);
    for (int i = threadIdx.x; i < n; i += 256) yr[i] = (xr[i] * scale) * w[i];
}

__global__ void k_mtp_concat(const float * __restrict__ e, const float * __restrict__ h, float * __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int t = i / (kHc * 2 * kEmbd), rem = i % (kHc * 2 * kEmbd), c = rem / (2 * kEmbd), j = rem % (2 * kEmbd);
    out[i] = j < kEmbd ? e[std::size_t(t) * kEmbd + j] : h[(std::size_t(t) * kHc + c) * kEmbd + (j - kEmbd)];
}

__global__ void k_window_cells(const std::int64_t * __restrict__ pos0p, int width, std::int32_t * __restrict__ cells,
                               std::int32_t * __restrict__ n_cells) {
    const std::int64_t p = *pos0p + blockIdx.x;
    const std::int64_t n = p + 1 < width ? p + 1 : width, first = p + 1 - n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) cells[std::size_t(blockIdx.x) * width + i] = std::int32_t(first + i);
    if (threadIdx.x == 0) n_cells[blockIdx.x] = std::int32_t(n);
}

__global__ void __launch_bounds__(1024) k_argmax(const float * __restrict__ x, int n, std::int32_t * __restrict__ out) {
    __shared__ float bv[32];
    __shared__ int bi[32];
    float v = -INFINITY;
    int idx = n;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > v) { v = x[i]; idx = i; }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    if (lane == 0) { bv[warp] = v; bi[warp] = idx; }
    __syncthreads();
    if (warp == 0) {
        v = lane < int(blockDim.x / 32) ? bv[lane] : -INFINITY;
        idx = lane < int(blockDim.x / 32) ? bi[lane] : n;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (lane == 0) *out = idx;
    }
}

__global__ void k_store_rows(const float * __restrict__ src, float * __restrict__ dst, const std::int64_t * __restrict__ pos0p, int row,
                             int n, std::int64_t ring) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[std::size_t((*pos0p + i / row) % ring) * row + i % row] = src[i];
}

__device__ __forceinline__ std::uint64_t global_ns() {
    std::uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void __launch_bounds__(256) k_link_signal(const std::int32_t * __restrict__ ids, const float * __restrict__ w,
                                                     const std::uint8_t * __restrict__ on_cpu, const float * __restrict__ x, int T,
                                                     ExpertLink * link, const std::int64_t * seq) {
    for (int i = threadIdx.x; i < T * kUsed; i += blockDim.x) {
        link->ids[i] = ids[i];
        link->weights[i] = w[i];
        link->on_cpu[i] = on_cpu[i];
    }
    for (int i = threadIdx.x; i < T * kEmbd; i += blockDim.x) link->x[i] = x[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *reinterpret_cast<volatile std::int64_t *>(&link->req) = *seq;
}

__global__ void __launch_bounds__(256) k_link_wait(ExpertLink * link, const std::int64_t * seq, float * __restrict__ out, int T,
                                                   int * error) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const std::int64_t want = *seq;
        const std::uint64_t t0 = global_ns();
        ok = 1;
        while (*reinterpret_cast<volatile std::int64_t *>(&link->done) != want) {
            if (global_ns() - t0 > 1500000000ull) {  // under Windows' 2 s GPU watchdog
                ok = 0;
                atomicExch(error, 1);
                break;
            }
            __nanosleep(200);
        }
        __threadfence_system();
    }
    __syncthreads();
    const volatile float * src = link->out;
    for (int i = threadIdx.x; i < T * kEmbd; i += blockDim.x) out[i] = ok ? src[i] : 0.0f;
}

__global__ void k_moe_slots(const std::int32_t * __restrict__ ids, const std::int32_t * __restrict__ map, std::int32_t * __restrict__ slots,
                            int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) slots[i] = map[ids[i]];
}

__global__ void k_moe_plan(const std::int32_t * __restrict__ ids, const std::int32_t * __restrict__ map, std::int32_t * __restrict__ slots,
                           std::int32_t * __restrict__ host_slots, std::uint8_t * __restrict__ on_cpu, int n, int zc_permille) {
    if (threadIdx.x != 0) return;
    int misses = 0;
    for (int i = 0; i < n; ++i) misses += map[ids[i]] < 0;
    const int to_gpu = (misses * zc_permille + 500) / 1000;
    for (int i = 0, m = 0; i < n; ++i) {
        const int slot = map[ids[i]];
        slots[i] = slot;
        const bool miss = slot < 0, zc = miss && m < to_gpu;
        m += miss;
        host_slots[i] = zc ? ids[i] : -1;
        on_cpu[i] = miss && !zc;
    }
}

__global__ void k_moe_combine_sum(const float * __restrict__ gpu, const float * __restrict__ cpu, const float * __restrict__ shared,
                                  const float * __restrict__ sg, float * __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (gpu[i] + (cpu ? cpu[i] : 0.0f)) + shared[i] * sigmoidf_(sg[i / kEmbd]);
}

__global__ void k_moe_combine(const float * __restrict__ pairs, const float * __restrict__ hpairs, const float * __restrict__ cpu,
                              const float * __restrict__ shared, const float * __restrict__ sg, float * __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int t = i / kEmbd, r = i % kEmbd;
    float m = pairs[std::size_t(t * kUsed) * kEmbd + r];
#pragma unroll
    for (int k = 1; k < kUsed; ++k) m = m + pairs[std::size_t(t * kUsed + k) * kEmbd + r];
    if (hpairs)
#pragma unroll
        for (int k = 0; k < kUsed; ++k) m = m + hpairs[std::size_t(t * kUsed + k) * kEmbd + r];
    out[i] = (m + cpu[i]) + shared[i] * sigmoidf_(sg[t]);
}

}  // namespace

// ------------------------------------------------------------------------------------------------

void hc_expand(const float * x, float * res, int T, cudaStream_t s) {
    k_hc_expand<<<blocks(std::size_t(T) * kHcd, 256), 256, 0, s>>>(x, res, T * kHcd);
    launched("hc_expand");
}
void hc_norm(const float * res, const float * w, float * xn, int T, float eps, cudaStream_t s) {
    k_hc_norm<<<T * kHc, 256, 0, s>>>(res, w, xn, eps);
    launched("hc_norm");
}
void hc_lowrank_act(float * lo, int n, cudaStream_t s) {
    k_hc_lowrank_act<<<blocks(n, 256), 256, 0, s>>>(lo, n);
    launched("hc_lowrank_act");
}
void hc_gate_mean(const float * xn, const float * gate, float * mixed, int T, cudaStream_t s) {
    k_hc_gate_mean<<<blocks(std::size_t(T) * kEmbd, 256), 256, 0, s>>>(xn, gate, mixed, T * kEmbd);
    launched("hc_gate_mean");
}
void hc_combine(float * res, const float * out, const float * inject, int T, cudaStream_t s) {
    k_hc_combine<<<blocks(std::size_t(T) * kHcd, 256), 256, 0, s>>>(res, out, inject, T * kHcd);
    launched("hc_combine");
}

void ple_gate(const float * key, const float * value, const float * res, const float * norm_key, const float * norm_query,
              const float * norm_conv, float * gated, float * normalized, float * gates, int T, float eps, cudaStream_t s) {
    k_ple_gate<<<T * kHc, 256, 0, s>>>(key, value, res, norm_key, norm_query, norm_conv, gated, normalized, gates, eps);
    launched("ple_gate");
}
void ple_conv_add(float * res, const float * gated, const float * normalized, const float * conv_w, float * hist, int T,
                  cudaStream_t s) {
    if (T < 1) throw std::runtime_error("ple_conv_add: T out of range");
    k_ple_conv_add<<<blocks(kHcd, 256), 256, 0, s>>>(res, gated, normalized, conv_w, hist, T);
    launched("ple_conv_add");
}

void dn_conv(const float * qkv, float * conv_state, const float * conv_w, float * out, int T, float eps, cudaStream_t s, float * snap) {
    if (!snap && T > kDnConvTile && !serial_dn()) {  // prompt chunks: tiles of tokens in parallel (bitwise the same)
        k_dn_conv_tiles<<<dim3(kDnConvDim / kDnState, unsigned((T + kDnConvTile - 1) / kDnConvTile)), kDnState, 0, s>>>(qkv, conv_state, conv_w,
                                                                                                                 out, T, eps);
        launched("dn_conv_tiles");
        k_dn_conv_state<<<blocks(kDnConvDim, 256), 256, 0, s>>>(qkv, conv_state, T);
        launched("dn_conv_state");
        return;
    }
    k_dn_conv<<<kDnConvDim / kDnState, kDnState, 0, s>>>(qkv, conv_state, conv_w, out, T, eps, snap);
    launched("dn_conv");
}
void dn_recurrence(const float * conv_out, const float * z, const float * beta, const float * alpha, const float * dt_bias,
                   const float * a, const float * norm_w, float * S, float * out, int T, float eps, cudaStream_t s, float * snap) {
    if (!snap && T > kMaxTokens && !serial_dn()) {  // prompt chunks: state rows in parallel, then the norm (bitwise the same)
        k_dn_rows<<<dim3(kDnVHeads, kDnRowBlocks), 32 * kDnRowWarps, 0, s>>>(conv_out, beta, alpha, dt_bias, a, S, out, T);
        launched("dn_rows");
        k_dn_norm_gate<<<dim3(unsigned(T), kDnVHeads), kDnState, 0, s>>>(z, norm_w, out, eps);
        launched("dn_norm_gate");
        return;
    }
    k_dn_recurrence<<<kDnVHeads, kDnThreads, 0, s>>>(conv_out, z, beta, alpha, dt_bias, a, norm_w, S, out, T, eps, snap);
    launched("dn_recurrence");
}

void attn_prep(const float * q_full, const float * k, const float * v, const float * q_norm, const float * k_norm,
               const double * rope_inv_freq, float * q, float * gate, half * k_cache, half * v_cache, const std::int64_t * pos0, int T,
               float eps, cudaStream_t s) {
    k_attn_prep<<<dim3(T, kHeads + kKvHeads), kHeadDim, 0, s>>>(q_full, k, v, q_norm, k_norm, rope_inv_freq, q, gate, k_cache, v_cache,
                                                                 pos0, eps);
    launched("attn_prep");
}

std::size_t attn_work_floats(int T) { return std::size_t(kAttnMaxChunks) * kKvHeads * kGroup * T * kAttnStride; }

template <int T> constexpr int attn_smem() { return 2 * kGroup * T * kHeadDim * int(sizeof(float)); }

template <int T>
static void attn_launch(const float * q, const half * k_cache, const half * v_cache, const std::int64_t * pos0, float scale, float * work,
                        cudaStream_t s) {
    k_attn_partial<T><<<dim3(kAttnMaxChunks, kKvHeads), kAttnChunk, attn_smem<T>(), s>>>(q, k_cache, v_cache, pos0, scale, work);
}

void attn_decode(const float * q, const float * gate, const half * k_cache, const half * v_cache, const std::int64_t * pos0, int T,
                 float scale, float * work, float * out, cudaStream_t s) {
    switch (T) {
    case 1: attn_launch<1>(q, k_cache, v_cache, pos0, scale, work, s); break;
    case 2: attn_launch<2>(q, k_cache, v_cache, pos0, scale, work, s); break;
    case 3: attn_launch<3>(q, k_cache, v_cache, pos0, scale, work, s); break;
    case 4: attn_launch<4>(q, k_cache, v_cache, pos0, scale, work, s); break;
    default: throw std::runtime_error("attn_decode: T must be 1..4");
    }
    launched("attn_partial");
    k_attn_combine<<<T * kHeads, kHeadDim, 0, s>>>(work, gate, pos0, T, out);
    launched("attn_combine");
}

void store_rows(const float * src, float * dst, const std::int64_t * pos0, int row, int T, std::int64_t ring, cudaStream_t s) {
    k_store_rows<<<blocks(std::size_t(T) * row, 256), 256, 0, s>>>(src, dst, pos0, row, T * row, ring);
    launched("store_rows");
}

void link_signal(const std::int32_t * ids, const float * weights, const std::uint8_t * on_cpu, const float * x, int T, ExpertLink * link,
                 const std::int64_t * seq, cudaStream_t s) {
    k_link_signal<<<1, 256, 0, s>>>(ids, weights, on_cpu, x, T, link, seq);
    launched("link_signal");
}
void link_wait(ExpertLink * link, const std::int64_t * seq, float * out, int T, int * error, cudaStream_t s) {
    k_link_wait<<<1, 256, 0, s>>>(link, seq, out, T, error);
    launched("link_wait");
}

void ple_hist_rebuild(const float * prev, const float * normalized, float * hist, int n_keep, cudaStream_t s) {
    k_ple_hist_rebuild<<<blocks(kHcd, 256), 256, 0, s>>>(prev, normalized, hist, n_keep);
    launched("ple_hist_rebuild");
}
void rms_norm_rows(const float * x, const float * w, float * y, int rows, int n, float eps, cudaStream_t s) {
    k_rms_norm_rows<<<rows, 256, 0, s>>>(x, w, y, n, eps);
    launched("rms_norm_rows");
}
void mtp_concat(const float * e, const float * h, float * out, int T, cudaStream_t s) {
    const int n = T * kHc * 2 * kEmbd;
    k_mtp_concat<<<blocks(n, 256), 256, 0, s>>>(e, h, out, n);
    launched("mtp_concat");
}
void window_cells(const std::int64_t * pos0, int T, int width, std::int32_t * cells, std::int32_t * n_cells, cudaStream_t s) {
    k_window_cells<<<T, 256, 0, s>>>(pos0, width, cells, n_cells);
    launched("window_cells");
}
void argmax(const float * x, int n, std::int32_t * out, cudaStream_t s) {
    k_argmax<<<1, 1024, 0, s>>>(x, n, out);
    launched("argmax");
}

void init_kernels() {
    check(cudaFuncSetAttribute(k_attn_partial<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem<1>()), "attn smem");
    check(cudaFuncSetAttribute(k_attn_partial<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem<2>()), "attn smem");
    check(cudaFuncSetAttribute(k_attn_partial<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem<3>()), "attn smem");
    check(cudaFuncSetAttribute(k_attn_partial<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, attn_smem<4>()), "attn smem");
    experts_init();
}

void router_topk(const float * logits, std::int32_t * ids, float * weights, int T, cudaStream_t s) {
    k_router_topk<<<T, kExperts, 0, s>>>(logits, ids, weights);
    launched("router_topk");
}
void swiglu(const float * g, const float * u, float * h, int n, cudaStream_t s) {
    k_swiglu<<<blocks(n, 256), 256, 0, s>>>(g, u, h, n);
    launched("swiglu");
}
void ffn_combine(const float * moe, const float * shared, const float * shared_gate, float * out, int T, cudaStream_t s) {
    k_ffn_combine<<<blocks(std::size_t(T) * kEmbd, 256), 256, 0, s>>>(moe, shared, shared_gate, out, T * kEmbd);
    launched("ffn_combine");
}

// unit-sized words over the grid, then the tail bytes
template <class U>
__global__ void k_copy_sm(std::uint8_t * __restrict__ dst, const std::uint8_t * __restrict__ src, std::size_t words, std::size_t bytes) {
    const std::size_t stride = std::size_t(gridDim.x) * blockDim.x, first = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    for (std::size_t i = first; i < words; i += stride) reinterpret_cast<U *>(dst)[i] = reinterpret_cast<const U *>(src)[i];
    for (std::size_t i = words * sizeof(U) + first; i < bytes; i += stride) dst[i] = src[i];
}

void copy_sm(void * dst, const void * src, std::size_t bytes, cudaStream_t s) {
    if (!bytes) return;
    const std::uintptr_t a = reinterpret_cast<std::uintptr_t>(dst) | reinterpret_cast<std::uintptr_t>(src);
    auto * d = static_cast<std::uint8_t *>(dst);
    const auto * sp = static_cast<const std::uint8_t *>(src);
    auto grid = [](std::size_t words) { return unsigned(std::min<std::size_t>(std::max<std::size_t>(1, (words + 255) / 256), 1024)); };
    if (a % 16 == 0) k_copy_sm<uint4><<<grid(bytes / 16), 256, 0, s>>>(d, sp, bytes / 16, bytes);
    else if (a % 4 == 0) k_copy_sm<std::uint32_t><<<grid(bytes / 4), 256, 0, s>>>(d, sp, bytes / 4, bytes);
    else k_copy_sm<std::uint8_t><<<grid(bytes), 256, 0, s>>>(d, sp, bytes, bytes);
    launched("copy_sm");
}

void moe_slots(const std::int32_t * ids, const std::int32_t * map, std::int32_t * slots, int T, cudaStream_t s) {
    k_moe_slots<<<blocks(std::size_t(T) * kUsed, 256), 256, 0, s>>>(ids, map, slots, T * kUsed);
    launched("moe_slots");
}
void moe_plan(const std::int32_t * ids, const std::int32_t * map, std::int32_t * slots, std::int32_t * host_slots, std::uint8_t * on_cpu,
              int T, int zc_permille, cudaStream_t s) {
    k_moe_plan<<<1, 32, 0, s>>>(ids, map, slots, host_slots, on_cpu, T * kUsed, zc_permille);
    launched("moe_plan");
}
void moe_combine_sum(const float * gpu, const float * cpu, const float * shared, const float * shared_gate, float * out, int T,
                     cudaStream_t s) {
    k_moe_combine_sum<<<blocks(std::size_t(T) * kEmbd, 256), 256, 0, s>>>(gpu, cpu, shared, shared_gate, out, T * kEmbd);
    launched("moe_combine_sum");
}
void moe_combine(const float * gpu_pairs, const float * host_pairs, const float * cpu, const float * shared, const float * shared_gate,
                 float * out, int T, cudaStream_t s) {
    k_moe_combine<<<blocks(std::size_t(T) * kEmbd, 256), 256, 0, s>>>(gpu_pairs, host_pairs, cpu, shared, shared_gate, out, T * kEmbd);
    launched("moe_combine");
}

}  // namespace ninfer::flashnext::cuda
