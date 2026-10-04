#include "flashnext/cuda/ops.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "flashnext/cuda/device.h"

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
    float ss = 0.0f;
    for (int i = threadIdx.x; i < kEmbd; i += 256) ss += x[i] * x[i];
    ss = block_sum(ss, sh);
    const float scale = 1.0f / sqrtf(ss / float(kEmbd) + eps);
    for (int i = threadIdx.x; i < kEmbd; i += 256) y[i] = (x[i] * scale) * wc[i];
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
    float win[kPleHist + kMaxTokens];
    for (int j = 0; j < kPleHist; ++j) win[j] = hist[std::size_t(j) * kHcd + ch];
    for (int t = 0; t < T; ++t) win[kPleHist + t] = normalized[std::size_t(t) * kHcd + ch];
    float w[kPleKernel];
    for (int k = 0; k < kPleKernel; ++k) w[k] = conv_w[ch * kPleKernel + k];
    for (int t = 0; t < T; ++t) {
        float acc = 0.0f;
        for (int k = 0; k < kPleKernel; ++k) {
            const float term = win[kPleHist + t - (kPleKernel - 1 - k) * kPleDilation] * w[k];
            acc = k == 0 ? term : acc + term;
        }
        const std::size_t i = std::size_t(t) * kHcd + ch;
        res[i] = res[i] + (gated[i] + siluf_(acc));
    }
    for (int j = 0; j < kPleHist; ++j) hist[std::size_t(j) * kHcd + ch] = win[T + j];
}

// ------------------------------------------------------------------------------------------------
// Gated DeltaNet

// One block per head of 128 channels (32 q/k heads, then 48 v heads).
__global__ void __launch_bounds__(kDnState) k_dn_conv(const float * __restrict__ qkv, float * __restrict__ state, const float * __restrict__ w,
                                                      float * __restrict__ out, int T, float eps) {
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
                                                              float * __restrict__ out, int T, float eps) {
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
    }
#pragma unroll
    for (int r = 0; r < kDnRowsPerWarp; ++r) Sh[(warp * kDnRowsPerWarp + r) * (kDnState / 4) + lane] = st[r];
}

// ------------------------------------------------------------------------------------------------
// full attention

// grid (T, 24 q heads + 2 kv heads), one thread per dim
__global__ void __launch_bounds__(kHeadDim) k_attn_prep(const float * __restrict__ q_full, const float * __restrict__ k,
                                                        const float * __restrict__ v, const float * __restrict__ q_norm,
                                                        const float * __restrict__ k_norm, const double * __restrict__ inv_freq,
                                                        float * __restrict__ q, float * __restrict__ gate, half * __restrict__ k_cache,
                                                        half * __restrict__ v_cache, std::int64_t pos0, float eps) {
    __shared__ float ys[kHeadDim];
    __shared__ float sh[32];
    const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    const std::int64_t pos = pos0 + t;
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
                                                             const half * __restrict__ v_cache, std::int64_t pos0, float scale,
                                                             float * __restrict__ work) {
    constexpr int Q = kGroup * T;
    extern __shared__ float smem[];
    float * qs = smem;                   // [Q][256]
    float * sc = smem + Q * kHeadDim;    // [256 cells][Q]
    const int ch = blockIdx.x, hk = blockIdx.y, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const std::int64_t n_kv = pos0 + T;
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

__global__ void __launch_bounds__(kHeadDim) k_attn_combine(const float * __restrict__ work, const float * __restrict__ gate, int n_chunks,
                                                           int T, float * __restrict__ out) {
    const int tt = blockIdx.x / kHeads, h = blockIdx.x % kHeads, d = threadIdx.x;
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
    if (T < 1 || T > kMaxTokens) throw std::runtime_error("ple_conv_add: T out of range");
    k_ple_conv_add<<<blocks(kHcd, 256), 256, 0, s>>>(res, gated, normalized, conv_w, hist, T);
    launched("ple_conv_add");
}

void dn_conv(const float * qkv, float * conv_state, const float * conv_w, float * out, int T, float eps, cudaStream_t s) {
    k_dn_conv<<<kDnConvDim / kDnState, kDnState, 0, s>>>(qkv, conv_state, conv_w, out, T, eps);
    launched("dn_conv");
}
void dn_recurrence(const float * conv_out, const float * z, const float * beta, const float * alpha, const float * dt_bias,
                   const float * a, const float * norm_w, float * S, float * out, int T, float eps, cudaStream_t s) {
    k_dn_recurrence<<<kDnVHeads, kDnThreads, 0, s>>>(conv_out, z, beta, alpha, dt_bias, a, norm_w, S, out, T, eps);
    launched("dn_recurrence");
}

void attn_prep(const float * q_full, const float * k, const float * v, const float * q_norm, const float * k_norm,
               const double * rope_inv_freq, float * q, float * gate, half * k_cache, half * v_cache, std::int64_t pos0, int T,
               float eps, cudaStream_t s) {
    k_attn_prep<<<dim3(T, kHeads + kKvHeads), kHeadDim, 0, s>>>(q_full, k, v, q_norm, k_norm, rope_inv_freq, q, gate, k_cache, v_cache,
                                                                 pos0, eps);
    launched("attn_prep");
}

std::size_t attn_work_floats(std::int64_t max_ctx, int T) {
    return std::size_t((max_ctx + kAttnChunk - 1) / kAttnChunk) * kKvHeads * kGroup * T * kAttnStride;
}

template <int T>
static void attn_launch(const float * q, const half * k_cache, const half * v_cache, std::int64_t pos0, float scale, float * work,
                        int n_chunks, cudaStream_t s) {
    const int smem = 2 * kGroup * T * kHeadDim * int(sizeof(float));
    static bool configured = false;
    if (!configured) {
        check(cudaFuncSetAttribute(k_attn_partial<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem), "attn smem");
        configured = true;
    }
    k_attn_partial<T><<<dim3(n_chunks, kKvHeads), kAttnChunk, smem, s>>>(q, k_cache, v_cache, pos0, scale, work);
}

void attn_decode(const float * q, const float * gate, const half * k_cache, const half * v_cache, std::int64_t pos0, int T,
                 float scale, float * work, float * out, cudaStream_t s) {
    const int n_chunks = int((pos0 + T + kAttnChunk - 1) / kAttnChunk);
    switch (T) {
    case 1: attn_launch<1>(q, k_cache, v_cache, pos0, scale, work, n_chunks, s); break;
    case 2: attn_launch<2>(q, k_cache, v_cache, pos0, scale, work, n_chunks, s); break;
    case 3: attn_launch<3>(q, k_cache, v_cache, pos0, scale, work, n_chunks, s); break;
    case 4: attn_launch<4>(q, k_cache, v_cache, pos0, scale, work, n_chunks, s); break;
    default: throw std::runtime_error("attn_decode: T must be 1..4");
    }
    launched("attn_partial");
    k_attn_combine<<<T * kHeads, kHeadDim, 0, s>>>(work, gate, n_chunks, T, out);
    launched("attn_combine");
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

}  // namespace ninfer::flashnext::cuda
