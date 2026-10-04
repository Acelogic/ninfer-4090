#include "flashnext/cuda/qsa.h"

#include <cub/block/block_scan.cuh>

#include <algorithm>
#include <stdexcept>

#include "flashnext/cuda/device.h"
#include "flashnext/cuda/ops.h"

// The indexer arithmetic below is written with explicit __f*_rn intrinsics: nvcc would otherwise fuse
// a*b - c*d into an FMA, which the reference (MSVC, no contraction) does not, and a single rounding
// difference in a block score can change which blocks are selected.

namespace ninfer::flashnext::cuda {

namespace {

static_assert(kQsaDim == 128 && kQsaHeads == 4, "the warp layout below assumes 4 indexer heads of 128");

inline void launched(const char * what) { check(cudaGetLastError(), what); }

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// ------------------------------------------------------------------------------------------------
// RMSNorm and rotation of one 128-vector held one element per thread (blockDim.x == 128), exactly as
// the reference's rms_norm (double sum of float squares in index order, float mean, 1/sqrtf) and
// rope (double angle, float cos/sin, x0*c - x1*s and x0*s + x1*c without contraction).
__device__ float norm_rope_128(float x, const float * __restrict__ w, const double * __restrict__ inv_freq, std::int64_t pos,
                               float eps, float * xs, float * scale_s) {
    const int d = threadIdx.x;
    xs[d] = x;
    __syncthreads();
    if (d == 0) {
        double sum = 0.0;
        for (int i = 0; i < kQsaDim; ++i) sum += double(__fmul_rn(xs[i], xs[i]));
        const float mean = float(sum / double(kQsaDim));
        *scale_s = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    }
    __syncthreads();
    const float y = __fmul_rn(__fmul_rn(x, *scale_s), w[d]);
    __syncthreads();
    xs[d] = y;
    __syncthreads();
    float out = y;
    if (d < kQsaRot) {
        const int i = d % (kQsaRot / 2);
        const double theta = double(pos) * inv_freq[i];
        const float c = float(cos(theta)), s = float(sin(theta));
        const float x0 = xs[i], x1 = xs[i + kQsaRot / 2];
        out = d < kQsaRot / 2 ? __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s)) : __fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c));
    }
    return out;
}

// grid T/4 + 1: candidate block pos0/4 + blockIdx.x, updated only if this step completes it
__global__ void __launch_bounds__(kQsaDim) k_qsa_blocks(const float * __restrict__ raw, const float * __restrict__ k_norm,
                                                        const double * __restrict__ inv_freq, float * __restrict__ blocks,
                                                        const std::int64_t * __restrict__ pos0p, int T, float eps) {
    __shared__ float xs[kQsaDim];
    __shared__ float scale_s;
    const std::int64_t pos0 = *pos0p;
    const std::int64_t b = pos0 / kQsaRatio + blockIdx.x, last = b * kQsaRatio + kQsaRatio - 1;
    if (last < pos0 || last >= pos0 + T) return;
    const int d = threadIdx.x;
    const float * r = raw + std::size_t(b) * kQsaRatio * kQsaDim + d;
    float acc = r[0];
#pragma unroll
    for (int i = 1; i < kQsaRatio; ++i) acc = __fadd_rn(acc, r[i * kQsaDim]);
    const float pooled = __fmul_rn(acc, 1.0f / float(kQsaRatio));
    blocks[std::size_t(b) * kQsaDim + d] = norm_rope_128(pooled, k_norm, inv_freq, b * kQsaRatio, eps, xs, &scale_s);
}

// grid (T, 4 heads)
__global__ void __launch_bounds__(kQsaDim) k_qsa_query(float * __restrict__ q, const float * __restrict__ q_norm,
                                                       const double * __restrict__ inv_freq, const std::int64_t * __restrict__ pos0p,
                                                       float eps) {
    __shared__ float xs[kQsaDim];
    __shared__ float scale_s;
    float * row = q + (std::size_t(blockIdx.x) * kQsaHeads + blockIdx.y) * kQsaDim;
    const float y = norm_rope_128(row[threadIdx.x], q_norm, inv_freq, *pos0p + blockIdx.x, eps, xs, &scale_s);
    row[threadIdx.x] = y;
}

// ------------------------------------------------------------------------------------------------
// block scores: score[t][b] = sum_h relu(q[t][h] . blocks[b]) for every complete block b of every
// sparse query t, bit-identical to the reference's AVX-512 dot: two 16-lane accumulators over 32-element
// steps (here lane j of a warp accumulates elements j, 32+j, 64+j, 96+j with FMAs, so lanes l and l+16
// hold the reference's acc0[l] and acc1[l]), then acc0 + acc1 and _mm512_reduce_add_ps's halving tree
// (pairs 8, 4, 2, 1 apart). A warp handles one block row against the D = 4 * NQ query heads of NQ
// queries at once; tree_reduce performs the five tree levels for all D dots with a transposing
// exchange (at each level a lane keeps half of its dots and swaps the other half with its partner),
// which pairs the same lanes as the reference tree, so every dot rounds exactly as there.

constexpr int kScoreWarps = 8;
constexpr int kScoreRowsPerWarp = 32;
constexpr int kScoreRows = kScoreWarps * kScoreRowsPerWarp;  // block rows per CUDA block

__host__ __device__ constexpr int ilog2(int x) { return x <= 1 ? 0 : 1 + ilog2(x / 2); }

// Reduces c[0..D) over the warp, level by level (offsets 16, 8, 4, 2, 1). Afterwards this lane holds the
// full dot (lane >> (5 - log2 D)) in c[0].
template <int D> __device__ __forceinline__ void tree_reduce(float (&c)[D], int lane) {
    int n = D;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        if (n > 1) {
            const bool upper = (lane & o) != 0;
#pragma unroll
            for (int i = 0; i < D / 2; ++i) {
                if (i < n / 2) {
                    const float keep = upper ? c[i + n / 2] : c[i];
                    const float send = upper ? c[i] : c[i + n / 2];
                    c[i] = __fadd_rn(keep, __shfl_xor_sync(0xffffffffu, send, o));
                }
            }
            n /= 2;
        } else {
            c[0] = __fadd_rn(c[0], __shfl_xor_sync(0xffffffffu, c[0], o));
        }
    }
}

template <int NQ>
__global__ void __launch_bounds__(kScoreWarps * 32) k_qsa_scores(const float * __restrict__ q, const float * __restrict__ blocks,
                                                                 const std::int64_t * __restrict__ pos0p, int T,
                                                                 std::int64_t row_stride, float * __restrict__ scores) {
    constexpr int D = kQsaHeads * NQ;
    constexpr int SHIFT = 5 - ilog2(D);  // dot m ends in lanes (m << SHIFT) .. (m << SHIFT) + 2^SHIFT - 1
    const std::int64_t pos0 = *pos0p;
    const int t0 = blockIdx.y * NQ;
    const int nq = min(NQ, T - t0);
    const std::int64_t last_pos = pos0 + t0 + nq - 1;
    if (last_pos + 1 <= kQsaWidth) return;  // every query here is dense
    const std::int64_t nb_max = (last_pos + 1) / kQsaRatio;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const std::int64_t r0 = std::int64_t(blockIdx.x) * kScoreRows + std::int64_t(warp) * kScoreRowsPerWarp;
    if (r0 >= nb_max) return;
    const std::int64_t r1 = min(r0 + kScoreRowsPerWarp, nb_max);

    float qv[D][4];
#pragma unroll
    for (int m = 0; m < D; ++m) {
        const int qi = m / kQsaHeads;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            qv[m][k] = qi < nq ? q[(std::size_t(t0 + qi) * kQsaHeads + m % kQsaHeads) * kQsaDim + 32 * k + lane] : 0.0f;
        }
    }
    // the writer lane of query qi holds head 0 of it; heads 1..3 sit 1, 2, 3 << SHIFT lanes above
    const int my_q = lane >> (SHIFT + 2);
    const bool writer = (lane & ((4 << SHIFT) - 1)) == 0 && my_q < nq;
    const std::int64_t my_pos = pos0 + t0 + my_q;
    const std::int64_t my_nb = (writer && my_pos + 1 > kQsaWidth) ? (my_pos + 1) / kQsaRatio : 0;
    float * my_out = scores + std::size_t(t0 + (writer ? my_q : 0)) * row_stride;

    const float * kb = blocks + std::size_t(r0) * kQsaDim + lane;
    float k0 = kb[0], k1 = kb[32], k2 = kb[64], k3 = kb[96];
    for (std::int64_t row = r0; row < r1; ++row) {
        float n0 = 0.0f, n1 = 0.0f, n2 = 0.0f, n3 = 0.0f;
        if (row + 1 < r1) {
            const float * nk = blocks + std::size_t(row + 1) * kQsaDim + lane;
            n0 = nk[0];
            n1 = nk[32];
            n2 = nk[64];
            n3 = nk[96];
        }
        float c[D];
#pragma unroll
        for (int m = 0; m < D; ++m) {
            float a = __fmaf_rn(qv[m][0], k0, 0.0f);
            a = __fmaf_rn(qv[m][1], k1, a);
            a = __fmaf_rn(qv[m][2], k2, a);
            c[m] = __fmaf_rn(qv[m][3], k3, a);
        }
        tree_reduce<D>(c, lane);
        const float r = c[0] > 0.0f ? c[0] : 0.0f;
        const float h1 = __shfl_down_sync(0xffffffffu, r, 1 << SHIFT);
        const float h2 = __shfl_down_sync(0xffffffffu, r, 2 << SHIFT);
        const float h3 = __shfl_down_sync(0xffffffffu, r, 3 << SHIFT);
        if (row < my_nb) my_out[row] = __fadd_rn(__fadd_rn(__fadd_rn(r, h1), h2), h3);  // ((0 + r0) + r1) + r2) + r3
        k0 = n0;
        k1 = n1;
        k2 = n2;
        k3 = n3;
    }
}

// ------------------------------------------------------------------------------------------------
// selection: one CUDA block per query. A radix select (12 + 10 + 10 bits of the float bits; scores are
// >= +0 so the bits order like the values) finds u*, the K-th largest score with K = ceil(want / 4).
// Then every block above u* is taken, and the first `need` blocks equal to u* in block order, the last
// of which (the K-th) contributes only want % 4 cells when that is not 0. Each thread owns a contiguous
// range of blocks, so two block-wide scans order the output.

constexpr int kSelThreads = 1024;
constexpr int kSelBins = 4096;

using SelScan = cub::BlockScan<int, kSelThreads>;

__global__ void __launch_bounds__(kSelThreads) k_qsa_select(const float * __restrict__ scores, const std::int64_t * __restrict__ pos0p,
                                                            std::int64_t row_stride, std::int32_t * __restrict__ cells,
                                                            std::int32_t * __restrict__ n_cells) {
    __shared__ int hist[kSelBins];
    __shared__ typename SelScan::TempStorage scan_tmp;
    __shared__ unsigned found_digit;
    __shared__ int found_above;
    const int t = blockIdx.x, tid = threadIdx.x;
    const std::int64_t pos = *pos0p + t;
    std::int32_t * out = cells + std::size_t(t) * kQsaWidth;
    if (pos + 1 <= kQsaWidth) {  // dense
        for (int i = tid; i <= int(pos); i += kSelThreads) out[i] = i;
        if (tid == 0) n_cells[t] = std::int32_t(pos + 1);
        return;
    }
    const std::int64_t tail_start = (pos + 1) / kQsaRatio * kQsaRatio;
    const int nb = int(tail_start / kQsaRatio);
    const int tail = int(pos + 1 - tail_start);
    const int want = kQsaWidth - tail;
    const int K = (want + kQsaRatio - 1) / kQsaRatio, rem = want % kQsaRatio;
    // this thread's blocks: [begin, begin + per), a multiple of 4 so that rows load as uint4 (the score
    // rows are 64-float aligned and padded, so a group that starts below nb is inside the row)
    const int per = ((nb + kSelThreads - 1) / kSelThreads + 3) & ~3;
    const int begin = tid * per;
    const uint4 * row4 = reinterpret_cast<const uint4 *>(scores + std::size_t(t) * row_stride);
    auto load4 = [&](int i) { return i < nb ? row4[i / 4] : make_uint4(0u, 0u, 0u, 0u); };

    unsigned prefix = 0, prefix_mask = 0;
    int kr = K;
#pragma unroll 1
    for (int p = 0; p < 3; ++p) {
        const int shift = p == 0 ? 20 : (p == 1 ? 10 : 0), nbins = p == 0 ? 4096 : 1024;
        for (int i = tid; i < nbins; i += kSelThreads) hist[i] = 0;
        __syncthreads();
#pragma unroll 4
        for (int j = 0; j < per; j += 4) {
            const uint4 v4 = load4(begin + j);
            const unsigned vs[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const bool match = begin + j + e < nb && (vs[e] & prefix_mask) == prefix;
                if (match) atomicAdd(&hist[(vs[e] >> shift) & unsigned(nbins - 1)], 1);
            }
        }
        __syncthreads();
        // thread g owns the g-th group of bins counted from the top; an exclusive scan gives the
        // number of matching keys in higher bins
        const int bpt = nbins / kSelThreads;
        const int hi = nbins - tid * bpt;  // this thread's bins: [hi - bpt, hi)
        int local = 0;
        for (int j = 0; j < bpt; ++j) local += hist[hi - 1 - j];
        int above;
        SelScan(scan_tmp).ExclusiveSum(local, above);
        for (int j = 0; j < bpt; ++j) {
            const int bin = hi - 1 - j, c = hist[bin];
            if (above < kr && above + c >= kr) {
                found_digit = unsigned(bin);
                found_above = above;
            }
            above += c;
        }
        __syncthreads();
        prefix |= found_digit << shift;
        prefix_mask |= unsigned(nbins - 1) << shift;
        kr -= found_above;
        __syncthreads();
    }
    const unsigned ustar = prefix;
    const int need = kr;  // blocks equal to u* to take, in block order (>= 1)

    // ranks of the equal blocks, then output offsets, both in block order
    int n_eq = 0;
    for (int j = 0; j < per; j += 4) {
        const uint4 v4 = load4(begin + j);
        const unsigned vs[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) n_eq += (begin + j + e < nb && vs[e] == ustar) ? 1 : 0;
    }
    int eq_base;
    SelScan(scan_tmp).ExclusiveSum(n_eq, eq_base);
    __syncthreads();
    auto cells_of = [&](unsigned v, int idx, int & er) {
        if (idx >= nb) return 0;
        if (v > ustar) return kQsaRatio;
        if (v != ustar) return 0;
        const int r = er++;
        if (r >= need) return 0;
        return (r == need - 1 && rem != 0) ? rem : kQsaRatio;
    };
    int n_out = 0, er = eq_base;
    for (int j = 0; j < per; j += 4) {
        const uint4 v4 = load4(begin + j);
        const unsigned vs[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) n_out += cells_of(vs[e], begin + j + e, er);
    }
    int o;
    SelScan(scan_tmp).ExclusiveSum(n_out, o);
    if (n_out > 0) {
        er = eq_base;
        for (int j = 0; j < per; j += 4) {
            const uint4 v4 = load4(begin + j);
            const unsigned vs[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int idx = begin + j + e, nc = cells_of(vs[e], idx, er);
                for (int k = 0; k < nc; ++k) out[o + k] = idx * kQsaRatio + k;
                o += nc;
            }
        }
    }
    for (int j = tid; j < tail; j += kSelThreads) out[want + j] = std::int32_t(tail_start + j);
    if (tid == 0) n_cells[t] = std::int32_t(want + tail);
}

// ------------------------------------------------------------------------------------------------
// attention over the selected cells. A CUDA block handles one (token, kv head, split): the 12 query
// heads of that kv head over chunks split, split + n_split, ... of 64 cells, with a running softmax;
// k_attn_sparse_combine merges the splits and applies the gate. In a chunk, four threads share a cell
// for q . k (each takes every fourth 8-dim group of the row, so they read 64 contiguous bytes at a
// time), and every thread owns one of the 256 dims for p . v.

constexpr int kSpThreads = 256;
constexpr int kSpChunk = 64;                                           // cells per chunk
constexpr int kSpMaxChunks = (kQsaWidth + kSpChunk - 1) / kSpChunk;   // 33
constexpr int kSpStride = kHeadDim + 2;                                // m, l, acc[256]

int sparse_splits(int T) {
    const int s = (512 + 2 * T - 1) / (2 * T);  // about 512 CUDA blocks in all
    return s < 1 ? 1 : (s > kSpMaxChunks ? kSpMaxChunks : s);
}

__global__ void __launch_bounds__(kSpThreads) k_attn_sparse_partial(const float * __restrict__ q, const half * __restrict__ k_cache,
                                                                    const half * __restrict__ v_cache, const std::int32_t * __restrict__ cells,
                                                                    const std::int32_t * __restrict__ n_cells, float scale, int n_split,
                                                                    float * __restrict__ work) {
    __shared__ __align__(16) float qs[kGroup * kHeadDim];   // [12][256]
    __shared__ float sc[kSpChunk * kGroup];                 // [64 cells][12]
    __shared__ int cell_s[kSpChunk];
    __shared__ float m_s[kGroup], l_s[kGroup], alpha_s[kGroup];
    const int sp = blockIdx.x, hk = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int n = n_cells[t];
    const int n_chunks = (n + kSpChunk - 1) / kSpChunk;
    for (int i = tid; i < kGroup * kHeadDim; i += kSpThreads) {
        qs[i] = q[(std::size_t(t) * kHeads + hk * kGroup + i / kHeadDim) * kHeadDim + i % kHeadDim];
    }
    if (tid < kGroup) {
        m_s[tid] = -INFINITY;
        l_s[tid] = 0.0f;
    }
    float o[kGroup];
#pragma unroll
    for (int i = 0; i < kGroup; ++i) o[i] = 0.0f;
    __syncthreads();

    const int c = tid >> 2, part = tid & 3;  // q . k: cell c of the chunk, 8-dim groups part, part+4, ...
    for (int ch = sp; ch < n_chunks; ch += n_split) {
        const int idx = ch * kSpChunk + c;
        const bool valid = idx < n;
        const int cell = valid ? cells[std::size_t(t) * kQsaWidth + idx] : 0;
        if (part == 0) cell_s[c] = cell;
        float acc[kGroup];
#pragma unroll
        for (int i = 0; i < kGroup; ++i) acc[i] = 0.0f;
        if (valid) {
            const uint4 * kr = reinterpret_cast<const uint4 *>(k_cache + (std::size_t(cell) * kKvHeads + hk) * kHeadDim);
#pragma unroll 2
            for (int j = 0; j < kHeadDim / 32; ++j) {
                const int d8 = part + 4 * j;
                const uint4 raw = kr[d8];
                const half2 * hp = reinterpret_cast<const half2 *>(&raw);
                float kf[8];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float2 f = __half22float2(hp[e]);
                    kf[2 * e] = f.x;
                    kf[2 * e + 1] = f.y;
                }
#pragma unroll
                for (int i = 0; i < kGroup; ++i) {
                    const float4 a = *reinterpret_cast<const float4 *>(qs + i * kHeadDim + d8 * 8);
                    const float4 b = *reinterpret_cast<const float4 *>(qs + i * kHeadDim + d8 * 8 + 4);
                    acc[i] += kf[0] * a.x + kf[1] * a.y + kf[2] * a.z + kf[3] * a.w + kf[4] * b.x + kf[5] * b.y + kf[6] * b.z + kf[7] * b.w;
                }
            }
        }
#pragma unroll
        for (int i = 0; i < kGroup; ++i) {
            acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], 1);
            acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], 2);
        }
        if (part == 0) {
#pragma unroll
            for (int i = 0; i < kGroup; ++i) sc[c * kGroup + i] = valid ? acc[i] * scale : -INFINITY;
        }
        __syncthreads();

        for (int i = warp; i < kGroup; i += kSpThreads / 32) {  // a warp per query head: 2 cells per lane
            const float s0 = sc[lane * kGroup + i], s1 = sc[(lane + 32) * kGroup + i];
            const float cm = warp_max(fmaxf(s0, s1));
            const float mo = m_s[i], mn = fmaxf(mo, cm);  // the chunk holds a valid cell, so mn is finite
            const float p0 = s0 == -INFINITY ? 0.0f : expf(s0 - mn), p1 = s1 == -INFINITY ? 0.0f : expf(s1 - mn);
            sc[lane * kGroup + i] = p0;
            sc[(lane + 32) * kGroup + i] = p1;
            const float sum = warp_sum(p0 + p1);
            if (lane == 0) {
                const float alpha = mo == -INFINITY ? 0.0f : expf(mo - mn);
                l_s[i] = l_s[i] * alpha + sum;
                m_s[i] = mn;
                alpha_s[i] = alpha;
            }
        }
        __syncthreads();

        const int nv = min(kSpChunk, n - ch * kSpChunk);
#pragma unroll
        for (int i = 0; i < kGroup; ++i) o[i] *= alpha_s[i];
        for (int cc = 0; cc < nv; ++cc) {
            const float vv = __half2float(v_cache[(std::size_t(cell_s[cc]) * kKvHeads + hk) * kHeadDim + tid]);
#pragma unroll
            for (int i = 0; i < kGroup; ++i) o[i] += sc[cc * kGroup + i] * vv;
        }
        __syncthreads();
    }

    float * w = work + ((std::size_t(t) * kKvHeads + hk) * n_split + sp) * kGroup * kSpStride;
#pragma unroll
    for (int i = 0; i < kGroup; ++i) w[i * kSpStride + 2 + tid] = o[i];
    if (tid < kGroup) {
        w[tid * kSpStride] = m_s[tid];
        w[tid * kSpStride + 1] = l_s[tid];
    }
}

__global__ void __launch_bounds__(kHeadDim) k_attn_sparse_combine(const float * __restrict__ work, const float * __restrict__ gate,
                                                                  int n_split, float * __restrict__ out) {
    const int t = blockIdx.x / kHeads, h = blockIdx.x % kHeads, d = threadIdx.x;
    const int hk = h / kGroup, g = h % kGroup;
    const float * base = work + (std::size_t(t) * kKvHeads + hk) * n_split * kGroup * kSpStride + g * kSpStride;
    float M = -INFINITY;
    for (int sp = 0; sp < n_split; ++sp) M = fmaxf(M, base[std::size_t(sp) * kGroup * kSpStride]);
    float L = 0.0f, acc = 0.0f;
    for (int sp = 0; sp < n_split; ++sp) {
        const float * w = base + std::size_t(sp) * kGroup * kSpStride;
        if (w[0] == -INFINITY) continue;
        const float f = expf(w[0] - M);
        L += w[1] * f;
        acc += w[2 + d] * f;
    }
    const std::size_t o = (std::size_t(t) * kHeads + h) * kHeadDim + d;
    out[o] = (acc / L) * sigmoidf_(gate[o]);
}

}  // namespace

// ================================================================================================

void qsa_update_blocks(const float * idx_raw, const float * k_norm, const double * rope_inv_freq, float * blocks,
                       const std::int64_t * pos0, int T, float eps, cudaStream_t s) {
    if (T < 1) throw std::runtime_error("qsa_update_blocks: T must be positive");
    k_qsa_blocks<<<T / kQsaRatio + 1, kQsaDim, 0, s>>>(idx_raw, k_norm, rope_inv_freq, blocks, pos0, T, eps);
    launched("qsa_update_blocks");
}

void qsa_query(float * q, const float * q_norm, const double * rope_inv_freq, const std::int64_t * pos0, int T, float eps,
               cudaStream_t s) {
    k_qsa_query<<<dim3(T, kQsaHeads), kQsaDim, 0, s>>>(q, q_norm, rope_inv_freq, pos0, eps);
    launched("qsa_query");
}

static std::int64_t score_stride(std::int64_t max_ctx) { return (max_ctx / kQsaRatio + 63) / 64 * 64; }  // 64-float aligned rows

std::size_t qsa_select_work_bytes(int T, std::int64_t max_ctx) { return std::size_t(T) * std::size_t(score_stride(max_ctx)) * sizeof(float); }

void qsa_scores(const float * q, const float * blocks, const std::int64_t * pos0, int T, std::int64_t max_ctx, void * work, cudaStream_t s) {
    const std::int64_t stride = score_stride(max_ctx);
    const unsigned gx = unsigned((stride + kScoreRows - 1) / kScoreRows);
    float * scores = static_cast<float *>(work);
    if (T == 1) {
        k_qsa_scores<1><<<dim3(gx, 1), kScoreWarps * 32, 0, s>>>(q, blocks, pos0, T, stride, scores);
    } else if (T == 2) {
        k_qsa_scores<2><<<dim3(gx, 1), kScoreWarps * 32, 0, s>>>(q, blocks, pos0, T, stride, scores);
    } else {
        k_qsa_scores<4><<<dim3(gx, unsigned((T + 3) / 4)), kScoreWarps * 32, 0, s>>>(q, blocks, pos0, T, stride, scores);
    }
    launched("qsa_scores");
}

void qsa_pick(const void * work, const std::int64_t * pos0, int T, std::int64_t max_ctx, std::int32_t * cells, std::int32_t * n_cells,
              cudaStream_t s) {
    k_qsa_select<<<T, kSelThreads, 0, s>>>(static_cast<const float *>(work), pos0, score_stride(max_ctx), cells, n_cells);
    launched("qsa_pick");
}

void qsa_select(const float * q, const float * blocks, const std::int64_t * pos0, int T, std::int64_t max_ctx, void * work,
                std::int32_t * cells, std::int32_t * n_cells, cudaStream_t s) {
    qsa_scores(q, blocks, pos0, T, max_ctx, work, s);
    qsa_pick(work, pos0, T, max_ctx, cells, n_cells, s);
}

std::size_t attn_sparse_work_floats(int T) {  // enough for any T' <= T (T' * splits(T') is not monotonic)
    std::size_t parts = 0;
    for (int t = 1; t <= T; ++t) parts = std::max(parts, std::size_t(t) * std::size_t(sparse_splits(t)));
    return parts * kKvHeads * kGroup * kSpStride;
}

void attn_sparse(const float * q, const float * gate, const half * k_cache, const half * v_cache, const std::int32_t * cells,
                 const std::int32_t * n_cells, int T, float scale, float * work, float * out, cudaStream_t s) {
    const int n_split = sparse_splits(T);
    k_attn_sparse_partial<<<dim3(n_split, kKvHeads, T), kSpThreads, 0, s>>>(q, k_cache, v_cache, cells, n_cells, scale, n_split, work);
    launched("attn_sparse_partial");
    k_attn_sparse_combine<<<T * kHeads, kHeadDim, 0, s>>>(work, gate, n_split, out);
    launched("attn_sparse_combine");
}

void qsa_init() {}

}  // namespace ninfer::flashnext::cuda
