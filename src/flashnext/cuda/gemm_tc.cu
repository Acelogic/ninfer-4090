// Prompt-sized products with Q8_0 weights on the tensor cores, at FP32-class accuracy: the weights enter as
// their exact int8 codes (which fp16 holds exactly) with each 32-block's scale applied in FP32 to the
// block's partial sum, and every activation as two fp16 terms (hi = fp16(v), lo = fp16(v - hi): about
// 22 bits) after a per-row power-of-two scale that puts the row's largest value in [2^14, 2^15). The hi
// products accumulate in FP32; the lo ones (at most 2^-11 of the hi ones) in fp16 within a block. The same
// arithmetic as experts_gpu_batch's Fp16x2 mode (4e-7 relative error against exact math there).
#include "flashnext/cuda/gemm_tc.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>

#include "flashnext/cuda/device.h"

namespace ninfer::flashnext::cuda {

namespace {

constexpr int kBM = 64, kBN = 256, kBK = 32, kThreads = 256, kHalfN = kBN / 2;

__device__ __forceinline__ int swz(int row, int chunk) { return chunk ^ ((row >> 1) & 3); }

__device__ __forceinline__ void ldmatrix_x4(std::uint32_t (&r)[4], const void * p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_f32(float (&d)[4], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16(std::uint32_t (&d)[2], const std::uint32_t (&a)[4], std::uint32_t b0, std::uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ std::uint32_t pack_half2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const std::uint32_t *>(&h);
}
__device__ __forceinline__ float row_scale(float m) {
    if (!(m > 0.0f) || !isfinite(m)) return 1.0f;
    int e;
    frexpf(m, &e);
    return ldexpf(1.0f, min(max(15 - e, -100), 100));
}

// per row: the power-of-two scale of its fp16 terms (s) and its inverse
__global__ void __launch_bounds__(256) k_row_scales(const float * __restrict__ x, int T, int K, float2 * __restrict__ sc) {
    const int t = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (t >= T) return;
    const float4 * row = reinterpret_cast<const float4 *>(x + std::size_t(t) * K);
    float m = 0.0f;
    for (int i = lane; i < K / 4; i += 32) {
        const float4 v = row[i];
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if (lane == 0) {
        const float s = row_scale(m);
        sc[t] = make_float2(s, 1.0f / s);
    }
}

struct Smem {
    uint4 a[2][2][kBM][4];  // [buffer][hi, lo][row][16-byte chunk]
    uint4 b[2][kBN][4];
    float scale[2][kBN];
    float ainv[kBM];
    float as[kBM];
};

// y[t][n] = sum_k x[t][k] w[n][k]: a 64 x 256 tile per block; thread tid decodes weight row n0 + tid
__global__ void __launch_bounds__(kThreads, 1) k_gemm_q8(const float * __restrict__ x, const std::int8_t * __restrict__ codes,
                                                         const __half * __restrict__ scales, const float2 * __restrict__ rs, float * __restrict__ y,
                                                         int T, int N, int K) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto & S = *reinterpret_cast<Smem *>(smem_raw);
    const int tid = threadIdx.x, t0 = blockIdx.y * kBM, n0 = blockIdx.x * kBN;
    const int KT = K / kBK;
    if (tid < kBM) {
        const bool ok = t0 + tid < T;
        S.as[tid] = ok ? rs[t0 + tid].x : 0.0f;
        S.ainv[tid] = ok ? rs[t0 + tid].y : 0.0f;
    }
    __syncthreads();
    // activations: thread tid converts 8 values (16-byte chunk achunk) of row ar per stage
    const int ar = tid >> 2, achunk = tid & 3;
    const bool avalid = t0 + ar < T;
    const float * arow = x + std::size_t(avalid ? t0 + ar : 0) * K + 8 * achunk;
    const float asc = S.as[ar];
    // weights: row n of this thread
    const int n = n0 + tid;
    const bool wvalid = n < N;
    const uint4 * wrow = reinterpret_cast<const uint4 *>(codes + std::size_t(wvalid ? n : 0) * K);
    const __half * srow = scales + std::size_t(wvalid ? n : 0) * (K / kBK);

    float4 xa, xb;
    uint4 w0, w1;
    float wsc;
    auto fetch = [&](int kt) {
        if (avalid) {
            xa = reinterpret_cast<const float4 *>(arow + kt * kBK)[0];
            xb = reinterpret_cast<const float4 *>(arow + kt * kBK)[1];
        } else {
            xa = xb = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        if (wvalid) {
            w0 = wrow[2 * kt];
            w1 = wrow[2 * kt + 1];
            wsc = __half2float(srow[kt]);
        } else {
            w0 = w1 = make_uint4(0u, 0u, 0u, 0u);
            wsc = 0.0f;
        }
    };
    // int8 -> fp16 without conversions: fp16 0x6400 + u is 1024 + u; bias the code to u = c + 128, subtract 1152
    auto stash_a = [&](int buf) {
        const float v[8] = {xa.x * asc, xa.y * asc, xa.z * asc, xa.w * asc, xb.x * asc, xb.y * asc, xb.z * asc, xb.w * asc};
        std::uint32_t hi[4], lo[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __half2 h = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
            hi[i] = *reinterpret_cast<const std::uint32_t *>(&h);
            lo[i] = pack_half2(v[2 * i] - __low2float(h), v[2 * i + 1] - __high2float(h));
        }
        S.a[buf][0][ar][swz(ar, achunk)] = make_uint4(hi[0], hi[1], hi[2], hi[3]);
        S.a[buf][1][ar][swz(ar, achunk)] = make_uint4(lo[0], lo[1], lo[2], lo[3]);
    };
    // weights are decoded halfway through the MMAs (which keep running while the decoding issues)
    auto stash_b = [&](int buf) {
        const std::uint32_t c[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        const __half2 bias = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
        std::uint32_t o[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const std::uint32_t u = c[i] ^ 0x80808080u;
            const std::uint32_t p0 = __byte_perm(u, 0x64646464u, 0x4140), p1 = __byte_perm(u, 0x64646464u, 0x4342);
            const __half2 h0 = __hsub2(*reinterpret_cast<const __half2 *>(&p0), bias), h1 = __hsub2(*reinterpret_cast<const __half2 *>(&p1), bias);
            o[2 * i] = *reinterpret_cast<const std::uint32_t *>(&h0);
            o[2 * i + 1] = *reinterpret_cast<const std::uint32_t *>(&h1);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) S.b[buf][tid][swz(tid, i)] = make_uint4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
        S.scale[buf][tid] = wsc;
    };

    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;
    const int mrows = min(T - t0, kBM) - wm * 32;
    float acc[2][8][4] = {};
    fetch(0);
    stash_a(0);
    stash_b(0);
    __syncthreads();
    for (int kt = 0; kt < KT; ++kt) {
        const int buf = kt & 1;
        const bool more = kt + 1 < KT;
        if (more) fetch(kt + 1);
        if (mrows > 0) {
            std::uint32_t af[2][2][2][4];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                if (i * 16 < mrows)
#pragma unroll
                    for (int kk = 0; kk < 2; ++kk) {
                        const int r = wm * 32 + i * 16 + (lane & 15);
#pragma unroll
                        for (int l = 0; l < 2; ++l) ldmatrix_x4(af[i][kk][l], &S.a[buf][l][r][swz(r, 2 * kk + (lane >> 4))]);
                    }
#pragma unroll
            for (int jn = 0; jn < 8; ++jn) {
                const int nb = (jn < 4 ? 0 : kHalfN) + wn * 32 + (jn & 3) * 8;
                const int br = nb + (lane & 7);
                std::uint32_t bf[4];
                ldmatrix_x4(bf, &S.b[buf][br][swz(br, lane >> 3)]);
                const float2 sc = *reinterpret_cast<const float2 *>(&S.scale[buf][nb + 2 * (lane & 3)]);
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    if (i * 16 >= mrows) continue;
                    float t[4] = {0.f, 0.f, 0.f, 0.f};
                    mma_f32(t, af[i][0][0], bf[0], bf[1]);
                    mma_f32(t, af[i][1][0], bf[2], bf[3]);
                    std::uint32_t tl[2] = {0u, 0u};
                    mma_f16(tl, af[i][0][1], bf[0], bf[1]);
                    mma_f16(tl, af[i][1][1], bf[2], bf[3]);
                    const float2 l01 = __half22float2(*reinterpret_cast<const __half2 *>(&tl[0]));
                    const float2 l23 = __half22float2(*reinterpret_cast<const __half2 *>(&tl[1]));
                    t[0] += l01.x, t[1] += l01.y, t[2] += l23.x, t[3] += l23.y;
                    acc[i][jn][0] = fmaf(sc.x, t[0], acc[i][jn][0]);
                    acc[i][jn][1] = fmaf(sc.y, t[1], acc[i][jn][1]);
                    acc[i][jn][2] = fmaf(sc.x, t[2], acc[i][jn][2]);
                    acc[i][jn][3] = fmaf(sc.y, t[3], acc[i][jn][3]);
                }
                if (jn == 3 && more) stash_b(buf ^ 1);
            }
        } else if (more) {
            stash_b(buf ^ 1);
        }
        if (more) stash_a(buf ^ 1);
        __syncthreads();
    }
    if (mrows <= 0) return;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int r = wm * 32 + i * 16 + half * 8 + (lane >> 2);
            if (t0 + r >= T) continue;
            const float inv = S.ainv[r];
            float * yr = y + std::size_t(t0 + r) * N;
#pragma unroll
            for (int jn = 0; jn < 8; ++jn) {
                const int nc = n0 + (jn < 4 ? 0 : kHalfN) + wn * 32 + (jn & 3) * 8 + 2 * (lane & 3);
                if (nc + 1 < N) *reinterpret_cast<float2 *>(yr + nc) = make_float2(acc[i][jn][2 * half] * inv, acc[i][jn][2 * half + 1] * inv);
                else if (nc < N) yr[nc] = acc[i][jn][2 * half] * inv;
            }
        }
    }
}

std::once_flag init_once;

}  // namespace

bool gemm_q8_tc_supported(int n, int k) { return k % kBK == 0 && k >= kBK && n % 2 == 0; }

void gemm_q8_tc(const std::int8_t * codes, const void * scales, int n, int k, const float * x, float * y, int T, float * row_scales,
                cudaStream_t s) {
    if (T <= 0) return;
    if (!gemm_q8_tc_supported(n, k)) throw std::runtime_error("gemm_q8_tc: unsupported shape");
    std::call_once(init_once, [] {
        check(cudaFuncSetAttribute(k_gemm_q8, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(Smem))), "gemm_q8_tc smem");
        check(cudaFuncSetAttribute(k_gemm_q8, cudaFuncAttributePreferredSharedMemoryCarveout, 64), "gemm_q8_tc carveout");
    });
    auto * rs = reinterpret_cast<float2 *>(row_scales);
    k_row_scales<<<(T + 7) / 8, 256, 0, s>>>(x, T, k, rs);
    check(cudaGetLastError(), "gemm_q8_tc scales");
    const dim3 grid(unsigned((n + kBN - 1) / kBN), unsigned((T + kBM - 1) / kBM));
    k_gemm_q8<<<grid, kThreads, sizeof(Smem), s>>>(x, codes, static_cast<const __half *>(scales), rs, y, T, n, k);
    check(cudaGetLastError(), "gemm_q8_tc");
}

}  // namespace ninfer::flashnext::cuda
