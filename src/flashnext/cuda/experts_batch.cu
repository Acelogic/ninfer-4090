#include "flashnext/cuda/experts_batch.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>

#include "flashnext/cuda/device.h"
#include "flashnext/cuda/ops.h"
#include "flashnext/iq3s_grid.h"
#include "flashnext/quants.h"

namespace ninfer::flashnext::cuda {

namespace {

// A tile is kBM pairs of one slot against kBN weight rows: for gate/up, 128 gate rows and the 128 up
// rows of the same FF columns (so SwiGLU happens in registers); for down, 256 output columns. One
// 8-warp block per SM.
//
// What limits these kernels is L2 bandwidth (about 2.2 TB/s here) more than arithmetic:
//  - activation rows are re-read once per column tile, hence the wide tiles; they are read with full
//    sectors and kept out of L1;
//  - each weight row is read by one thread, a few bytes per stage, so L1 must hold a block's 256
//    rows: shared memory stays at or below 64 KB so that the other 64 KB of the SM's 128 KB is L1.
constexpr int kBM = 64, kBN = 256, kThreads = 256;
constexpr int kHalfN = kBN / 2;  // columns of the left half (gate rows, or the first 128 outputs)
constexpr int kSub = 32;         // weights per scale (sub-block) in every weight format
constexpr int kMaxBatchTokens = 16384;
constexpr int kCarveout = 64;    // percent of the 100 KB maximum: 64 KB shared, 64 KB L1

__device__ std::uint32_t d_grid_b[512];
__constant__ std::int8_t c_iq4nl_b[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

std::size_t align256(std::size_t v) { return (v + 255) & ~std::size_t(255); }

// Tiles needed for P pairs at most: every slot's last tile may be partial.
int max_tiles(int pairs) { return pairs / kBM + std::min(pairs, kExperts) + 1; }

// The fp16 terms of the tensor-core modes reuse memory: those of h overwrite h in place (a row of 640
// floats holds its 640 hi and 640 lo halves), those of x sit at the start of y, which the down GEMM
// writes only after gate/up has read them.
struct Workspace {
    int * counts = nullptr;  // [0]: tiles, [1]: cached pairs
    int4 * tiles = nullptr;  // {slot, first sorted position, rows, 0}
    int * order = nullptr;   // sorted position -> pair index
    float * xinv = nullptr;  // per token: 1 / the power-of-two scale of its fp16 terms
    float * hinv = nullptr;  // per sorted position: the same for h
    float * h = nullptr;     // [pairs][640] SwiGLU activations by sorted position; then [pairs][hi 640 | lo 640] fp16
    float * y = nullptr;     // [pairs][2560] expert outputs by pair index; before that [2][tokens][2560] fp16 terms of x
    __half * xs = nullptr;   // = y
    __half * hs = nullptr;   // = h
    std::size_t bytes = 0;
};

Workspace carve(void * base, int n_tokens) {
    const std::size_t pairs = std::size_t(n_tokens) * kUsed, tokens = std::size_t(n_tokens);
    std::size_t at = 0;
    auto take = [&](std::size_t bytes) { const std::size_t o = at; at += align256(bytes); return o; };
    const std::size_t o_c = take(2 * sizeof(int)), o_t = take(sizeof(int4) * std::size_t(max_tiles(int(pairs)))), o_o = take(sizeof(int) * pairs),
                      o_xi = take(sizeof(float) * tokens), o_hi = take(sizeof(float) * pairs), o_h = take(sizeof(float) * pairs * kExpertFF),
                      o_y = take(sizeof(float) * pairs * kEmbd);
    static_assert(2 * sizeof(__half) * kEmbd <= sizeof(float) * kUsed * kEmbd, "x terms must fit in y");
    Workspace w;
    w.bytes = at;
    if (!base) return w;
    auto * b = static_cast<std::uint8_t *>(base);
    w.counts = reinterpret_cast<int *>(b + o_c);
    w.tiles = reinterpret_cast<int4 *>(b + o_t);
    w.order = reinterpret_cast<int *>(b + o_o);
    w.xinv = reinterpret_cast<float *>(b + o_xi);
    w.hinv = reinterpret_cast<float *>(b + o_hi);
    w.h = reinterpret_cast<float *>(b + o_h);
    w.y = reinterpret_cast<float *>(b + o_y);
    w.xs = reinterpret_cast<__half *>(w.y);
    w.hs = reinterpret_cast<__half *>(w.h);
    return w;
}

__device__ __forceinline__ bool cached(int s) { return s >= 0 && s < kExperts; }

// ---- grouping ----
// One block of 512 threads (one per possible slot). The pairs are split into 16 contiguous ranges, one
// per warp. Pass 1 counts each slot per range, a scan over ranges and slots gives every (range, slot)
// its first position, pass 2 writes the pairs in index order. The order is stable, so the positions
// (and therefore which rows share a tile) are the same on every run. Each warp reads its range 256
// pairs at a time (8 loads in flight), then walks them 32 at a time.
constexpr int kGroupWarps = kExperts / 32, kGroupUnroll = 8;

__device__ __forceinline__ void load_slots(const std::int32_t * __restrict__ slots, int b, int hi, int (&s)[kGroupUnroll]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int u = 0; u < kGroupUnroll; ++u) {
        const int p = b + 32 * u + lane;
        const int v = p < hi ? slots[p] : -1;
        s[u] = cached(v) ? v : -1;
    }
}

__global__ void __launch_bounds__(kExperts) k_group(const std::int32_t * __restrict__ slots, int pairs, int * __restrict__ counts,
                                                   int4 * __restrict__ tiles, int * __restrict__ order) {
    __shared__ int before[kGroupWarps][kExperts];  // pairs of slot s in earlier ranges, then: placed so far
    __shared__ int start[kExperts];
    __shared__ int part[2][32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int w = 0; w < kGroupWarps; ++w) before[w][tid] = 0;
    if (tid < 64) part[tid >> 5][lane] = 0;
    __syncthreads();
    const int per = (pairs + kGroupWarps - 1) / kGroupWarps;
    const int lo = warp * per, hi = min(pairs, lo + per);
    for (int b = lo; b < hi; b += 32 * kGroupUnroll) {
        int sv[kGroupUnroll];
        load_slots(slots, b, hi, sv);
#pragma unroll
        for (int u = 0; u < kGroupUnroll; ++u) {
            const int s = sv[u];
            const unsigned peers = __match_any_sync(0xffffffffu, s);
            if (s >= 0 && lane == __ffs(peers) - 1) before[warp][s] += __popc(peers);
            __syncwarp();
        }
    }
    __syncthreads();
    int count = 0;
    for (int w = 0; w < kGroupWarps; ++w) {
        const int c = before[w][tid];
        before[w][tid] = count;
        count += c;
    }
    // exclusive scans over the slots of the pair counts and the tile counts
    const int tcount = (count + kBM - 1) / kBM;
    int a = count, t = tcount;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int ua = __shfl_up_sync(0xffffffffu, a, o), ut = __shfl_up_sync(0xffffffffu, t, o);
        if (lane >= o) a += ua, t += ut;
    }
    if (lane == 31) part[0][warp] = a, part[1][warp] = t;
    __syncthreads();
    if (warp == 0) {
        const int a0 = part[0][lane], t0 = part[1][lane];
        int sa = a0, st = t0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int ua = __shfl_up_sync(0xffffffffu, sa, o), ut = __shfl_up_sync(0xffffffffu, st, o);
            if (lane >= o) sa += ua, st += ut;
        }
        part[0][lane] = sa - a0;
        part[1][lane] = st - t0;
    }
    __syncthreads();
    const int first = part[0][warp] + a - count, tfirst = part[1][warp] + t - tcount;
    start[tid] = first;
    for (int i = 0; i < tcount; ++i) tiles[tfirst + i] = make_int4(tid, first + i * kBM, min(kBM, count - i * kBM), 0);
    if (tid == kExperts - 1) counts[0] = tfirst + tcount, counts[1] = first + count;
    __syncthreads();
    for (int b = lo; b < hi; b += 32 * kGroupUnroll) {
        int sv[kGroupUnroll];
        load_slots(slots, b, hi, sv);
#pragma unroll
        for (int u = 0; u < kGroupUnroll; ++u) {
            const int s = sv[u];
            const unsigned peers = __match_any_sync(0xffffffffu, s);
            if (s >= 0) order[start[s] + before[warp][s] + __popc(peers & ((1u << lane) - 1u))] = b + 32 * u + lane;
            __syncwarp();
            if (s >= 0 && lane == __ffs(peers) - 1) before[warp][s] += __popc(peers);
            __syncwarp();
        }
    }
}

// ---- weight sub-blocks ----
// load(k) brings the raw bytes of sub-block k (32 weights) of one weight row into registers. The
// kernels keep two of these and alternate between them (sub-block k + 2 loads while k + 1 waits to be
// decoded), so a field shared by a group of sub-blocks is loaded on each object's first sub-block of
// the group: k % group < kDepth.
// The decoders turn the bytes into small integers and one FP32 scale, both exact: scale * q is the
// value dequantize_row gives, bit for bit. decode_f32 gives 16 of the integers (half 0 or 1 of the
// sub-block, in K order) as floats; decode_f16 gives all 32 as fp16 (which holds every one of them
// exactly) packed in four uint4.

struct F32Tables {
    const float4 * grid;  // IQ3_S grid entries as four floats
    const float * lut;    // IQ4_NL values
};
struct F16Tables {
    const uint2 * grid;          // IQ3_S grid entries as four fp16
    const uint2 * signs;         // sign nibble -> XOR masks for four fp16
    const std::uint32_t * lut2;  // byte -> fp16 pair (IQ4_NL value of the low nibble, of the high nibble)
};

constexpr int kDepth = 2;

__device__ __forceinline__ std::uint32_t pack_half2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const std::uint32_t *>(&h);
}

// 4 bytes from a 2-byte aligned address (IQ3_S blocks are 110 bytes)
__device__ __forceinline__ std::uint32_t ld32_a2(const std::uint8_t * p) {
    if ((reinterpret_cast<std::uintptr_t>(p) & 2) == 0) return *reinterpret_cast<const std::uint32_t *>(p);
    const auto * q = reinterpret_cast<const std::uint16_t *>(p);
    return q[0] | (std::uint32_t(q[1]) << 16);
}

// IQ3_S gate/up row: ggml's weight order (see iq3s_signed in quants.cpp).
struct SubIQ3S {
    std::uint32_t qs0, qs1, signs;  // this sub-block's 8 grid bytes and 4 sign bytes
    std::uint32_t qh0, qh1, sc, d;  // per 256-block: high index bits, 4-bit scales, fp16 scale
    int ib = 0;
    __device__ __forceinline__ void load(const std::uint8_t * row, const std::uint8_t *, int kt) {
        const std::uint8_t * b = row + std::size_t(kt >> 3) * sizeof(BlockIQ3_S);
        ib = kt & 7;
        qs0 = ld32_a2(b + offsetof(BlockIQ3_S, qs) + 8 * ib);
        qs1 = ld32_a2(b + offsetof(BlockIQ3_S, qs) + 8 * ib + 4);
        signs = ld32_a2(b + offsetof(BlockIQ3_S, signs) + 4 * ib);
        if (ib < kDepth) {
            d = *reinterpret_cast<const std::uint16_t *>(b);
            qh0 = ld32_a2(b + offsetof(BlockIQ3_S, qh));
            qh1 = ld32_a2(b + offsetof(BlockIQ3_S, qh) + 4);
            sc = ld32_a2(b + offsetof(BlockIQ3_S, scales));
        }
    }
    // grid indices of the two entries of group l (weights 8l .. 8l+3 and 8l+4 .. 8l+7)
    __device__ __forceinline__ void index(int l, int & i1, int & i2) const {
        const int qh = int(((ib < 4 ? qh0 : qh1) >> (8 * (ib & 3))) & 0xFF);
        const std::uint32_t pair = ((l < 2 ? qs0 : qs1) >> (16 * (l & 1))) & 0xFFFF;
        i1 = int(pair & 0xFF) | ((qh << (8 - 2 * l)) & 256);
        i2 = int(pair >> 8) | ((qh << (7 - 2 * l)) & 256);
    }
    __device__ __forceinline__ float scale() const {
        const int ls = int((sc >> (4 * ib)) & 0xF);
        return __half2float(__ushort_as_half(std::uint16_t(d))) * float(1 + 2 * ls);
    }
    __device__ __forceinline__ float decode_f32(const F32Tables & t, int half, float (&q)[16]) const {
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            const int l = 2 * half + g;
            int i1, i2;
            index(l, i1, i2);
            const float4 g1 = t.grid[i1], g2 = t.grid[i2];
            const std::uint32_t sg = signs >> (8 * l);
            const float m[8] = {g1.x, g1.y, g1.z, g1.w, g2.x, g2.y, g2.z, g2.w};
#pragma unroll
            for (int j = 0; j < 8; ++j) q[8 * g + j] = (sg >> j) & 1 ? -m[j] : m[j];
        }
        return scale();
    }
    __device__ __forceinline__ float decode_f16(const F16Tables & t, uint4 (&w)[4]) const {
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            int i1, i2;
            index(l, i1, i2);
            const uint2 g1 = t.grid[i1], g2 = t.grid[i2];
            const std::uint32_t sg = signs >> (8 * l);
            const uint2 s1 = t.signs[sg & 0xF], s2 = t.signs[(sg >> 4) & 0xF];
            w[l] = make_uint4(g1.x ^ s1.x, g1.y ^ s1.y, g2.x ^ s2.x, g2.y ^ s2.y);
        }
        return scale();
    }
};

// 16 bytes of IQ4_NL codes: low nibbles are weights 0..15 of the sub-block, high nibbles 16..31.
__device__ __forceinline__ void iq4_f32(const std::uint32_t (&c)[4], const float * lut, int half, float (&q)[16]) {
#pragma unroll
    for (int i = 0; i < 16; ++i) q[i] = lut[(c[i >> 2] >> (8 * (i & 3) + 4 * half)) & 0xF];
}
__device__ __forceinline__ void iq4_f16(const std::uint32_t (&c)[4], const std::uint32_t * lut2, uint4 (&w)[4]) {
    std::uint32_t lo[8], hi[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const std::uint32_t word = c[i >> 1], sh = 16 * (i & 1);
        const std::uint32_t a = lut2[(word >> sh) & 0xFF], b = lut2[(word >> (sh + 8)) & 0xFF];
        lo[i] = __byte_perm(a, b, 0x5410);  // low nibbles of bytes 2i, 2i+1: weights 2i, 2i+1
        hi[i] = __byte_perm(a, b, 0x7632);  // high nibbles: weights 16+2i, 17+2i
    }
    w[0] = make_uint4(lo[0], lo[1], lo[2], lo[3]);
    w[1] = make_uint4(lo[4], lo[5], lo[6], lo[7]);
    w[2] = make_uint4(hi[0], hi[1], hi[2], hi[3]);
    w[3] = make_uint4(hi[4], hi[5], hi[6], hi[7]);
}

// IQ4_XS gate/up row (136-byte blocks: the codes are 8-byte aligned).
struct SubIQ4XS {
    uint2 q0, q1;
    uint2 meta;  // per 256-block: d | scales_h << 16, scales_l
    int ib = 0;
    __device__ __forceinline__ void load(const std::uint8_t * row, const std::uint8_t *, int kt) {
        const std::uint8_t * b = row + std::size_t(kt >> 3) * sizeof(BlockIQ4_XS);
        ib = kt & 7;
        const auto * q = reinterpret_cast<const uint2 *>(b + offsetof(BlockIQ4_XS, qs) + 16 * ib);
        q0 = q[0];
        q1 = q[1];
        if (ib < kDepth) meta = *reinterpret_cast<const uint2 *>(b);
    }
    __device__ __forceinline__ float scale() const {
        const int ls = int((meta.y >> (4 * ib)) & 0xF) | int(((meta.x >> (16 + 2 * ib)) & 3) << 4);
        return __half2float(__ushort_as_half(std::uint16_t(meta.x))) * float(ls - 32);
    }
    __device__ __forceinline__ float decode_f32(const F32Tables & t, int half, float (&q)[16]) const {
        iq4_f32({q0.x, q0.y, q1.x, q1.y}, t.lut, half, q);
        return scale();
    }
    __device__ __forceinline__ float decode_f16(const F16Tables & t, uint4 (&w)[4]) const {
        iq4_f16({q0.x, q0.y, q1.x, q1.y}, t.lut2, w);
        return scale();
    }
};

// IQ4_NL down row from the code plane (16 bytes per sub-block) and the fp16 scale plane (rows of 40
// bytes, read 4 scales at a time).
struct SubIQ4NL {
    uint4 q;
    uint2 s;
    int kt = 0;
    __device__ __forceinline__ void load(const std::uint8_t * codes, const std::uint8_t * scales, int k) {
        kt = k;
        q = reinterpret_cast<const uint4 *>(codes)[k];
        if ((k & 3) < kDepth) s = reinterpret_cast<const uint2 *>(scales)[k >> 2];
    }
    __device__ __forceinline__ float scale() const {
        return __half2float(__ushort_as_half(std::uint16_t(((kt & 2) ? s.y : s.x) >> (16 * (kt & 1)))));
    }
    __device__ __forceinline__ float decode_f32(const F32Tables & t, int half, float (&v)[16]) const {
        iq4_f32({q.x, q.y, q.z, q.w}, t.lut, half, v);
        return scale();
    }
    __device__ __forceinline__ float decode_f16(const F16Tables & t, uint4 (&w)[4]) const {
        iq4_f16({q.x, q.y, q.z, q.w}, t.lut2, w);
        return scale();
    }
};

// Q8_0 down row: 32 int8 codes per sub-block.
struct SubQ8 {
    uint4 q0, q1;
    uint2 s;
    int kt = 0;
    __device__ __forceinline__ void load(const std::uint8_t * codes, const std::uint8_t * scales, int k) {
        kt = k;
        q0 = reinterpret_cast<const uint4 *>(codes)[2 * k];
        q1 = reinterpret_cast<const uint4 *>(codes)[2 * k + 1];
        if ((k & 3) < kDepth) s = reinterpret_cast<const uint2 *>(scales)[k >> 2];
    }
    __device__ __forceinline__ float scale() const {
        return __half2float(__ushort_as_half(std::uint16_t(((kt & 2) ? s.y : s.x) >> (16 * (kt & 1)))));
    }
    __device__ __forceinline__ float decode_f32(const F32Tables &, int half, float (&v)[16]) const {
        const uint4 c = half ? q1 : q0;
        const std::uint32_t w[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
        for (int i = 0; i < 16; ++i) v[i] = float(std::int8_t(w[i >> 2] >> (8 * (i & 3))));
        return scale();
    }
    // int8 -> fp16 without conversions: fp16 0x6400 + u is 1024 + u for a byte u; bias the code to
    // u = c + 128, then subtract 1152.
    __device__ __forceinline__ float decode_f16(const F16Tables &, uint4 (&w)[4]) const {
        const std::uint32_t c[8] = {q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
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
        for (int i = 0; i < 4; ++i) w[i] = make_uint4(o[4 * i], o[4 * i + 1], o[4 * i + 2], o[4 * i + 3]);
        return scale();
    }
};

template <GgufType W> struct SubOf;
template <> struct SubOf<GgufType::IQ3_S> { using T = SubIQ3S; };
template <> struct SubOf<GgufType::IQ4_XS> { using T = SubIQ4XS; };
template <> struct SubOf<GgufType::IQ4_NL> { using T = SubIQ4NL; };
template <> struct SubOf<GgufType::Q8_0> { using T = SubQ8; };

struct Params {
    const std::uint8_t * pool;
    std::size_t slot_bytes, gate_row, up_off, down_off, scale_off;
    const int * counts;
    const int4 * tiles;
    const int * order;
    const float * x;
    float * h;
    float * y;
    const __half * xs;  // tensor-core modes: fp16 terms of x ([2][tokens][2560]) and h ([pairs][2][640])
    const __half * hs;
    const float * xinv;
    const float * hinv;
    std::size_t xs_limb;  // halves between the hi and the lo plane of x
};

// The weight row that thread tid of a block loads and decodes: for gate/up the gate row (tid < 128)
// or up row of FF column blockIdx.x * 128 + tid % 128, for down output row blockIdx.x * 256 + tid.
template <bool kDown, GgufType W>
__device__ __forceinline__ void weight_row(const Params & P, int slot, int tid, const std::uint8_t *& codes, const std::uint8_t *& scales) {
    const std::uint8_t * base = P.pool + std::size_t(slot) * P.slot_bytes;
    if constexpr (kDown) {
        constexpr std::size_t code_row = W == GgufType::IQ4_NL ? kExpertFF / 2 : kExpertFF;
        const int n = blockIdx.x * kBN + tid;
        codes = base + P.down_off + std::size_t(n) * code_row;
        scales = base + P.scale_off + std::size_t(n) * (kExpertFF / 32) * 2;
    } else {
        const int j = blockIdx.x * kHalfN + (tid & (kHalfN - 1));
        codes = base + (tid < kHalfN ? 0 : P.up_off) + std::size_t(j) * P.gate_row;
        scales = nullptr;
    }
}

__device__ __forceinline__ float silu_mul(float g, float u) { return g / (1.0f + expf(-g)) * u; }

// ---- FP32 path (CUDA cores) ----
// Stages of 16 K (half a sub-block, so that two fp32 stages fit in 48 KB). 8 warps; warp w covers rows
// 32 * (w / 4) .. +31 and columns 32 * (w % 4) .. +31 of each half of the 256 (gate | up, or output
// columns c and 128 + c). A thread holds 8 rows x (4 + 4) columns. Both tiles are K-major so the
// inner loop reads float4s that most lanes share (broadcast). Activations: thread tid loads float4
// tid % 4 of row tid / 4 of each stage (full 64-byte rows, kept out of L1).
constexpr int kBKF = 16;

template <GgufType W>
struct F32Smem {
    float a[2][kBKF][kBM];
    float b[2][kBKF][kBN];
    float4 grid[W == GgufType::IQ3_S ? 512 : 1];
    float lut[16];
    const float * arows[kBM];
};

template <bool kDown, GgufType W>
__global__ void __launch_bounds__(kThreads, 1) k_f32(const Params P) {
    using Sub = typename SubOf<W>::T;
    constexpr int K = kDown ? kExpertFF : kEmbd, ST = K / kBKF;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto & S = *reinterpret_cast<F32Smem<W> *>(smem_raw);
    if (int(blockIdx.y) >= P.counts[0]) return;
    const int4 tile = P.tiles[blockIdx.y];
    const int tid = threadIdx.x;
    if constexpr (W == GgufType::IQ3_S)
        for (int i = tid; i < 512; i += kThreads) {
            const std::uint32_t g = d_grid_b[i];
            S.grid[i] = make_float4(float(g & 0xFF), float((g >> 8) & 0xFF), float((g >> 16) & 0xFF), float(g >> 24));
        }
    if (tid < 16) S.lut[tid] = float(c_iq4nl_b[tid]);
    if (tid < kBM)
        S.arows[tid] = tid >= tile.z ? nullptr
                       : kDown       ? P.h + std::size_t(tile.y + tid) * kExpertFF
                                     : P.x + std::size_t(P.order[tile.y + tid] / kUsed) * kEmbd;
    __syncthreads();
    const F32Tables tables{S.grid, S.lut};
    const int chunk = tid & 3, ar = tid >> 2;
    const float * arow = S.arows[ar];
    const std::uint8_t *codes, *scales;
    weight_row<kDown, W>(P, tile.x, tid, codes, scales);

    float4 ra;
    Sub rb0, rb1;  // even and odd sub-blocks
    auto fetch = [&](int st) { ra = arow ? __ldcg(reinterpret_cast<const float4 *>(arow + st * kBKF) + chunk) : make_float4(0.f, 0.f, 0.f, 0.f); };
    auto stash = [&](int buf, int st, const Sub & rb) {
        S.a[buf][4 * chunk][ar] = ra.x;
        S.a[buf][4 * chunk + 1][ar] = ra.y;
        S.a[buf][4 * chunk + 2][ar] = ra.z;
        S.a[buf][4 * chunk + 3][ar] = ra.w;
        float q[16];
        const float sc = rb.decode_f32(tables, st & 1, q);
#pragma unroll
        for (int k = 0; k < 16; ++k) S.b[buf][k][tid] = sc * q[k];
    };

    const int warp = tid >> 5, lane = tid & 31;
    // warps 0-3 take rows 0..31, one per SM sub-partition, so small tiles still use all four
    const int r0 = (warp >> 2) * 32 + (lane >> 3) * 8, c0 = (warp & 3) * 32 + (lane & 7) * 4;
    const bool active = (warp >> 2) * 32 < tile.z;  // warps whose 32 rows are all past the tile skip the math
    float acc[8][8] = {};
    // stage st is half st % 2 of sub-block st / 2; at stage 2m + 1, sub-block m + 2 starts loading
    auto step = [&](int st, const Sub & next, Sub & future) {
        const int buf = st & 1;
        if (st + 1 < ST) fetch(st + 1);
        if ((st & 1) && st / 2 + 2 < ST / 2) future.load(codes, scales, st / 2 + 2);
        if (active) {
#pragma unroll
            for (int k = 0; k < kBKF; ++k) {
                const float4 a0 = *reinterpret_cast<const float4 *>(&S.a[buf][k][r0]);
                const float4 a1 = *reinterpret_cast<const float4 *>(&S.a[buf][k][r0 + 4]);
                const float4 b0 = *reinterpret_cast<const float4 *>(&S.b[buf][k][c0]);
                const float4 b1 = *reinterpret_cast<const float4 *>(&S.b[buf][k][c0 + kHalfN]);
                const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
            }
        }
        if (st + 1 < ST) stash(buf ^ 1, st + 1, next);
        __syncthreads();
    };
    fetch(0);
    rb0.load(codes, scales, 0);
    rb1.load(codes, scales, 1);
    stash(0, 0, rb0);
    __syncthreads();
    for (int st = 0; st < ST; st += 4) {
        step(st, rb0, rb1);      // stashes stage st + 1: sub-block st / 2
        step(st + 1, rb1, rb0);  // stashes st + 2 (sub-block st / 2 + 1); loads st / 2 + 2 into rb0
        step(st + 2, rb1, rb1);  // stashes st + 3 (st / 2 + 1)
        step(st + 3, rb0, rb1);  // stashes st + 4 (st / 2 + 2); loads st / 2 + 3 into rb1
    }
    if (!active) return;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = r0 + i;
        if (r >= tile.z) break;
        if constexpr (kDown) {
            float * yr = P.y + std::size_t(P.order[tile.y + r]) * kEmbd + blockIdx.x * kBN + c0;
            *reinterpret_cast<float4 *>(yr) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            *reinterpret_cast<float4 *>(yr + kHalfN) = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
        } else {
            float * hr = P.h + std::size_t(tile.y + r) * kExpertFF + blockIdx.x * kHalfN + c0;
            *reinterpret_cast<float4 *>(hr) = make_float4(silu_mul(acc[i][0], acc[i][4]), silu_mul(acc[i][1], acc[i][5]),
                                                          silu_mul(acc[i][2], acc[i][6]), silu_mul(acc[i][3], acc[i][7]));
        }
    }
}

// ---- tensor-core path ----
// mma.sync m16n8k16 with fp16 operands. The weight operand is the sub-block's small integer, which
// fp16 holds exactly; the sub-block's FP32 scale multiplies the partial sum of each 32-wide stage
// before it joins the FP32 accumulator. So the weights are exact and only the activations are
// rounded. k_split_x / k_split_h store each activation row times a power of two that puts its largest
// value in [2^14, 2^15) (no overflow, no precision lost to small rows; undone exactly at the end) as
// LIMBS fp16 terms: hi = fp16(v), lo = fp16(v - hi). LIMBS = 2 keeps about 22 bits of each
// activation, LIMBS = 1 eleven.
//
// The lo terms are at most 8 (half an fp16 step at 2^14) and the weights at most 128 in magnitude, so
// a 32-term partial sum of lo * w stays below 2^15 and can be accumulated in fp16, which the 4090's
// tensor cores do at twice the FP32-accumulate rate; its rounding is 2^-11 of a term that is itself
// 2^-11 of the result.
//
// 8 warps; warp w covers rows 32 * (w / 4) .. +31 and the same B columns as the FP32 path, as
// 2 (m16) x 8 (n8) fragments. Smem rows are 64 bytes (32 halves) with their four 16-byte chunks
// XOR-swizzled by row, so ldmatrix's 8 rows land on distinct banks. Activations arrive by cp.async.
constexpr int kBK = kSub;

__device__ __forceinline__ int swz(int row, int chunk) { return chunk ^ ((row >> 1) & 3); }

template <GgufType W, int LIMBS>
struct TcSmem {
    uint4 a[2][LIMBS][kBM][4];
    uint4 b[2][kBN][4];
    float scale[2][kBN];
    uint2 grid[W == GgufType::IQ3_S ? 512 : 1];
    uint2 signs[16];
    std::uint32_t lut2[W == GgufType::IQ3_S ? 1 : 256];
    const __half * arows[kBM];
    float ainv[kBM];
};

__device__ __forceinline__ void ldmatrix_x4(std::uint32_t (&r)[4], const void * p) {
    const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
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

__device__ __forceinline__ void cp_async16(void * dst, const void * src, bool valid) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

__device__ __forceinline__ float row_scale(float m) {
    if (!(m > 0.0f) || !isfinite(m)) return 1.0f;
    int e;
    frexpf(m, &e);  // m = f * 2^e, f in [0.5, 1)
    return ldexpf(1.0f, min(max(15 - e, -100), 100));
}

template <bool kDown, GgufType W, int LIMBS>
__global__ void __launch_bounds__(kThreads, 1) k_tc(const Params P) {
    using Sub = typename SubOf<W>::T;
    constexpr int K = kDown ? kExpertFF : kEmbd, KT = K / kBK;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    auto & S = *reinterpret_cast<TcSmem<W, LIMBS> *>(smem_raw);
    if (int(blockIdx.y) >= P.counts[0]) return;
    const int4 tile = P.tiles[blockIdx.y];
    const int tid = threadIdx.x;
    if constexpr (W == GgufType::IQ3_S) {
        for (int i = tid; i < 512; i += kThreads) {
            const std::uint32_t g = d_grid_b[i];
            S.grid[i] = make_uint2(pack_half2(float(g & 0xFF), float((g >> 8) & 0xFF)), pack_half2(float((g >> 16) & 0xFF), float(g >> 24)));
        }
        if (tid < 16)
            S.signs[tid] = make_uint2((tid & 1 ? 0x8000u : 0u) | (tid & 2 ? 0x80000000u : 0u),
                                      (tid & 4 ? 0x8000u : 0u) | (tid & 8 ? 0x80000000u : 0u));
    } else if constexpr (W != GgufType::Q8_0) {
        S.lut2[tid] = pack_half2(float(c_iq4nl_b[tid & 0xF]), float(c_iq4nl_b[tid >> 4]));
    }
    const __half * planes = kDown ? P.hs : P.xs;
    const std::size_t limb = kDown ? std::size_t(kExpertFF) : P.xs_limb;
    if (tid < kBM) {
        const __half * row = planes;  // rows past the tile: any valid address, zero-filled
        float inv = 0.0f;
        if (tid < tile.z) {
            const int pos = tile.y + tid;
            if constexpr (kDown) {
                row = P.hs + std::size_t(pos) * 2 * kExpertFF;
                inv = P.hinv[pos];
            } else {
                const int t = P.order[pos] / kUsed;
                row = P.xs + std::size_t(t) * kEmbd;
                inv = P.xinv[t];
            }
        }
        S.arows[tid] = row;
        S.ainv[tid] = inv;
    }
    __syncthreads();
    const F16Tables tables{S.grid, S.signs, S.lut2};
    const int ar = tid >> 2, achunk = tid & 3;  // cp.async: thread copies chunk achunk of row ar, every limb
    const __half * arow = S.arows[ar];
    const bool avalid = ar < tile.z;
    const std::uint8_t *codes, *scales;
    weight_row<kDown, W>(P, tile.x, tid, codes, scales);

    Sub rb0, rb1;  // even and odd sub-blocks
    auto copy_a = [&](int buf, int kt) {
#pragma unroll
        for (int l = 0; l < LIMBS; ++l)
            cp_async16(&S.a[buf][l][ar][swz(ar, achunk)], arow + l * limb + kt * kBK + 8 * achunk, avalid);
        cp_async_commit();
    };
    auto stash_b = [&](int buf, const Sub & rb) {
        uint4 w[4];
        S.scale[buf][tid] = rb.decode_f16(tables, w);
#pragma unroll
        for (int c = 0; c < 4; ++c) S.b[buf][tid][swz(tid, c)] = w[c];
    };

    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;  // warps 0-3 take rows 0..31: one per SM sub-partition
    const int mrows = tile.z - wm * 32;  // rows of this warp inside the tile
    float acc[2][8][4] = {};
    // Stage kt: the activations of kt + 1 copy in and the weights of kt + 2 load during the math; the
    // weights of kt + 1 are decoded halfway through the MMAs (which keep running while the decoding
    // issues), not after them: every warp leaves the barrier together, so decoding after the math would
    // leave the tensor cores idle meanwhile.
    auto step = [&](int kt, const Sub & next, Sub & future) {
        const int buf = kt & 1;
        const bool more = kt + 1 < KT;
        if (more) copy_a(buf ^ 1, kt + 1);
        if (kt + 2 < KT) future.load(codes, scales, kt + 2);
        if (mrows > 0) {
            std::uint32_t af[2][2][LIMBS][4];
#pragma unroll
            for (int i = 0; i < 2; ++i)
                if (i * 16 < mrows)
#pragma unroll
                    for (int kk = 0; kk < 2; ++kk) {
                        const int r = wm * 32 + i * 16 + (lane & 15);
#pragma unroll
                        for (int l = 0; l < LIMBS; ++l) ldmatrix_x4(af[i][kk][l], &S.a[buf][l][r][swz(r, 2 * kk + (lane >> 4))]);
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
                    if constexpr (LIMBS == 2) {
                        std::uint32_t tl[2] = {0u, 0u};
                        mma_f16(tl, af[i][0][1], bf[0], bf[1]);
                        mma_f16(tl, af[i][1][1], bf[2], bf[3]);
                        const float2 l01 = __half22float2(*reinterpret_cast<const __half2 *>(&tl[0]));
                        const float2 l23 = __half22float2(*reinterpret_cast<const __half2 *>(&tl[1]));
                        t[0] += l01.x, t[1] += l01.y, t[2] += l23.x, t[3] += l23.y;
                    }
                    acc[i][jn][0] = fmaf(sc.x, t[0], acc[i][jn][0]);
                    acc[i][jn][1] = fmaf(sc.y, t[1], acc[i][jn][1]);
                    acc[i][jn][2] = fmaf(sc.x, t[2], acc[i][jn][2]);
                    acc[i][jn][3] = fmaf(sc.y, t[3], acc[i][jn][3]);
                }
                if (jn == 3 && more) stash_b(buf ^ 1, next);
            }
        } else if (more) {
            stash_b(buf ^ 1, next);
        }
        cp_async_wait_all();
        __syncthreads();
    };
    copy_a(0, 0);
    rb0.load(codes, scales, 0);
    rb1.load(codes, scales, 1);
    stash_b(0, rb0);
    cp_async_wait_all();
    __syncthreads();
    for (int kt = 0; kt < KT; kt += 2) {
        step(kt, rb1, rb0);
        step(kt + 1, rb0, rb1);
    }
    if (mrows <= 0) return;
    // C fragment: c[0], c[1] at row lane / 4, columns 2 (lane % 4) + {0, 1}; c[2], c[3] eight rows lower
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int r = wm * 32 + i * 16 + half * 8 + (lane >> 2);
            const bool ok = r < tile.z;
            const float inv = S.ainv[r];
            if constexpr (kDown) {
                if (!ok) continue;
                float * yr = P.y + std::size_t(P.order[tile.y + r]) * kEmbd + blockIdx.x * kBN + 2 * (lane & 3);
#pragma unroll
                for (int jn = 0; jn < 8; ++jn) {
                    const int nb = (jn < 4 ? 0 : kHalfN) + wn * 32 + (jn & 3) * 8;
                    *reinterpret_cast<float2 *>(yr + nb) = make_float2(acc[i][jn][2 * half] * inv, acc[i][jn][2 * half + 1] * inv);
                }
            } else {
                if (!ok) continue;
                float * hr = P.h + std::size_t(tile.y + r) * kExpertFF + blockIdx.x * kHalfN + wn * 32 + 2 * (lane & 3);
#pragma unroll
                for (int jn = 0; jn < 4; ++jn)
                    *reinterpret_cast<float2 *>(hr + jn * 8) =
                        make_float2(silu_mul(acc[i][jn][2 * half] * inv, acc[i][jn + 4][2 * half] * inv),
                                    silu_mul(acc[i][jn][2 * half + 1] * inv, acc[i][jn + 4][2 * half + 1] * inv));
            }
        }
    }
}

// A row of N floats -> its fp16 terms times row_scale(max |row|), and 1 / that scale; a warp per row.
// The whole row is read before anything is written, so the terms may overwrite the row (h).
template <int N, int LIMBS>
__device__ __forceinline__ void split_row(const float * src, __half * hi, std::size_t limb, float * inv) {
    constexpr int V = N / 128;  // float4 per lane
    const int lane = threadIdx.x & 31;
    float4 v[V];
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        v[i] = reinterpret_cast<const float4 *>(src)[lane + 32 * i];
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v[i].x), fabsf(v[i].y)), fmaxf(fabsf(v[i].z), fabsf(v[i].w))));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float s = row_scale(m);
    __syncwarp();
#pragma unroll
    for (int i = 0; i < V; ++i) {
        const float4 w = make_float4(v[i].x * s, v[i].y * s, v[i].z * s, v[i].w * s);
        const __half2 h01 = __floats2half2_rn(w.x, w.y), h23 = __floats2half2_rn(w.z, w.w);
        reinterpret_cast<uint2 *>(hi)[lane + 32 * i] =
            make_uint2(*reinterpret_cast<const std::uint32_t *>(&h01), *reinterpret_cast<const std::uint32_t *>(&h23));
        if constexpr (LIMBS == 2)
            reinterpret_cast<uint2 *>(hi + limb)[lane + 32 * i] =
                make_uint2(pack_half2(w.x - __low2float(h01), w.y - __high2float(h01)), pack_half2(w.z - __low2float(h23), w.w - __high2float(h23)));
    }
    if (lane == 0) *inv = 1.0f / s;
}

// Tensor-core modes, before gate/up: x -> fp16 terms ([limb][token][2560]), a warp per token.
template <int LIMBS>
__global__ void __launch_bounds__(256) k_split_x(const float * __restrict__ x, int n, __half * __restrict__ xs, std::size_t limb,
                                                 float * __restrict__ xinv) {
    const int t = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (t >= n) return;
    split_row<kEmbd, LIMBS>(x + std::size_t(t) * kEmbd, xs + std::size_t(t) * kEmbd, limb, xinv + t);
}

// Tensor-core modes, between gate/up and down: h -> fp16 terms in place ([pos][limb][640]), a warp per
// cached position.
template <int LIMBS>
__global__ void __launch_bounds__(256) k_split_h(float * h, const int * __restrict__ counts, float * __restrict__ hinv) {
    const int pos = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (pos >= counts[1]) return;
    float * row = h + std::size_t(pos) * kExpertFF;
    split_row<kExpertFF, LIMBS>(row, reinterpret_cast<__half *>(row), kExpertFF, hinv + pos);
}

// out[t] = sum over k = 0..9 of the cached pairs' weight * y, in that order.
__global__ void __launch_bounds__(kEmbd / 4) k_reduce(const std::int32_t * __restrict__ slots, const float * __restrict__ weights,
                                                      const float * __restrict__ y, float * __restrict__ out) {
    const int t = blockIdx.x, c = threadIdx.x;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int k = 0; k < kUsed; ++k) {
        const int p = t * kUsed + k;
        if (!cached(slots[p])) continue;
        const float w = weights[p];
        const float4 v = __ldcs(reinterpret_cast<const float4 *>(y + std::size_t(p) * kEmbd) + c);
        acc.x = fmaf(w, v.x, acc.x);
        acc.y = fmaf(w, v.y, acc.y);
        acc.z = fmaf(w, v.z, acc.z);
        acc.w = fmaf(w, v.w, acc.w);
    }
    reinterpret_cast<float4 *>(out + std::size_t(t) * kEmbd)[c] = acc;
}

template <class Kernel>
void set_smem(Kernel k, std::size_t bytes) {
    check(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, int(bytes)), "experts_batch smem");
    check(cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, kCarveout), "experts_batch carveout");
}

template <bool kDown, GgufType W>
void init_kernels_for() {
    static_assert(sizeof(F32Smem<W>) <= kCarveout * 1024 - 1024 && sizeof(TcSmem<W, 2>) <= kCarveout * 1024 - 1024,
                  "shared memory over the carveout");
    set_smem(k_f32<kDown, W>, sizeof(F32Smem<W>));
    set_smem(k_tc<kDown, W, 1>, sizeof(TcSmem<W, 1>));
    set_smem(k_tc<kDown, W, 2>, sizeof(TcSmem<W, 2>));
}

template <bool kDown, GgufType W>
void launch(BatchMath math, dim3 grid, const Params & P, cudaStream_t s) {
    switch (math) {
    case BatchMath::Fp32: k_f32<kDown, W><<<grid, kThreads, sizeof(F32Smem<W>), s>>>(P); break;
    case BatchMath::Fp16x2: k_tc<kDown, W, 2><<<grid, kThreads, sizeof(TcSmem<W, 2>), s>>>(P); break;
    case BatchMath::Fp16: k_tc<kDown, W, 1><<<grid, kThreads, sizeof(TcSmem<W, 1>), s>>>(P); break;
    }
}

std::once_flag init_once;

}  // namespace

std::size_t experts_batch_workspace_bytes(int max_tokens) {
    if (max_tokens < 1 || max_tokens > kMaxBatchTokens) throw std::runtime_error("experts_batch: max_tokens out of range");
    return carve(nullptr, max_tokens).bytes;
}

void experts_batch_init() {
    std::call_once(init_once, [] {
        check(cudaMemcpyToSymbol(d_grid_b, k_iq3s_grid, sizeof(k_iq3s_grid)), "experts_batch grid");
        init_kernels_for<false, GgufType::IQ3_S>();
        init_kernels_for<false, GgufType::IQ4_XS>();
        init_kernels_for<true, GgufType::IQ4_NL>();
        init_kernels_for<true, GgufType::Q8_0>();
    });
}

void experts_gpu_batch(const ExpertLayout & l, const std::uint8_t * pool, int n_tokens, const float * x, const std::int32_t * slots,
                       const float * weights, float * out, void * workspace, cudaStream_t stream, BatchMath math) {
    if (n_tokens <= 0) return;
    if (n_tokens > kMaxBatchTokens) throw std::runtime_error("experts_gpu_batch: too many tokens");
    if (l.gate_type != GgufType::IQ3_S && l.gate_type != GgufType::IQ4_XS) throw std::runtime_error("experts_gpu_batch: gate/up type");
    if (l.down_type != GgufType::IQ4_NL && l.down_type != GgufType::Q8_0) throw std::runtime_error("experts_gpu_batch: down type");
    experts_batch_init();
    const Workspace w = carve(workspace, n_tokens);
    const int pairs = n_tokens * kUsed;
    const Params P{pool, l.slot_bytes, l.gate_row, l.up_off, l.down_off, l.scale_off, w.counts, w.tiles, w.order, x, w.h, w.y,
                   w.xs, w.hs,         w.xinv,     w.hinv,   std::size_t(n_tokens) * kEmbd};
    const bool two = math == BatchMath::Fp16x2;
    k_group<<<1, kExperts, 0, stream>>>(slots, pairs, w.counts, w.tiles, w.order);
    check(cudaGetLastError(), "experts_batch group");
    if (math != BatchMath::Fp32) {
        if (two) k_split_x<2><<<(n_tokens + 7) / 8, 256, 0, stream>>>(x, n_tokens, w.xs, P.xs_limb, w.xinv);
        else k_split_x<1><<<(n_tokens + 7) / 8, 256, 0, stream>>>(x, n_tokens, w.xs, P.xs_limb, w.xinv);
        check(cudaGetLastError(), "experts_batch split x");
    }
    const unsigned tiles = unsigned(max_tiles(pairs));
    const dim3 gu(kExpertFF / kHalfN, tiles), dn(kEmbd / kBN, tiles);
    if (l.gate_type == GgufType::IQ3_S) launch<false, GgufType::IQ3_S>(math, gu, P, stream);
    else launch<false, GgufType::IQ4_XS>(math, gu, P, stream);
    check(cudaGetLastError(), "experts_batch gate/up");
    if (math != BatchMath::Fp32) {
        if (two) k_split_h<2><<<(pairs + 7) / 8, 256, 0, stream>>>(w.h, w.counts, w.hinv);
        else k_split_h<1><<<(pairs + 7) / 8, 256, 0, stream>>>(w.h, w.counts, w.hinv);
        check(cudaGetLastError(), "experts_batch split h");
    }
    if (l.down_type == GgufType::IQ4_NL) launch<true, GgufType::IQ4_NL>(math, dn, P, stream);
    else launch<true, GgufType::Q8_0>(math, dn, P, stream);
    check(cudaGetLastError(), "experts_batch down");
    k_reduce<<<n_tokens, kEmbd / 4, 0, stream>>>(slots, weights, w.y, out);
    check(cudaGetLastError(), "experts_batch reduce");
}

}  // namespace ninfer::flashnext::cuda
