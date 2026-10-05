// KV streaming kernels: see kv_stream.h.
//
// k_kv_resolve and k_kv_copy are adapted from Strata's resolve_kernel / copy_kernel
// (github.com/Niko1221/Strata, src/kernels/cuda/kv_stream.cu):
//
//   MIT License
//   Copyright (c) 2026 Niko1221 and the Strata contributors
//   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
//   associated documentation files (the "Software"), to deal in the Software without restriction, including
//   without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//   copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
//   following conditions: The above copyright notice and this permission notice shall be included in all
//   copies or substantial portions of the Software.
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
//   LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
//   EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
//   IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//   USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "flashnext/cuda/kv_stream.h"

#include <stdexcept>
#include <string>

#include "flashnext/cuda/device.h"

namespace ninfer::flashnext::cuda {

namespace {

constexpr int RT = kKvResolveThreads;
constexpr int kRowHalves = 2 * 256;                  // one cell of K (or V): 2 kv heads x 256
constexpr int kPageHalves = kKvPage * kRowHalves;    // 2,048 halves = 4 KiB per page of K (or V)

void launched(const char * what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("kv_stream ") + what + ": " + cudaGetErrorString(e));
}

// Block-wide exclusive prefix sum of one int per thread (RT threads); total = the sum over the block.
__device__ int block_scan(int v, int * warp_sums, int & total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[w] = x;
    __syncthreads();
    if (w == 0) {
        int t = warp_sums[lane];
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, t, o);
            if (lane >= o) t += y;
        }
        warp_sums[lane] = t;
    }
    __syncthreads();
    total = warp_sums[31];
    const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    __syncthreads();
    return excl;
}

// One CUDA block of RT threads. Mode 1 (cells != nullptr): the blocks named by T selections; mode 2: the
// blocks of positions [0, *pos0 + T).
__global__ void __launch_bounds__(RT) k_kv_resolve(KvStreamMap m, std::int32_t * __restrict__ cells, const std::int32_t * __restrict__ n_cells,
                                                   int T, int width, const std::int64_t * __restrict__ pos0p, bool translate) {
    __shared__ int s_nmiss, s_lookups, s_cut, s_over;
    __shared__ int warp_sums[32];
    const int epoch = m.ctl[0] + 1;
    const int n = int(m.n_slots);
    if (threadIdx.x == 0) {
        s_nmiss = 0;
        s_lookups = 0;
        s_over = 0;
    }
    __syncthreads();
    // 1. a resident block takes this call's epoch and its reference bit; a missing block is claimed exactly
    //    once (-1 -> -2) into the miss list
    int lookups = 0;
    auto lookup = [&](int b) {
        ++lookups;
        const int sl = m.page_table[b];
        if (sl >= 0) {
            m.slot_stamp[sl] = epoch;
            m.slot_ref[sl] = 1;
        } else if (sl == -1 && atomicCAS(&m.page_table[b], -1, -2) == -1) {
            const int k = atomicAdd(&s_nmiss, 1);
            if (k < n) {
                m.miss_block[k] = b;
            } else {  // more distinct blocks than slots: leave it unmapped
                m.page_table[b] = -1;
                s_over = 1;
            }
        }
    };
    if (cells) {
        for (int t = 0; t < T; ++t) {
            const int nc = n_cells[t];
            const std::int32_t * c = cells + std::size_t(t) * width;
            for (int i = threadIdx.x; i < nc; i += RT) {
                const int b = c[i] / kKvPage;
                if (i > 0 && c[i - 1] / kKvPage == b) continue;  // cells ascend: one lookup per block
                lookup(b);
            }
        }
    } else {
        const int nb = int((*pos0p + T + kKvPage - 1) / kKvPage);
        for (int b = threadIdx.x; b < nb; b += RT) lookup(b);
    }
    atomicAdd(&s_lookups, lookups);
    __syncthreads();
    // 2. one victim per miss, by a clock sweep from the hand. A slot this call uses (stamp == epoch) is never
    //    taken; a referenced one loses its bit as the hand passes and is taken on a later pass.
    const int need = min(s_nmiss, n);
    int hand = m.ctl[1], got = 0;
    for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
        const int j = int((std::int64_t(hand) + threadIdx.x) % n);
        const bool mine = m.slot_stamp[j] == epoch;
        const bool cand = !mine && (m.slot_block[j] < 0 || m.slot_ref[j] == 0);
        int total = 0;
        const int rank = block_scan(cand ? 1 : 0, warp_sums, total);
        const int want = need - got;
        if (threadIdx.x == 0) s_cut = RT;
        __syncthreads();
        if (cand && rank == want - 1) s_cut = threadIdx.x + 1;  // the hand stops just past the last slot taken
        __syncthreads();
        const int cut = s_cut;
        if (cand && rank < want) {
            m.miss_slot[got + rank] = j;
            m.slot_stamp[j] = epoch;  // taken: a sweep that wraps around must not take it twice
        } else if (threadIdx.x < cut && !mine) {
            m.slot_ref[j] = 0;
        }
        got += total < want ? total : want;
        hand = int((std::int64_t(hand) + cut) % n);
        __syncthreads();
    }
    // 3. re-point the table; kv_copy_misses fills the slots
    const int placed = got < need ? got : need;
    for (int k = threadIdx.x; k < need; k += RT) {
        const int b = m.miss_block[k];
        if (k >= placed) {  // overflow: never happens with a legal n_slots
            m.page_table[b] = -1;
            continue;
        }
        const int sl = m.miss_slot[k];
        const int old = m.slot_block[sl];
        if (old >= 0) m.page_table[old] = -1;
        m.slot_block[sl] = b;
        m.slot_stamp[sl] = epoch;
        m.slot_ref[sl] = 1;
        m.page_table[b] = sl;
    }
    if (threadIdx.x == 0) {
        m.ctl[0] = epoch;
        m.ctl[1] = hand;
        m.ctl[2] = placed;
        if (placed < need || s_over) m.ctl[3] = 1;
        unsigned long long * c = reinterpret_cast<unsigned long long *>(m.ctl + 4);
        c[0] += (unsigned long long) placed;
        c[1] += (unsigned long long) s_lookups;
        c[2] += 1ull;
    }
    // 4. cells -> pool rows (decode steps: a few thousand cells)
    if (translate && cells) {
        __syncthreads();
        for (int t = 0; t < T; ++t) {
            const int nc = n_cells[t];
            std::int32_t * c = cells + std::size_t(t) * width;
            for (int i = threadIdx.x; i < nc; i += RT) {
                const int cell = c[i], sl = m.page_table[cell / kKvPage];
                c[i] = sl >= 0 ? sl * kKvPage + cell % kKvPage : 0;  // 0 only after an overflow (flagged)
            }
        }
    }
}

__global__ void k_kv_translate(const std::int32_t * __restrict__ page_table, std::int32_t * __restrict__ cells,
                               const std::int32_t * __restrict__ n_cells, int width) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells[t]) return;
    std::int32_t * c = cells + std::size_t(t) * width + i;
    const int cell = *c, sl = page_table[cell / kKvPage];
    *c = sl >= 0 ? sl * kKvPage + cell % kKvPage : 0;
}

// One CUDA block per missed block (grid-stride): its 4 KiB of K and of V from the host copy into its slot,
// 16 bytes per thread and load.
__global__ void k_kv_copy(KvStreamMap m, const uint4 * __restrict__ hk, const uint4 * __restrict__ hv, uint4 * __restrict__ pk,
                          uint4 * __restrict__ pv) {
    constexpr int kPage16 = kPageHalves * 2 / 16;  // 256 uint4 per page
    const int need = m.ctl[2];
    for (int k = blockIdx.x; k < need; k += gridDim.x) {
        const std::size_t b = std::size_t(m.miss_block[k]), sl = std::size_t(m.miss_slot[k]);
        for (int i = threadIdx.x; i < kPage16; i += blockDim.x) {
            pk[sl * kPage16 + i] = hk[b * kPage16 + i];
            pv[sl * kPage16 + i] = hv[b * kPage16 + i];
        }
    }
}

__global__ void k_kv_reset(KvStreamMap m) {
    const std::int64_t i0 = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x, st = std::int64_t(gridDim.x) * blockDim.x;
    for (std::int64_t i = i0; i < m.n_blocks; i += st) m.page_table[i] = -1;
    for (std::int64_t i = i0; i < m.n_slots; i += st) {
        m.slot_block[i] = -1;
        m.slot_stamp[i] = -1;
        m.slot_ref[i] = 0;
    }
    if (i0 < kKvCtlInts) m.ctl[i0] = 0;
}

std::size_t align16(std::size_t n) { return (n + 15) / 16 * 16; }

}  // namespace

std::size_t kv_map_bytes(std::int64_t n_blocks, std::int64_t n_slots) {
    return align16(std::size_t(n_blocks) * 4) + 5 * align16(std::size_t(n_slots) * 4) + align16(kKvCtlInts * 4);
}

KvStreamMap kv_map_layout(void * mem, std::int64_t n_blocks, std::int64_t n_slots) {
    if (n_slots < RT) throw std::runtime_error("kv_stream: a page cache needs at least " + std::to_string(RT) + " pages");
    if (n_blocks >= (std::int64_t(1) << 31) || n_slots >= (std::int64_t(1) << 29)) throw std::runtime_error("kv_stream: map too large");
    KvStreamMap m;
    char * p = static_cast<char *>(mem);
    auto take = [&](std::size_t bytes) {
        std::int32_t * r = reinterpret_cast<std::int32_t *>(p);
        p += align16(bytes);
        return r;
    };
    m.page_table = take(std::size_t(n_blocks) * 4);
    m.slot_block = take(std::size_t(n_slots) * 4);
    m.slot_stamp = take(std::size_t(n_slots) * 4);
    m.slot_ref = take(std::size_t(n_slots) * 4);
    m.miss_block = take(std::size_t(n_slots) * 4);
    m.miss_slot = take(std::size_t(n_slots) * 4);
    m.ctl = take(kKvCtlInts * 4);
    m.n_blocks = n_blocks;
    m.n_slots = n_slots;
    return m;
}

void kv_map_reset(const KvStreamMap & m, cudaStream_t s) {
    k_kv_reset<<<128, 256, 0, s>>>(m);
    launched("reset");
}

void kv_resolve(const KvStreamMap & m, std::int32_t * cells, const std::int32_t * n_cells, int T, int width, bool translate,
                cudaStream_t s) {
    if (T < 1) return;
    k_kv_resolve<<<1, RT, 0, s>>>(m, cells, n_cells, T, width, nullptr, translate);
    launched("resolve");
}

void kv_resolve_prefix(const KvStreamMap & m, const std::int64_t * pos0, int T, cudaStream_t s) {
    k_kv_resolve<<<1, RT, 0, s>>>(m, nullptr, nullptr, T, 0, pos0, false);
    launched("resolve_prefix");
}

void kv_translate(const KvStreamMap & m, std::int32_t * cells, const std::int32_t * n_cells, int T, int width, cudaStream_t s) {
    if (T < 1) return;
    k_kv_translate<<<dim3((width + 255) / 256, T), 256, 0, s>>>(m.page_table, cells, n_cells, width);
    launched("translate");
}

void kv_copy_misses(const KvStreamMap & m, const half * host_k, const half * host_v, half * pool_k, half * pool_v, cudaStream_t s) {
    k_kv_copy<<<96, 128, 0, s>>>(m, reinterpret_cast<const uint4 *>(host_k), reinterpret_cast<const uint4 *>(host_v),
                                 reinterpret_cast<uint4 *>(pool_k), reinterpret_cast<uint4 *>(pool_v));
    launched("copy");
}

KvStreamCounters kv_counters(const KvStreamMap & m) {
    std::int32_t c[kKvCtlInts] = {};
    KvStreamCounters r;
    if (!m.ctl) return r;
    check(cudaMemcpy(c, m.ctl, sizeof(c), cudaMemcpyDeviceToHost), "kv counters");
    const unsigned long long * u = reinterpret_cast<const unsigned long long *>(c + 4);
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];
    r.overflow = c[3] != 0;
    return r;
}

}  // namespace ninfer::flashnext::cuda
