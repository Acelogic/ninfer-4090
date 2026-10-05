// Where attention K/V rows live: the store targets of attn_prep.
#pragma once
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ninfer::flashnext::cuda {

// Where attn_prep stores the K/V rows ([2][256] fp16 per position) of a step's tokens.
struct KvStore {
    half * k = nullptr;  // row (pos % ring) of k and v; ring 0: row pos
    half * v = nullptr;
    std::int64_t ring = 0;
};

}  // namespace ninfer::flashnext::cuda
