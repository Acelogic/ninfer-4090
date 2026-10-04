// Routed experts of Qwen3.8-Flash-Next on the GPU, for the experts held in the VRAM expert cache.
//
// An expert lives in a fixed-size slot: gate rows, up rows, then the down projection. gate/up keep
// the GGUF's blocks (IQ3_S, or IQ4_XS in layer 2) so the cache holds as many experts as possible;
// the down projection (IQ4_NL, or Q8_0 in five layers) is split losslessly into a code plane and an
// fp16 scale plane so that every load is aligned.
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "flashnext/gguf.h"

namespace ninfer::flashnext::cuda {

constexpr int kExpertFF = 640;

struct ExpertLayout {
    GgufType gate_type{}, down_type{};
    std::size_t gate_row = 0;     // bytes per gate/up row (2560 weights)
    std::size_t up_off = 0;       // offsets within a slot
    std::size_t down_off = 0;     // codes [2560][640 or 320]
    std::size_t scale_off = 0;    // fp16 [2560][20]
    std::size_t slot_bytes = 0;
};

// Layout for a layer's expert types: gate/up IQ3_S or IQ4_XS, down IQ4_NL or Q8_0.
ExpertLayout expert_layout(GgufType gate_up, GgufType down);

// Writes expert e of one layer into a slot image of layout.slot_bytes bytes.
void pack_expert(const ExpertLayout & layout, const GgufTensor & gate, const GgufTensor & up, const GgufTensor & down, int e,
                 std::uint8_t * dst);

// Uploads the IQ3_S grid; called by init_kernels().
void experts_init();

// For T tokens with kUsed experts each: pair p = t * 10 + k runs on the GPU when slots[p] >= 0.
//   h[p] = silu(gate . x[t]) * (up . x[t]);   y[p] = weights[p] * (down . h[p])   (y[p] = 0 otherwise)
// pool: the layer's slots; x [T][2560]; h [T*10][640] scratch; y [T*10][2560].
void experts_gpu(const ExpertLayout & layout, const std::uint8_t * pool, const std::int32_t * slots, const float * weights,
                 const float * x, float * h, float * y, int T, cudaStream_t stream);

}  // namespace ninfer::flashnext::cuda
