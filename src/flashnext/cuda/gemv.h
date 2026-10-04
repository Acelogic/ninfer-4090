// Decode-shaped matrix-vector products for Qwen3.8-Flash-Next on the GPU.
//
// y[t][n] = sum_k W[n][k] * x[t][k] for t < T (T = 1..4 tokens, e.g. MTP verification), with FP32
// activations and outputs. Every kernel streams each weight row once for all T tokens; decode is
// bound by weight bandwidth, so FP32 activations cost nothing measurable and keep full precision.
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "flashnext/cuda/device.h"
#include "flashnext/gguf.h"

namespace ninfer::flashnext::cuda {

enum class WeightFormat : std::uint8_t {
    Q8_SPLIT,  // Q8_0 split into two planes: int8 codes [N][K], then fp16 scales [N][K/32]
    F32,       // [N][K]
    BF16,      // [N][K]
    Q6_K,      // ggml Q6_K blocks, rows of K/256 blocks (210 bytes each)
};

struct GpuWeight {
    WeightFormat format{};
    int n = 0;                      // output rows
    int k = 0;                      // input length
    const void * data = nullptr;    // codes (Q8_SPLIT) or the whole tensor
    const void * scales = nullptr;  // Q8_SPLIT only: fp16 [N][K/32]
};

struct DeviceWeight {
    GpuWeight view;
    DeviceBuffer buffer;
};

// Uploads a 2-D GGUF tensor (Q8_0, F32, BF16 or Q6_K) in its GEMV format. Q8_0 is relaid
// losslessly into the split planes so that every load is aligned.
DeviceWeight upload_gemv_weight(const GgufTensor & t);

// x: device FP32 [T][K], y: device FP32 [T][N]. T in 1..4. Asynchronous on `stream`.
void gemv(const GpuWeight & w, const float * x, float * y, int tokens, cudaStream_t stream);

}  // namespace ninfer::flashnext::cuda
