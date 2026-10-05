// Prompt-sized matrix products (many tokens at once) for Qwen3.8-Flash-Next.
//
// Y[T][N] = X[T][K] W[N][K]^T in full FP32: each weight matrix is dequantized exactly into an FP32
// scratch buffer and multiplied with cuBLAS SGEMM (no TF32), so prompt processing computes the same
// math as decoding, only batched.
#pragma once
#include <cstddef>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "flashnext/cuda/device.h"
#include "flashnext/cuda/gemv.h"

namespace ninfer::flashnext::cuda {

class Gemm {
public:
    // max_weight_elems: the largest N*K this instance will see (sizes the scratch buffer)
    Gemm(std::size_t max_weight_elems, cudaStream_t stream);
    ~Gemm();
    Gemm(const Gemm &) = delete;
    Gemm & operator=(const Gemm &) = delete;

    // x: device FP32 [T][K], y: device FP32 [T][N]. Asynchronous on the stream given at construction.
    void run(const GpuWeight & w, const float * x, float * y, int T);
    // Q8_0 weights through gemm_q8_tc (tensor cores, two-term fp16 activations) instead of SGEMM.
    void set_tensor_cores(bool on) { tc_ = on; }

private:
    cublasHandle_t handle_ = nullptr;
    cudaStream_t stream_ = nullptr;
    DeviceBuffer scratch_;
    DeviceBuffer row_scales_;
    bool tc_ = false;
};

}  // namespace ninfer::flashnext::cuda
