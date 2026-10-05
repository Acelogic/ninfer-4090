// Prompt-sized products with Q8_0 weights (the Q8_SPLIT planes of gemv.h) on the tensor cores: exact weights,
// activations as two fp16 terms, FP32 accumulation of the main terms (see gemm_tc.cu). An alternative to
// dequantizing into FP32 and cuBLAS SGEMM (Gemm), about 4e-7 relative error against exact math.
#pragma once
#include <cstdint>

#include <cuda_runtime.h>

namespace ninfer::flashnext::cuda {

// n output rows, k inputs (a multiple of 32; n even).
bool gemm_q8_tc_supported(int n, int k);

// y[T][n] = x[T][k] w[n][k]^T. codes: int8 [n][k]; scales: fp16 [n][k/32]; row_scales: 2 * T floats of
// device scratch. Asynchronous on s.
void gemm_q8_tc(const std::int8_t * codes, const void * scales, int n, int k, const float * x, float * y, int T, float * row_scales,
                cudaStream_t s);

}  // namespace ninfer::flashnext::cuda
