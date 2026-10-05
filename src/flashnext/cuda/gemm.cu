#include "flashnext/cuda/gemm.h"

#include "flashnext/cuda/gemm_tc.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace ninfer::flashnext::cuda {

namespace {

void cublas_check(cublasStatus_t st, const char * what) {
    if (st != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(int(st)));
}

// Q8 split planes -> FP32 rows (exact: int8 code times the fp16 block scale)
__global__ void k_dequant_q8(const std::int8_t * __restrict__ codes, const __half * __restrict__ scales, float * __restrict__ out,
                             std::size_t n, int k) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const std::size_t row = i / k;
    const int col = int(i % k);
    out[i] = float(codes[i]) * __half2float(scales[row * (k / 32) + col / 32]);
}

__global__ void k_dequant_bf16(const __nv_bfloat16 * __restrict__ w, float * __restrict__ out, std::size_t n) {
    const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(w[i]);
}

}  // namespace

Gemm::Gemm(std::size_t max_weight_elems, cudaStream_t stream)
    : stream_(stream), scratch_(max_weight_elems * sizeof(float)), row_scales_(std::size_t(16384) * 2 * sizeof(float)) {
    cublas_check(cublasCreate(&handle_), "cublasCreate");
    cublas_check(cublasSetStream(handle_, stream_), "cublasSetStream");
    // FP32 SGEMM; the default math mode never rounds the inputs to TF32 (that needs CUBLAS_TF32_TENSOR_OP_MATH)
    cublas_check(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH), "cublasSetMathMode");
}

Gemm::~Gemm() {
    if (handle_) cublasDestroy(handle_);
}

void Gemm::run(const GpuWeight & w, const float * x, float * y, int T) {
    // the tensor-core kernel puts a 64 x 256 tile on each block: a long, narrow product (the hyper-connection
    // down projection, K = 10240, N = 320) with few tiles leaves most SMs idle, where SGEMM splits K
    const long long tiles = ((w.n + 255) / 256) * ((T + 63) / 64);
    const bool narrow_long = w.k >= 8192 && tiles < 128;
    if (tc_ && !narrow_long && w.format == WeightFormat::Q8_SPLIT && gemm_q8_tc_supported(w.n, w.k) &&
        row_scales_.bytes() >= std::size_t(T) * 2 * sizeof(float)) {
        gemm_q8_tc(static_cast<const std::int8_t *>(w.data), w.scales, w.n, w.k, x, y, T, row_scales_.as<float>(), stream_);
        return;
    }
    const std::size_t n = std::size_t(w.n) * w.k;
    const float * wf = nullptr;
    switch (w.format) {
    case WeightFormat::F32:
        wf = static_cast<const float *>(w.data);
        break;
    case WeightFormat::Q8_SPLIT:
    case WeightFormat::BF16:
        if (n * sizeof(float) > scratch_.bytes()) throw std::runtime_error("gemm: weight larger than the scratch buffer");
        if (w.format == WeightFormat::Q8_SPLIT)
            k_dequant_q8<<<unsigned((n + 255) / 256), 256, 0, stream_>>>(static_cast<const std::int8_t *>(w.data),
                                                                          static_cast<const __half *>(w.scales), scratch_.as<float>(), n, w.k);
        else
            k_dequant_bf16<<<unsigned((n + 255) / 256), 256, 0, stream_>>>(static_cast<const __nv_bfloat16 *>(w.data), scratch_.as<float>(), n);
        check(cudaGetLastError(), "gemm dequant");
        wf = scratch_.as<float>();
        break;
    default:
        throw std::runtime_error("gemm: no batched path for this weight format");
    }
    // row-major Y[T][N] = X[T][K] W[N][K]^T, i.e. column-major Y^T (N x T) = W^T(N x K) X^T(K x T)
    const float one = 1.0f, zero = 0.0f;
    cublas_check(cublasSgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, w.n, T, w.k, &one, wf, w.k, x, w.k, &zero, y, w.n), "cublasSgemm");
}

}  // namespace ninfer::flashnext::cuda
