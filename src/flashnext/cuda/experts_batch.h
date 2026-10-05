// Routed experts of Qwen3.8-Flash-Next for a chunk of prompt tokens, over the experts held in the
// VRAM expert cache (slot layout: experts.h).
//
// A chunk of n tokens makes n * 10 (token, expert) pairs and touches nearly every expert, so instead
// of one matrix-vector product per pair the cached pairs are grouped by slot on the GPU and every
// slot runs two small GEMMs over its m tokens: gate/up [m x 2560] x [2560 x 1280] with SwiGLU, then
// down [m x 640] x [640 x 2560]. The weights are dequantized exactly inside the kernels.
//
// Every pair's result goes to its own row and each token adds its pairs in k order 0..9, so the
// result does not depend on how the work was scheduled: runs are bitwise reproducible.
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "flashnext/cuda/experts.h"

namespace ninfer::flashnext::cuda {

// How the GEMMs multiply. Every mode uses the exact weights and FP32 accumulation; they differ in how
// the activations enter. Relative error of a token's output against exact math, measured on real
// experts of every format (test_gpu_experts_batch): Fp32 ~1.6e-6, Fp16x2 ~4e-7, Fp16 ~5e-4.
enum class BatchMath {
    Fp32,    // FP32 FMA on the CUDA cores (sequential K sums); about 1.7x slower than Fp16x2
    Fp16x2,  // tensor cores; each activation as two fp16 terms (hi + lo, ~22 bits): FP32-class accuracy
    Fp16,    // tensor cores; activations rounded to fp16 once: about 15% faster than Fp16x2
};

// Workspace for up to max_tokens (1 .. 16384) tokens per call: about 128 KiB per token (one FP32 row
// of 2560 per pair dominates; 525 MiB for 4096 tokens).
std::size_t experts_batch_workspace_bytes(int max_tokens);

// Uploads the IQ3_S grid and sets kernel attributes; call once before capturing a graph.
void experts_batch_init();

// out[t] = sum over k = 0..9 with slots[t*10+k] cached of weights[t*10+k] * expert(x[t]), where
// expert(v) = down(silu(gate v) * up v) is the expert in that pool slot. A slot is cached when
// 0 <= slot < kExperts (slots index the layer's pool; -1 = computed elsewhere). Tokens without a
// cached pair get 0; the caller adds the CPU's part. x and out are [n_tokens][2560] device arrays.
// workspace: experts_batch_workspace_bytes(m) bytes with m >= n_tokens, 256-byte aligned.
// Everything runs on `stream` with no host synchronization and grids that depend on n_tokens only,
// so a call can be captured in a CUDA graph (one graph per n_tokens).
void experts_gpu_batch(const ExpertLayout & layout, const std::uint8_t * pool, int n_tokens, const float * x,
                       const std::int32_t * slots, const float * weights, float * out, void * workspace, cudaStream_t stream,
                       BatchMath math = BatchMath::Fp16x2);

// The same computation in three phases, for weights that arrive while a layer runs (prompt chunks that
// stream experts from host memory). Pairs are grouped by key: keys[t*10+k] in [0, 512), or < 0 for a
// pair computed elsewhere. experts_phased_run computes the pairs whose key is in [key0, key1), reading
// key k's weights from ptrs[k] (a device array of slot addresses in `layout`; only the entries of the
// range are read, when the kernels run). Call begin once, run for disjoint key ranges in any order (each
// pair is computed once), then end. A pair's result does not depend on its key, on the ranges or on
// where its weights live, so out is bitwise the same as experts_gpu_batch's for the same pairs.
// workspace: experts_phased_workspace_bytes(m) bytes, m >= n_tokens (about 136 KiB per token).
std::size_t experts_phased_workspace_bytes(int max_tokens);
void experts_phased_begin(int n_tokens, const float * x, const std::int32_t * keys, void * workspace, cudaStream_t stream,
                          BatchMath math = BatchMath::Fp16x2);
void experts_phased_run(const ExpertLayout & layout, const std::uint8_t * const * ptrs, int key0, int key1, int n_tokens, const float * x,
                        void * workspace, cudaStream_t stream, BatchMath math = BatchMath::Fp16x2);
void experts_phased_end(int n_tokens, const std::int32_t * keys, const float * weights, float * out, void * workspace, cudaStream_t stream);

}  // namespace ninfer::flashnext::cuda
