// GPU kernels of the Qwen3-VL vision encoder (src/flashnext/vision.h): FP32 activations, weights exact.
//
// Patches are in the merge order of the frontend: 2x2 blocks of the patch grid row by row, the four
// patches of a block (0,0), (0,1), (1,0), (1,1). Patch p of a gh x gw grid sits at row 2*by + (p%4)/2,
// column 2*bx + p%2, where block p/4 = (by, bx) of the (gh/2) x (gw/2) block grid.
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ninfer::flashnext::cuda {

constexpr int kVisHidden = 1152;
constexpr int kVisHeads = 16;
constexpr int kVisHeadDim = kVisHidden / kVisHeads;  // 72
constexpr int kVisFF = 4304;
constexpr int kVisPatchIn = 3 * 2 * 16 * 16;         // 1536: channels x temporal x 16 x 16
constexpr int kVisMergeIn = 4 * kVisHidden;          // 4608: a 2x2 block of patches

// One-time setup (the 2-D rope's frequencies).
void vision_init();

// dst[i] = float(src[i]), n values
void f16_to_f32(const half * src, float * dst, std::size_t n, cudaStream_t s);

// x [P][1152] += bias + the learned position embedding (a side x side grid, [side*side][1152]) resized to the gh x gw
// patch grid by bilinear interpolation with aligned corners (llama.cpp's ggml_interpolate, which matches Qwen3-VL's
// fast_pos_embed_interpolate), each patch at its grid cell (merge order).
void vis_embed_add(float * x, const float * bias, const float * pos_table, int side, int gh, int gw, cudaStream_t s);

// y [P][n] = LayerNorm(x [P][n]) * w + b
void vis_layer_norm(const float * x, const float * w, const float * b, float * y, int P, int n, float eps, cudaStream_t s);

// qkv [P][3*1152] += bias, then the 2-D rope on q and k: head dims (i, i+36) turn by row * f_i (i < 18) or
// column * f_(i-18) (18 <= i < 36), f_j = 10000^(-j/18), each patch at its grid cell (merge order).
void vis_qkv_rope(float * qkv, const float * bias, int gw, int P, cudaStream_t s);

// out [P][1152] = softmax(q k^T / sqrt(72)) v per head over all P patches (bidirectional), from qkv [P][3*1152].
void vis_attention(const float * qkv, float * out, int P, cudaStream_t s);

// x [rows][n] += bias (broadcast over rows)
void vis_bias_add(float * x, const float * bias, int rows, int n, cudaStream_t s);
// x [rows][n] = gelu(x + bias): tanh approximation (the ViT's MLP) or exact erf (the merger)
void vis_bias_gelu(float * x, const float * bias, int rows, int n, bool exact, cudaStream_t s);

}  // namespace ninfer::flashnext::cuda
