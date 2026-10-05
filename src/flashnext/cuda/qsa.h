// QSA (the qwen4exp block-sparse attention) on the GPU: indexer block keys, indexer queries, the
// per-query cell selection, and softmax attention gathered over the selected cells.
//
// Every function reproduces ReferenceModel::Impl::qsa_select / attention (src/flashnext/reference.cpp)
// and is checked against a port of that code by tools/flashnext/test_qsa.cu. The indexer keys, queries
// and scores are computed with the reference's float operations in the reference's order (no FMA
// contraction, the same dot-product reduction tree), so given the same raw keys and query projection
// the selected cells are identical to the reference's, ties included.
//
// Positions are read from device memory (pos0: position of the first of the T tokens) so that a step
// can be captured as a CUDA graph and replayed; every grid is sized by T and max_ctx only.
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ninfer::flashnext::cuda {

constexpr int kQsaRatio = 4;                          // tokens per indexer block
constexpr int kQsaHeads = 4;                          // indexer query heads
constexpr int kQsaDim = 128;                          // indexer head size
constexpr int kQsaTopK = 2048;
constexpr int kQsaWidth = kQsaTopK + kQsaRatio - 1;   // 2051: cells per query at most
constexpr int kQsaRot = 64;                           // NEOX rotation of the first 64 dims

// 1. Block keys. blocks [max_ctx/4][128] f32 holds, for every complete block b, the mean of raw keys
// 4b..4b+3 (idx_raw [ctx][128] f32), RMS-normed with k_norm [128] and rotated at position 4b. Updates
// the blocks completed by this step's T tokens (those with pos0 <= 4b+3 < pos0+T); idx_raw must already
// hold rows pos0 .. pos0+T-1. rope_inv_freq: [32] doubles, base^(-2i/64), in device memory.
void qsa_update_blocks(const float * idx_raw, const float * k_norm, const double * rope_inv_freq, float * blocks,
                       const std::int64_t * pos0, int T, float eps, cudaStream_t s);
// The same with idx_raw as a ring of raw_rows rows (position p at row p % raw_rows): a block needs its
// raw keys only until it is complete, so raw_rows >= T + 3 suffices instead of the whole context.
void qsa_update_blocks(const float * idx_raw, std::int64_t raw_rows, const float * k_norm, const double * rope_inv_freq, float * blocks,
                       const std::int64_t * pos0, int T, float eps, cudaStream_t s);
// With rope positions (ops.h, kRopeAxes): block b turns with the positions of its first member, row 4b - pos0 of
// rope_pos [T][3], so the table must also hold the 3 rows before it (rope_pos[-9 .. -1]). Null: position 4b.
void qsa_update_blocks(const float * idx_raw, std::int64_t raw_rows, const float * k_norm, const double * rope_inv_freq, float * blocks,
                       const std::int64_t * pos0, const std::int32_t * rope_pos, int T, float eps, cudaStream_t s);

// 2. Indexer queries, in place: q [T][4][128] (the indexer.q_proj output) -> per-head RMSNorm with
// q_norm [128], then rotation at position pos0 + t.
void qsa_query(float * q, const float * q_norm, const double * rope_inv_freq, const std::int64_t * pos0, int T, float eps,
               cudaStream_t s);
// With rope positions rope_pos [T][3] (ops.h, kRopeAxes); null: positions pos0 + t.
void qsa_query(float * q, const float * q_norm, const double * rope_inv_freq, const std::int64_t * pos0, const std::int32_t * rope_pos,
               int T, float eps, cudaStream_t s);

// 3. Selection. For token t at position p = pos0 + t, cells [t][0 .. n_cells[t]) receives the sorted
// cells it attends to: 0..p while p + 1 <= 2051 (dense), otherwise the best 2051 - tail cells of the
// complete blocks (whole blocks by score sum_h relu(q_h . block); ties to the lower block; the last
// block taken may be partial, its lowest cells first) followed by the tail (p+1)/4*4 .. p of the query's
// own incomplete block (empty when p+1 is a multiple of 4). cells: [T][kQsaWidth] int32; n_cells: [T].
// q: the [T][4][128] output of qsa_query. max_ctx: the context capacity the grids are sized for.
// work: qsa_select_work_bytes(T, max_ctx) bytes of device memory, 16-byte aligned (it also covers every
// smaller T). Any T >= 1.
std::size_t qsa_select_work_bytes(int T, std::int64_t max_ctx);
void qsa_select(const float * q, const float * blocks, const std::int64_t * pos0, int T, std::int64_t max_ctx, void * work,
                std::int32_t * cells, std::int32_t * n_cells, cudaStream_t s);
// qsa_select's two halves, for profiling: the block scores into work, then the pick from them
void qsa_scores(const float * q, const float * blocks, const std::int64_t * pos0, int T, std::int64_t max_ctx, void * work, cudaStream_t s);
void qsa_pick(const void * work, const std::int64_t * pos0, int T, std::int64_t max_ctx, std::int32_t * cells, std::int32_t * n_cells,
              cudaStream_t s);

// 4. Attention over the selected cells: for token t and head h (kv head h / 12),
// out = softmax(scale * q . k[cells]) v[cells] * sigmoid(gate). q, gate, out: [T][24][256] f32;
// k_cache, v_cache: [ctx][2][256] f16. work: attn_sparse_work_floats(T) floats, which also covers every
// smaller T. Any T >= 1.
// The cells are split into attn_sparse_splits(T) interleaved runs whose partial softmaxes are merged, so the
// rounding of a token's result depends on T. split_T > 0 uses split_T's split count instead: T queries run in
// groups (as KV streaming does) then round exactly like one call over split_T queries.
std::size_t attn_sparse_work_floats(int T);
int attn_sparse_splits(int T);
void attn_sparse(const float * q, const float * gate, const half * k_cache, const half * v_cache, const std::int32_t * cells,
                 const std::int32_t * n_cells, int T, float scale, float * work, float * out, cudaStream_t s, int split_T = 0);

// One-time kernel setup before capturing a graph (nothing to do at present; kept for the init sequence).
void qsa_init();

}  // namespace ninfer::flashnext::cuda
