// GPU kernels of the Qwen3.8-Flash-Next decode step other than the matrix-vector products.
//
// The shapes are the model's and fixed at compile time; Engine checks them against the GGUF. Every
// formula matches the FP32 reference (src/flashnext/reference.cpp), which documents where it comes
// from. Activations are FP32 [T][width] for T tokens at consecutive positions pos0 .. pos0+T-1.
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace ninfer::flashnext::cuda {

constexpr int kEmbd = 2560;
constexpr int kHc = 4;                 // hyper-connection streams
constexpr int kHcd = kHc * kEmbd;      // 10240
constexpr int kHcRank = 320;

constexpr int kDnState = 128;          // Gated DeltaNet head size
constexpr int kDnKHeads = 16;
constexpr int kDnVHeads = 48;
constexpr int kDnKeyDim = kDnKHeads * kDnState;          // 2048
constexpr int kDnVDim = kDnVHeads * kDnState;            // 6144
constexpr int kDnConvDim = 2 * kDnKeyDim + kDnVDim;      // 10240
constexpr int kDnConv = 4;

constexpr int kHeads = 24;             // full attention
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 256;
constexpr int kRot = 64;               // NEOX rotation of the first 64 dims
constexpr int kGroup = kHeads / kKvHeads;
constexpr int kIdxDim = 128;           // QSA indexer key size

constexpr int kExperts = 512;
constexpr int kUsed = 10;
constexpr int kFfShared = 640;

constexpr int kPleHeads = 16;
constexpr int kPleHeadDim = 160;
constexpr int kPleKernel = 4;
constexpr int kPleDilation = 3;
constexpr int kPleHist = (kPleKernel - 1) * kPleDilation;  // 9 earlier tokens

constexpr int kMaxTokens = 4;          // tokens per decode step (MTP verification needs up to 4)

// ---- hyper-connections ----
// res [T][HC][E] = x [T][E] copied to every stream
void hc_expand(const float * x, float * res, int T, cudaStream_t s);
// xn [T][HC][E] = RMSNorm of each stream of res, times w [HC*E]
void hc_norm(const float * res, const float * w, float * xn, int T, float eps, cudaStream_t s);
// lo = silu(lo / HC), n values
void hc_lowrank_act(float * lo, int n, cudaStream_t s);
// mixed [T][E] = (1/HC) sum_c xn[c] * sigmoid(gate[c])
void hc_gate_mean(const float * xn, const float * gate, float * mixed, int T, cudaStream_t s);
// res[t][c] += out[t] * 2 sigmoid(inject[t][c] / HC)
void hc_combine(float * res, const float * out, const float * inject, int T, cudaStream_t s);

// ---- PLE (layer 1) ----
// From the key projection [T][HCD], the value projection [T][E] and res: gated [T][HCD] =
// value * g[t][c], with g = sigmoid(signed sqrt(rms(key)*nk . rms(res)*nq / sqrt(E))), and
// normalized [T][HCD] = RMSNorm(gated) * nc.
void ple_gate(const float * key, const float * value, const float * res, const float * norm_key, const float * norm_query,
              const float * norm_conv, float * gated, float * normalized, float * gates, int T, float eps, cudaStream_t s);
// res += gated + silu(depthwise causal conv(normalized)); hist [9][HCD] holds the earlier tokens.
void ple_conv_add(float * res, const float * gated, const float * normalized, const float * conv_w, float * hist, int T,
                  cudaStream_t s);

// ---- Gated DeltaNet ----
// conv over [state | qkv] (kernel 4) then SiLU; q and k heads L2-normalised; state updated.
// snap (optional): [T-1][3][conv dim], the state after each token but the last (for rolling back
// rejected draft tokens).
void dn_conv(const float * qkv, float * conv_state, const float * conv_w, float * out, int T, float eps, cudaStream_t s,
             float * snap = nullptr);
// Gated delta rule per v head (k/q head h % 16), then RMSNorm(o) * norm_w * sigmoid(z). S: [48][128][128].
// snap (optional): [T-1][48][128][128], S after each token but the last.
void dn_recurrence(const float * conv_out, const float * z, const float * beta, const float * alpha, const float * dt_bias,
                   const float * a, const float * norm_w, float * S, float * out, int T, float eps, cudaStream_t s,
                   float * snap = nullptr);
// PLE history after keeping only the first n_keep of the last step's T tokens: the last 9 rows of
// [prev (the history before that step) | normalized[0 .. n_keep)].
void ple_hist_rebuild(const float * prev, const float * normalized, float * hist, int n_keep, cudaStream_t s);

// ---- MTP ----
// y[r] = RMSNorm(x[r]) * w over rows of n (n a multiple of 32)
void rms_norm_rows(const float * x, const float * w, float * y, int rows, int n, float eps, cudaStream_t s);
// out[t][c] = [e[t] | h[t][c]] for the 4 hyper-connection streams: [T][4][2 * 2560]
void mtp_concat(const float * e, const float * h, float * out, int T, cudaStream_t s);
// cells[t] = the last min(p + 1, 2051) positions up to p = pos0 + t; n_cells[t] = their count
void window_cells(const std::int64_t * pos0, int T, int width, std::int32_t * cells, std::int32_t * n_cells, cudaStream_t s);
// index of the largest of n values (ties to the lower index)
void argmax(const float * x, int n, std::int32_t * out, cudaStream_t s);

// ---- full attention ----
// Positions are read from device memory (pos0: the first token's position) so that a decode step
// can be replayed as a CUDA graph. rope_inv_freq: [kRot/2] doubles in device memory.
// q_full [T][24][q 256 | gate 256] -> q [T][24][256] (normed, rotated), gate [T][24][256];
// k, v [T][2][256] -> rows pos0.. of the fp16 caches [ctx][2][256] (k normed and rotated).
void attn_prep(const float * q_full, const float * k, const float * v, const float * q_norm, const float * k_norm,
               const double * rope_inv_freq, float * q, float * gate, half * k_cache, half * v_cache, const std::int64_t * pos0, int T,
               float eps, cudaStream_t s);
// Causal softmax attention of T queries over cells [0, pos0 + t] times sigmoid(gate). out [T][24][256].
// Dense attention covers at most kAttnMaxCells cells (beyond that QSA selects 2051 of them), in up to
// kAttnMaxChunks chunks; chunks past the context exit at once. work: attn_work_floats(T) floats.
constexpr int kAttnMaxCells = 2051;
constexpr int kAttnMaxChunks = (kAttnMaxCells + 255) / 256;
std::size_t attn_work_floats(int T);
void attn_decode(const float * q, const float * gate, const half * k_cache, const half * v_cache, const std::int64_t * pos0, int T,
                 float scale, float * work, float * out, cudaStream_t s);
// dst[((pos0 + t) % ring) * row + i] = src[t * row + i]: stores T rows at the device-held position, in a
// ring of `ring` rows
void store_rows(const float * src, float * dst, const std::int64_t * pos0, int row, int T, std::int64_t ring, cudaStream_t s);

// ---- MoE ----
// softmax over 512 logits, top 10 (ties to the lower id), weights renormalised over the ten
void router_topk(const float * logits, std::int32_t * ids, float * weights, int T, cudaStream_t s);
// h = silu(g) * u
void swiglu(const float * g, const float * u, float * h, int n, cudaStream_t s);
// out = moe + shared * sigmoid(shared_gate_logit[t])
void ffn_combine(const float * moe, const float * shared, const float * shared_gate, float * out, int T, cudaStream_t s);
// slots[i] = map[ids[i]] for the T * 10 selected experts (map: the layer's expert -> cache slot, -1 if not cached)
void moe_slots(const std::int32_t * ids, const std::int32_t * map, std::int32_t * slots, int T, cudaStream_t s);
// Assigns the T * 10 selected experts: slots[i] = the VRAM cache slot (or -1); of the rest, the first
// zc_permille / 1000 of them (rounded) are read by the GPU straight from host memory (host_slots[i] =
// expert id, else -1) and the others go to the CPU (on_cpu[i] = 1).
void moe_plan(const std::int32_t * ids, const std::int32_t * map, std::int32_t * slots, std::int32_t * host_slots, std::uint8_t * on_cpu,
              int T, int zc_permille, cudaStream_t s);
// out = (gpu + cpu) + shared * sigmoid(shared_gate_logit[t]), with gpu already summed per token
void moe_combine_sum(const float * gpu, const float * cpu, const float * shared, const float * shared_gate, float * out, int T,
                     cudaStream_t s);
// out[t] = sum_k gpu_pairs[t*10+k] (+ host_pairs[t*10+k]) + cpu[t] + shared[t] * sigmoid(shared_gate_logit[t]);
// host_pairs may be null
void moe_combine(const float * gpu_pairs, const float * host_pairs, const float * cpu, const float * shared, const float * shared_gate,
                 float * out, int T, cudaStream_t s);

// ---- GPU <-> CPU hand-off of the routed experts, through mapped host memory ----
// The GPU writes a layer's selections and FFN input, then raises req; the CPU computes the experts
// that are not cached, writes out, then raises done; the GPU waits for done and continues. Both
// flags carry the step's sequence number, so no reset is needed between steps.
struct alignas(64) ExpertLink {
    std::int64_t req;
    std::int64_t pad0[7];
    std::int64_t done;
    std::int64_t pad1[7];
    std::int32_t ids[kMaxTokens * kUsed];
    float weights[kMaxTokens * kUsed];
    std::uint8_t on_cpu[kMaxTokens * kUsed];  // the pairs the CPU computes (the GPU takes the rest)
    float x[kMaxTokens * kEmbd];
    float out[kMaxTokens * kEmbd];
};
// link: device address of the mapped ExpertLink; seq: device address of the step's sequence number
void link_signal(const std::int32_t * ids, const float * weights, const std::uint8_t * on_cpu, const float * x, int T, ExpertLink * link,
                 const std::int64_t * seq, cudaStream_t s);
// Waits for link->done == *seq, then copies link->out to out [T][2560]. After 1.5 s without an
// answer it gives up, writes zeros and sets *error (Windows resets a GPU after 2 s).
void link_wait(ExpertLink * link, const std::int64_t * seq, float * out, int T, int * error, cudaStream_t s);

// Lazy kernel setup (tables, shared-memory limits); call once before capturing a graph.
void init_kernels();

}  // namespace ninfer::flashnext::cuda
