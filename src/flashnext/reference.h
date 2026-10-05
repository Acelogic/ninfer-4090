// FP32 CPU reference forward pass of Qwen3.8-Flash-Next (GGUF architecture `qwen4exp`).
//
// This is the numerical oracle for every Flash-Next GPU kernel: plain C++, FP32 activations, weights
// dequantized row by row with the bit-exact ggml dequantizers in quants.h. The specification is the
// `qwen4exp` graph of the GenerelSchwerz llama.cpp fork (commit 60bc470); reference.cpp states each
// formula next to its code, including how every ambiguity was resolved.
//
// The model is stateful: forward() consumes new tokens after the ones it has already seen, keeping
// the attention KV cache, the QSA indexer keys, the Gated DeltaNet conv and recurrent states, and
// the PLE conv history. A chunk of tokens is processed as one batch (weights are dequantized once
// per chunk), with the recurrences run token by token, so a prefill matches token-by-token decoding.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "flashnext/gguf.h"

namespace ninfer::flashnext {

struct ReferenceConfig {
    int n_layer = 0;     // trunk layers (the MTP block is not part of this reference)
    int n_embd = 0;      // 2560
    int n_vocab = 0;     // 248320
    float rms_eps = 0;   // 1e-6, every RMS norm and the DeltaNet L2 norm

    // hyper-connections
    int hc = 0;          // 4 parallel residual streams
    int hc_rank = 0;     // 320

    // full attention (layers with (il + 1) % 4 == 0)
    int n_head = 0, n_head_kv = 0, head_dim = 0;  // 24, 2, 256
    int n_rot = 0;                                // 64: NEOX-style rotation of the first 64 dims
    float rope_base = 0;                          // 1e7
    int rope_sections[4] = {0, 0, 0, 0};          // [11, 11, 10, 0], interleaved M-RoPE (imrope)
    float kq_scale = 0;                           // 1 / sqrt(head_dim)

    // QSA indexer
    int idx_n_head = 0, idx_head_dim = 0, idx_top_k = 0;  // 4, 128, 2048
    std::vector<int> compress_ratio;                     // per layer; 4 on attention layers

    // Gated DeltaNet
    int ssm_conv = 0;      // 4
    int ssm_state = 0;     // 128 (head size of q, k and v)
    int ssm_k_heads = 0;   // 16
    int ssm_v_heads = 0;   // 48
    std::vector<bool> recurrent;  // per layer

    // MoE
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_ff_shexp = 0;  // 512, 10, 640, 640
    float expert_weights_scale = 0;  // absent in the GGUF and never read by qwen4exp: no scaling

    // PLE n-gram hash embedding
    int ple_layer = -1;            // 1
    int ple_ngram = 0;             // 3: 2-grams and 3-grams
    int ple_heads_per_ngram = 0;   // 8
    int ple_n_heads = 0;           // 16
    int ple_head_dim = 0;          // 160
    int ple_conv_kernel = 0;       // 4; dilation is ple_ngram
    std::int64_t ple_eos = 0;      // 248044
    std::int64_t ple_image_token = -1;  // 248056 (<|image_pad|>): the id the hash reads at image positions; -1: none
    std::vector<std::uint64_t> ple_multipliers;
    std::vector<std::uint64_t> ple_offsets, ple_vocab;

    static ReferenceConfig from_gguf(const GgufModel & m);
};

// Named intermediate activations, row-major [n_tokens][width], for layer-by-layer comparison.
// layer is -1 for the embedding and the head. pos0 is the position of the first row.
using ActivationHook = std::function<void(const std::string & name, int layer, std::int64_t pos0,
                                          std::int64_t n_tokens, std::int64_t width, const float * data)>;

// Debug teacher forcing: called right after the activation hook for the intermediates that feed later
// computation; returning true means `data` was overwritten (for example with llama.cpp's values), so
// each component can be checked on identical inputs.
using ActivationOverride = std::function<bool(const std::string & name, int layer, std::int64_t pos0,
                                              std::int64_t n_tokens, std::int64_t width, float * data)>;

struct ReferenceOptions {
    int n_threads = 0;            // 0: all hardware threads
    std::int64_t max_chunk = 1024;  // tokens per batched pass; a longer input is split (same result)
    // Diagnostic only: round intermediates the way llama.cpp does (8-bit blocks before every quantized
    // matmul, bf16 before the BF16 indexer weights, an f16 KV cache, f16 queries and attention
    // probabilities), to tell its rounding noise apart from real differences. Never for reference
    // results. cuda: the oracle server (GPU q8_1 inputs, routed experts on the CPU); cpu: llama.cpp
    // on the CPU with flash attention off (every matmul input in the CPU vec_dot type).
    enum class Emulate { none, llamacpp_cuda, llamacpp_cpu };
    Emulate emulate = Emulate::none;
};

class ReferenceModel {
public:
    ReferenceModel(const GgufModel & model, ReferenceOptions options = {});
    ~ReferenceModel();
    ReferenceModel(const ReferenceModel &) = delete;
    ReferenceModel & operator=(const ReferenceModel &) = delete;

    // Consumes `tokens` at positions n_past() .. n_past() + size - 1 and returns the logits of the
    // last one ([n_vocab]), or of every new token ([size][n_vocab]) when all_logits is set.
    std::vector<float> forward(const std::vector<std::int32_t> & tokens, bool all_logits = false);
    // The same with rope positions ([3][size] axis-major: temporal, height, width; null: text positions continuing
    // after the largest so far) and per-token input embedding rows (null entries, or a null array: token embeddings),
    // as Engine::forward(tokens, all_logits, ForwardInputs) takes them for images.
    std::vector<float> forward(const std::vector<std::int32_t> & tokens, bool all_logits, const std::int32_t * positions,
                               const float * const * embeddings);

    // Forgets every token (fresh sequence).
    void reset();

    std::int64_t n_past() const;
    const ReferenceConfig & config() const;

    // Called with named intermediates while forward() runs; pass nullptr to disable.
    void set_activation_hook(ActivationHook hook);
    void set_activation_override(ActivationOverride override_fn);

    // Checks the batched matmul kernel against a naive double-precision loop on real weights.
    // Returns the largest relative error seen.
    double self_test_matmul();

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace ninfer::flashnext
