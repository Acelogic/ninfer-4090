# Qwen3.8-Flash-Next in NInfer Extreme

Status: design and first measurements, 2026-10-03.

This document plans a Flash-Next runtime tailored to one machine: an RTX 4090 that also drives the
desktop (about 22.5 GiB usable), a Ryzen 9 7950X (16 cores, AVX-512 with VNNI), and 192 GiB of
DDR5-5200 in four DIMMs. The goal is the best quality and speed for this model on that machine.

## 1. The model

`general.architecture = qwen4exp`. Values are from the Unsloth UD-IQ4_XS GGUF; the GGUF and the
`qwen4exp` graph in the GenerelSchwerz llama.cpp fork are the specification. llama.cpp serves only as
a test oracle.

| Field | Value |
|---|---|
| Layers | 48, plus one MTP block in a separate GGUF |
| Hidden size | 2560 |
| Hyper-connections | 4 streams (residual is `[2560, 4]` per token), low rank 320 |
| Token mixers | 36 Gated DeltaNet layers; every fourth layer (3, 7, ..., 47) is full attention |
| Gated DeltaNet | 16 Q/K heads and 48 V heads of 128; conv kernel 4; output gate is **sigmoid** (Qwen3.5 uses silu) |
| Full attention | 24 query heads, 2 KV heads, head dim 256; query carries a sigmoid output gate; Q/K RMSNorm; interleaved multi-section RoPE over 64 dims, sections `[11, 11, 10, 0]`, base 1e7 |
| QSA | indexer of 4 heads x 128 on mean-pooled 4-token blocks; ReLU per head, summed; each query attends to the top 2048 (+3 tail) tokens. Contexts up to 2051 tokens attend densely. |
| MoE | 512 routed experts, 10 per token, width 640, softmax gating with normalized top-10 weights; one shared expert (width 640) with a sigmoid gate |
| PLE | at layer 1: 16 hashed n-gram rows (2- and 3-grams, 8 heads each) of 160 from a 320M-row table, then key/value projections, a gated value, and a depthwise causal conv (kernel 4, dilation 3) |
| Vocabulary | 248,320 |
| Context | 262,144 |

Bytes in the IQ4_XS file:

| Part | Size | Read per decoded token |
|---|---:|---:|
| Routed experts: gate/up IQ3_S, down IQ4_NL | 55.4 GiB | 10 experts x 48 layers = 1.04 GiB |
| PLE table, IQ4_NL | 26.8 GiB | 16 rows of 160 = about 1.4 KiB |
| Dense: attention, DeltaNet, hyper-connections, shared experts, router (Q8_0, F32) | about 4.2 GiB | all of it |
| Output head, Q6_K; token embedding, Q8_0 | 0.49 + 0.63 GiB | head: all; embedding: one row |

## 2. Where the time goes

Decode is memory traffic. Per token the dense part reads about 4.2 GiB (about 4.5 ms from VRAM at
about 900 GB/s). The experts read 1.04 GiB, which cannot all be in VRAM.

Measured on this machine (`bench/flashnext/cpu_moe_bench`):

| Measurement | Result |
|---|---:|
| RAM read bandwidth, 1 to 16 threads | 55 to 62 GB/s |
| CPU expert decode with ggml-cpu's kernels and our thread pool | 36.8 GB/s, 31.5 tok/s for the experts alone |
| Same with our Q4L gate/up kernel (below) | 49.9 GB/s, 37.9 tok/s for the experts alone |
| `CpuExperts`, all our kernels, 8 representative layers, 1 token | 52.6 GB/s, **40.0 tok/s** for the experts alone |
| `CpuExperts`, 3 tokens sharing their experts (MTP verification) | **99 tok/s** equivalent |
| Reference: the llama.cpp fork end to end, CPU experts, no MTP | 18.6 tok/s |
| Reference: the llama.cpp fork end to end, 68-slot GPU expert cache, MTP 2 | 43.7 tok/s decode, 240 to 280 tok/s prefill |

## 3. Design

### 3.1 Placement

- **VRAM:** dense weights (converted losslessly: Q8_0 maps exactly to NInfer `q8_g32_fp16`); output
  head; KV for the 12 attention layers plus indexer keys; DeltaNet and conv states; activations; and
  an **expert cache** that takes the rest (about 10 to 11 GiB, about 4,500 experts, 18% of all).
- **RAM:** all routed experts (gate/up repacked to Q4L, down IQ4_NL as shipped), the PLE table.

### 3.2 CPU expert kernels (done: measured)

IQ3_S stores each weight as a lattice index; decoding it was the bottleneck (34 GB/s). Every IQ3_S
weight is +/-{1,3,...,15} times a 4-bit block scale times a per-256 fp16 scale, so it repacks
**losslessly** into **Q4L**: a 4-bit code per weight decoded by one byte shuffle per 64 weights
(`_mm512_shuffle_epi8`), multiplied with `_mm512_maddubs_epi16`, scaled and accumulated with
`_mm512_dpwssd_epi32` (VNNI). Q4L costs 22% more bytes than IQ3_S but runs at RAM speed (51.6 GB/s
measured, against 34.4). Accuracy against a float reference matches ggml's IQ3_S kernel exactly.

### 3.3 Decode step

Per layer, on one CUDA stream driven by a host thread:

1. GPU: hyper-connection mix, token mixer (DeltaNet or attention), combine, FFN mix, router logits,
   softmax top-10. Copy the 10 expert ids and weights plus the 2560-wide FFN input to pinned memory.
2. GPU, concurrently: shared expert and every selected expert resident in the VRAM cache.
3. CPU, concurrently: the selected experts not in the cache, with the Q4L/IQ4_NL kernels on 16
   threads; write the weighted sum to pinned memory.
4. GPU: add the CPU part, combine.

Steps 1 to 4 per layer are captured as CUDA graphs so that one token costs about 49 graph launches.
Expected cost per token: GPU dense 4.5 ms + CPU misses (26 ms x miss rate) + about 1.5 ms of
GPU-CPU hand-offs. A 50 to 70% cache hit rate gives about 50 to 70 tok/s before MTP.

**Expert cache policy:** frequency-weighted with decay, filled from routing statistics; refreshed
between requests and slowly during decode, never on the critical path.

**MTP:** verifying k drafts processes k+1 tokens per layer; the CPU computes each needed expert once
for all tokens that chose it, so tokens that share experts share the RAM traffic.

### 3.4 Prefill

A chunk of C tokens selects nearly every expert in every layer. Instead of computing experts on the
CPU, stream each layer's experts to the GPU (1.11 GiB in the original IQ3_S/IQ4_NL form; about 44 ms
over PCIe 4.0 x16) into a double buffer, and run grouped GEMMs on the GPU. With C = 4096 to 8192 this
gives an estimated 2,000 to 4,000 tok/s, against 240 to 280 tok/s today.

### 3.5 Long context

llama.cpp evaluates QSA as dense attention with a mask (its source has "TODO: enable sparse
attention"), so its decode falls from 44 to 28 tok/s at 200K. A true sparse kernel gathers only the
selected 2,051 KV rows, so decode cost stays flat with context length.

### 3.6 Reuse from NInfer

`gated_delta_net` (state 128, Hqk 16, Hv 48, L2-normalised q/k: an exact fit), `causal_conv1d_silu`,
`gdn_gating`, `rmsnorm`, `rope`, `softmax_attention`, `sigmoid_mul`, `embedding`, `argmax`,
`sampling`, `linear` (`q8_g32_fp16`), and the Qwen text frontend. New: hyper-connection kernels, the
sigmoid-gated DeltaNet norm, QSA, PLE, the MoE router with CPU split, GPU kernels for Q4L/IQ3_S/IQ4_NL
experts, the Q6_K head, the host expert engine, and the decode/prefill orchestration.

## 4. Milestones

1. **CPU expert kernels.** Done (`src/flashnext/cpu_experts.*`, `tools/flashnext/test_cpu_experts`):
   - Formats: Q4L (from IQ3_S), Q4X (from IQ4_XS) and IQ4L (from IQ4_NL) are lossless relayouts; Q8_0
     is used as is.
   - Tokens that share an expert share one pass over its weights.
   - Accuracy: 1.1 to 1.4% relative error of the FFN output against exact math (8-bit activations,
     as in llama.cpp).
   - Speed: 40.0 tok/s for the experts alone, up from 31.5 with ggml's kernels.
   - Possible later: a higher-precision activation mode (int16 activations) at some speed cost.
2. **Reference forward pass.** Plain C++ FP32 implementation of one forward step from the GGUF,
   checked against the llama.cpp oracle (same greedy tokens, close logits). It becomes the oracle for
   every GPU kernel.
3. **GPU engine, correct.** Dense path on the GPU through NInfer ops, experts on the CPU; layer by
   layer agreement with the reference; greedy CLI.
4. **Decode speed.** Expert cache with measured routing statistics, overlap, CUDA graphs, MTP.
5. **Prefill and long context.** Streamed expert GEMMs, sparse QSA.
6. **Serving and quality.** HTTP server with the Qwen frontend and tools; weight quality choice
   (for example higher-precision experts within the RAM budget); Pi profile; evaluation.
