# Qwen3.8-Flash-Next in NInfer Extreme

Status: the engine runs end to end (decode and prompts up to 2,051 tokens), 2026-10-04.

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

## 3. The engine as built

Code: `src/flashnext/` (`engine.{h,cpp}`, `cpu_experts.*`, `reference.*`, `gguf.*`, `quants.*`) and
`src/flashnext/cuda/` (`gemv`, `gemm`, `ops`, `experts`). Tools: `tools/flashnext/` (`fn_generate`,
`ref_generate`, and a test per kernel family). Build: `build-gpu.ps1` (nvcc 13.3, sm_89) and
`build-cpu.cmd`.

### 3.1 Placement

- **VRAM:** every dense weight in its GGUF format (Q8_0 split losslessly into an int8 code plane and
  an fp16 scale plane for aligned loads; F32, BF16 and the Q6_K head as shipped), the attention KV
  cache (fp16) and indexer keys, DeltaNet and conv states, activations, and an **expert cache** in
  all remaining VRAM but a reserve for the desktop (default 1.5 GiB): about 15 GiB, 6,700 experts
  (27%), in the GGUF's own block formats so that it holds as many experts as possible.
- **RAM:** every routed expert in `CpuExperts`' lossless repack (Q4L, Q4X, IQ4L, Q8_0), and the PLE
  table, read in place from the memory-mapped GGUF.

### 3.2 Numerics

FP32 activations everywhere on the GPU; weights are dequantized exactly. The only rounding beyond
FP32 is the CPU experts' 8-bit activations (as in llama.cpp): about 1.1 to 1.4% relative error per
expert output. Checked against the FP32 reference (`reference.cpp`, itself checked against llama.cpp
component by component): every GPU intermediate agrees to about 1e-7 until the first CPU expert
contributes. Greedy continuations match the llama.cpp oracle on all test prompts, except at
near-ties where they follow the FP32 reference.

### 3.3 Decode step (1 to 4 tokens)

One CUDA graph per token count replays the whole step: positions live in device memory, and the
CPU's share of the experts goes through mapped host memory. Per layer the GPU computes the mixer,
attention or DeltaNet, and the router; it writes the selections and the FFN input to the layer's
`ExpertLink` and raises a flag; the host thread computes the experts that are not cached and raises
another flag; meanwhile the GPU computes the cached experts and the shared expert, then a one-block
kernel waits for the CPU's answer. GEMVs that read the same input are fused (`gemv_multi`).

### 3.4 Expert cache

Filled at load from routing counts (saved between runs), ranked by count per byte. After every
prompt, and every 256 decoded tokens, it is re-ranked from the routing of recent tokens (half-life
2,048 tokens) plus the long-run counts, and only clearly better experts are swapped in. A prompt's
routing predicts its continuation well: on a 2,000-token code prompt, decode hits rose from 19%
(cache calibrated on other text) to 50%.

### 3.5 Prompts

Chunks of up to 512 tokens: dense layers as exact FP32 GEMMs (each matrix dequantized into a scratch
buffer, cuBLAS SGEMM without TF32), the DeltaNet recurrence over the whole chunk, and the experts
split between the GPU cache and the CPU. Batched expert kernels (GPU and CPU) and batched attention
are in progress; until then prompts run at about 75 tok/s.

## 4. Measurements (RTX 4090 + Ryzen 9 7950X, decode, short context)

| Configuration | Decode |
|---|---:|
| llama.cpp fork, CPU experts, no MTP | 18.6 tok/s |
| llama.cpp fork, 68-slot GPU expert cache, MTP 2 | 43.7 tok/s |
| This engine, every expert on the CPU | 19.0 tok/s |
| This engine, 15 GiB expert cache (69% hits), kernels launched one by one | 40.1 tok/s |
| This engine, 15 GiB expert cache (69% hits), one CUDA graph per step | **56.0 tok/s** |

Profile of one decode token at about 66% cache hits (Nsight Systems): about 9.3 ms waiting for the
CPU's experts and about 10 ms of GPU kernels (5.1 ms dense GEMVs at about 850 GB/s, 1.9 ms cached
experts, 0.6 ms output head, about 2 ms of small kernels).

## 5. Next

1. Long context: QSA selection and gathered attention on the GPU (beyond 2,051 tokens).
2. Prompt speed: batched GPU expert GEMMs over the cache, a batched CPU expert mode, batched
   attention. Target: thousands of tokens per second.
3. Split each layer's cache misses between the CPU (RAM, about 52 GB/s) and the GPU reading the same
   bytes over PCIe (about 22 GB/s): about a third less waiting.
4. 16-bit activations for the CPU experts (quality), MTP (10 to 20% at today's miss cost).
5. Serving through NInfer's OpenAI/Anthropic server and Qwen frontend; Pi profile; evaluation;
   higher-precision expert quantizations within the RAM budget.
