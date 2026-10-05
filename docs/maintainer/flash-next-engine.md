# Qwen3.8-Flash-Next in NInfer Extreme

Status: the engine runs end to end at full context with speculative decoding and KV streaming, 2026-10-05.

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

Code: `src/flashnext/` (`engine.{h,cpp}`, `kv_cache.*`, `cpu_experts.*`, `reference.*`, `gguf.*`, `quants.*`) and
`src/flashnext/cuda/` (`gemv`, `gemm`, `ops`, `experts`, `qsa`, `kv_stream`). Tools: `tools/flashnext/` (`fn_generate`,
`ref_generate`, and a test per kernel family). Build: `build-gpu.ps1` (nvcc 13.3, sm_89) and
`build-cpu.cmd`.

### 3.1 Placement

- **VRAM:** every dense weight in its GGUF format (Q8_0 split losslessly into an int8 code plane and
  an fp16 scale plane for aligned loads; F32, BF16 and the Q6_K head as shipped), the attention KV
  cache (fp16; above a 32K window only a page cache of it, section 3.9) and indexer keys, DeltaNet and
  conv states, activations, and an **expert cache** in all remaining VRAM but a reserve for the
  desktop (default 1.5 GiB): about 15 GiB, 6,700 experts (27%), in the GGUF's own block formats so
  that it holds as many experts as possible.
- **RAM:** every routed expert in `CpuExperts`' lossless repack (Q4L, Q4X, IQ4L, Q8_0), the PLE
  table, read in place from the memory-mapped GGUF, and above a 32K window the attention K/V
  (pinned, 24 KiB per token of the window).

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

Chunks of 512 to 2,048 tokens (`prefill_chunk`):
- dense layers as exact FP32 GEMMs (each matrix dequantized into a scratch buffer, cuBLAS SGEMM
  without TF32);
- the DeltaNet recurrence over the whole chunk;
- cached experts as tensor-core GEMMs grouped by expert (`experts_gpu_batch`: exact weights,
  activations as two fp16 terms, 4e-7 relative error, bitwise reproducible);
- the other experts on the CPU (`CpuExperts::run_batch`: each expert read from RAM once per chunk,
  register-tiled VNNI micro-kernels), concurrently;
- the expert cache re-ranked between chunks, since a chunk's routing predicts the rest of the prompt.

### 3.6 Long context: QSA

Every attention layer keeps pooled indexer keys for blocks of 4 tokens; each query selects its 2,051
cells (the best blocks by the indexer score, ties to the lower block, plus its own incomplete block)
and attention is gathered over them (`cuda/qsa.cu`). Given the same inputs, selections are identical to
the FP32 reference. Decode costs 0.05 to 0.1 ms per attention layer even at 262K context. Raw indexer
keys live in a ring of a step's worth (a block needs them only until it is complete).

### 3.7 Speculative decoding (MTP)

The MTP head (shared-Q8_0 GGUF) is one more layer fed with the main model's last hidden streams and the
next token's embedding. `draft(next, k)` proposes tokens, `forward({next, drafts...}, true)` verifies
them in one step, and `rollback(n)` keeps the accepted prefix: DeltaNet conv and recurrent states are
kept after each token of a 2 to 4 token step and the PLE history is rebuilt, so rejected drafts are
undone exactly. Before every step the MTP layer catches up on the main model's tokens with their true
hidden states, so drafts always attend to exact entries. Its 512 experts stay in VRAM (2.5 GB). The
layer attends to its last 2,051 positions only, so its K/V is a ring of `cap + 2,051` rows (rounded up
to 256; 5.5 MiB with 512-token passes) instead of the whole window: a pass of up to `cap` tokens
never overwrites a row it still reads, and attention reads the same values in the same order.

### 3.8 Serving state

`snapshot()` / `restore()` save and return to the recurrent state after a token sequence (116 MiB plus
the MTP layer's K/V ring, about 20 ms); attention keys stay in the caches by position (with KV
streaming, in the host copy), and restore checks they still hold the snapshot's tokens. A server
reuses a conversation's prefix this way. The MTP ring is overwritten by later positions, so it travels
with the snapshot (with the MTP position): a restore puts back exactly the rows the next drafts read.

### 3.9 Long context: KV streaming

At 24 KiB per token (fp16 K and V of 2 heads of 256, 12 attention layers) a 262K window's attention
cache would take 6 GiB of VRAM, all of it out of the expert cache. Each query reads only its 2,051
selected cells, so above a 32K window the engine keeps the K/V in RAM and a working set in VRAM
(`kv_cache.{h,cpp}`, `cuda/kv_stream.{h,cu}`; the resolve and copy kernels are adapted from Strata,
MIT):

- **Host copy.** Per attention layer, the authoritative K/V of every position in pinned,
  device-mapped RAM, laid out as a resident cache ([max_ctx][2][256] fp16, 1 KiB per token and
  tensor).
- **Page cache.** VRAM holds `kv_resident` cells per layer (default 32,768: 768 MiB for the 12 layers)
  as pages of 4 cells (one QSA block, 8 KiB of K and V), with a page table block -> slot.
- **Writers.** `attn_prep` stores every row in the host copy and, when its block is resident, in its
  page, so pages never go stale: nothing is invalidated after a rejected draft, a rollback or a
  restore.
- **Decode steps** (inside the CUDA graph): after `qsa_select`, `kv_resolve` (one block of 1,024
  threads) marks the selected blocks used, claims the missing ones, picks victims with a CLOCK sweep
  that never evicts a block this call uses, repoints the page table and rewrites the selected cells
  into page rows; `kv_copy_misses` reads the missed pages from the host copy over PCIe; `attn_sparse`
  reads the page pool exactly as it read the resident cache.
- **Prompt chunks** that end within the page cache make every block of `[0, pos0 + T)` resident (it
  fits). Longer chunks beyond it are **staged**: the layer's rows `[0, pos0)` are DMA'd from the host
  copy into a full-context pool on a side stream, overlapped with the layers before it (one pool for
  the 12 layers, refilled per layer), while `attn_prep` writes the chunk's rows into the pool, the host
  copy and any resident page. Short chunks beyond it (up to `max(kv_group_tokens, pos0 / 512)` tokens:
  64 at 32K depth, 488 at 250K) instead attend in groups of 15 queries, each resolved like a decode
  step: a staged chunk at 250K depth moves 6 GB over PCIe whatever its length (about 0.2 s that a short
  chunk cannot hide), while groups cost about 0.25 ms per token more than one attention call.
- **Exactness.** Attention reads the same fp16 values in the same order (groups use the whole chunk's
  split count), so streamed and resident engines agree bit for bit: `test_qsa`'s parity test, and
  identical logits hashes on full runs (section 4.1).
- **Switches.** `EngineOptions::kv_stream` (-1, the default: on when the window exceeds
  `kv_resident`; 0 off; 1 on), `kv_resident`, `kv_group_tokens`; `fn_generate --kv-stream 0|1
  --kv-resident N --kv-group-tokens N`, which also prints the page counters. The staging pool is the
  only VRAM cost that grows with the window (2 KiB per token, needed during prompts only):
  `kv_stage_bytes()` / `kv_stage_borrow()` let the prompt planner lend it from the expert cache;
  without that the engine keeps a pool of its own (`kv_stage_cells`, default the window).

## 4. Measurements (RTX 4090 + Ryzen 9 7950X)

| Configuration | Result |
|---|---:|
| llama.cpp fork: decode with a 68-slot GPU expert cache and MTP 2 | 43.7 tok/s |
| llama.cpp fork: prompt processing | 240 to 280 tok/s |
| This engine: decode, every expert on the CPU | 19.0 tok/s |
| This engine: decode, 15 GiB expert cache, one CUDA graph per step (code prompt, 66 to 69% hits) | 54 to 56 tok/s |
| This engine: decode after a 2,600-token prompt (cache re-ranked, 85 to 92% hits) | 63 to 69 tok/s |
| This engine: greedy decode with MTP, 2 to 3 drafts (chat prompt, 72 to 84% accepted) | **78 tok/s** |
| This engine: prompt processing, 2,600 tokens, 512- and 1,024-token chunks | 427 and 512 tok/s |
| This engine: prompt processing, 7,800 tokens, 2,048-token chunks | **757 tok/s** |

Profile of one decode token at about 66% cache hits (Nsight Systems): about 9.3 ms waiting for the
CPU's experts and about 10 ms of GPU kernels (5.1 ms dense GEMVs at about 850 GB/s, 1.9 ms cached
experts, 0.6 ms output head, about 2 ms of small kernels).

Measured and rejected: letting the GPU read a share of each step's cache misses straight from host
memory. Those reads draw on the same DRAM bandwidth as the CPU experts; a 30% share made decode 2.7x
slower.

### 4.1 Context window

Without KV streaming the window is reserved in VRAM when the engine loads, about 28.7 KB per token
(fp16 KV of the 12 attention layers, QSA block keys and the MTP layer's KV), and that memory comes out
of the expert cache. With KV streaming (section 3.9, on above a 32K window) a 262K window costs the
0.75 GiB page cache, the indexer keys (384 MiB), the staging pool while it is not lent (512 MiB) and a
5.5 MiB MTP ring. Short prompts, idle machine, `fn_generate -n 256` (chat: oracle p3 with `--mtp
--draft 2`; code: p2 without MTP), mean of 3 runs each (2026-10-05, the same routing statistics):

| Window | KV | Expert cache (MTP / no MTP) | Chat, MTP 2 | Code, no MTP |
|---|---|---:|---:|---:|
| 64K | in VRAM | 10.62 / 13.81 GiB | 84.4 tok/s | 49.6 tok/s |
| 64K | streamed | 11.36 / 14.43 GiB | 86.8 tok/s | 50.7 tok/s |
| 262K | in VRAM | 5.37 / 8.93 GiB | 65.0 tok/s | 43.0 tok/s |
| 262K | streamed | 10.61 / 13.68 GiB | 84.5 tok/s | 49.7 tok/s |
| 262K | streamed, staging pool lent | 11.11 / 14.18 GiB | 85.5 tok/s | 50.4 tok/s |

(The MTP ring alone, `--kv-stream 0`, adds 0.49 GiB at 262K.) A 262K window now decodes short
prompts as fast as a 64K window did.

At depth, `fn_generate` with a 250,139-token prompt of this repository's source with three facts
planted at 10%, 50% and 90% depth, 2,048-token chunks, MTP 2, 262K window: with the K/V in VRAM the
expert cache is 2.76 GiB, prefill 521 tok/s, decode 33.6 tok/s; streamed: 8.81 GiB, 677 tok/s,
41.1 tok/s; all three facts retrieved both ways. The same prompt with the cache pinned to 4 GiB in
both configurations (`--cache-mib 4096`) gives bitwise identical prompt, follow-up and decode logits
resident and streamed, and the same prefill speed (579 and 593 tok/s): the staging DMA (346 GiB over
the prompt) is hidden behind the layers between attention layers.

Short prompts at depth (`--followup`, after that 250K prompt, cache pinned to 4 GiB, seconds):

| Tokens | K/V in VRAM | Streamed, in groups | Streamed, staged |
|---:|---:|---:|---:|
| 8 | 0.183 | 0.160 | 0.399 |
| 32 | 0.350 | 0.293 | 0.551 |
| 128 | 0.730 | 0.773 | 0.901 |
| 256 | 1.102 | 1.171 | 1.226 |
| 512 | 1.664 | 1.761 | 1.641 |
| 1024 | 2.611 | 2.668 | 2.473 |

Hence groups up to depth / 512 tokens. KV streaming is checked by `test_qsa --kv-only` (two layers
resident and streamed through every step kind, bit-identical outputs, and the MTP ring against a full
cache) and by `fn_generate --hash` with a pinned cache, resident against streamed: the oracle prompts
with MTP drafting (identical logits and drafts, also against the engine before these changes), a
64K-token prompt with MTP and follow-ups of 8 (grouped), 100 (grouped) and 200 (staged) tokens, the 250K
prompt with follow-ups of 8 to 1,024 tokens, and `--test-snapshot --snapshot-detour 5000` (the MTP ring
wraps between the snapshot and the restore).

Earlier measurement, through the server before KV streaming (256,125-token prompt, three facts at 10%,
50% and 90%): prefill 412 tok/s (512-token chunks; 10 min 21 s cold); all three facts retrieved;
decode 33 to 36 tok/s at that depth; a follow-up turn reused all 256K tokens and started in 1.0 s.

## 5. Next

1. Lend the KV staging pool from the expert cache during prompts (the prompt planner), so that a
   262K window costs no expert-cache VRAM beyond the 0.75 GiB page cache and the indexer keys. Then
   measure deep prompt chunks with `--kv-stream 0` and `1`: staging (6 GB of DMA per chunk at 250K
   depth) competes with streamed experts for PCIe; if it shows, stage only the blocks the chunk's
   selections name.
2. Find why a short request through the server decodes slower than `fn_generate` with the same
   window (46 against 77 tok/s; different prompts, so measure like for like first).
3. Higher-precision expert quantizations within the RAM budget, and an evaluation through Pi.
