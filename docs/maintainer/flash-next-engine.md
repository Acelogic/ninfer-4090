# Qwen3.8-Flash-Next in NInfer Extreme

Status: the engine runs end to end at full context with speculative decoding, KV streaming, image
input and an expert cache that follows decoding, 2026-10-05.

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

A step of T tokens reads each dense weight once (the GEMVs take up to 4 tokens) and each expert the
CPU computes once (`CpuExperts::run` groups a step's pairs by expert); the GPU's cached experts run
per (token, expert) pair. So a step's cost is the GPU's dense part (8 ms for one token, 10 ms for
four, at 64K) plus, per layer, the CPU's distinct experts (57-66 us each, 40-46 GB/s), which run beside
the GPU's cached experts: in an MTP verification step of four tokens every layer waits for the CPU
(none of 408 layers in an Nsight trace waited under 20 us), so the CPU's distinct experts per step
decide its cost (section 4.4). While the cache swaps experts in the background (section 3.4) a step's
and a draft's own uploads and downloads (the step record, embeddings, PLE rows, logits, the MTP
pass's inputs, the expert maps, state copies for rollback) go through the SMs (`copy_sm`) rather than a
copy engine: behind a batch of swap DMAs they waited up to 12 ms.

### 3.4 Expert cache

Filled at load from routing counts (saved between runs), ranked by count per byte; the per-layer
sizes stay as chosen at load. A prompt that borrows VRAM (section 3.5) refills the borrowed slots,
when it ends, with the uncached experts its routing ranks highest (the routing of recent tokens,
half-life 2,048 tokens, plus half the long-run counts).

**Decode-time adaptation** (`EngineOptions::decode_adapt`, on; `engine.cpp` "Decode-time
adaptation"). The routing of the last steps predicts the next steps' far better than a prompt does:
a conversation drifts. Recorded on the 32K-token code prompt below, the best static cache chosen from
the prompt's routing would have kept 64% of the 256 decoded tokens' pairs in VRAM (the engine kept
63%), the best static cache chosen from the decode routing itself 81%, and a cache that follows the
last steps 75% (the MTP run's continuation drifted further: 37%, 76% and 64-67%). After every decode
step:

- each step's pairs count as uses of their experts (`use`, decayed with a half-life of 32 steps);
- per layer, the uncached experts with a use replace the cached ones that score at least 1.5 uses
  lower, at most 192 per step; the score is the decayed uses plus a weak prior, 5 tokens' worth of the
  decayed routing above (mostly the prompt's), which ranks the experts without recent uses;
- the swaps run on their own stream beside the next step. The evicted expert leaves the maps before
  that step (the CPU computes it meanwhile); its slot is refilled from the CPU's pinned copy (DMA, then
  the GPU conversion of section 3.5, byte-identical to the original); the new expert enters the maps at
  the first step after its copy landed. One batch is in flight at a time. The batches (8 experts: their
  copies and one conversion) are queued while the CPU spin-waits for the GPU's layers
  (`host_experts`): queued between steps they delayed the following draft passes by as long as their
  DMAs took. Every layer's map is one device array, uploaded once per step before its kernels.

With adaptation the whole cache is no longer re-ranked after prompts: after a 32K-token prompt that
took 0.4 s (some 3,300 swaps) and gained less over 256 decoded tokens than the adaptation does in a
few steps (decode 54.0 against 49.7 tok/s; with MTP 56.8 against 57.0). `NINFER_FN_RERANK=1` restores
it; without adaptation (`decode_adapt = false`, or no pinned CPU copy) the cache is re-ranked after
every prompt and every 256 decoded tokens as before. Only placement changes: a pair runs with the
GPU's FP32 or the CPU's 16-bit arithmetic, as with any cache change, so decode logits differ from a
non-adapting run in the last bits while prompt logits stay bitwise the same (section 4.4).

The parameters (`AdaptParams`; `NINFER_FN_ADAPT="hl,beta,margin,min_use,every,max_swaps"` overrides
them) come from replaying recorded routing through the policy offline: `NINFER_FN_ROUTING_DUMP=<file>`
records every routed pair and the cache's contents (format in `engine.cpp`), and a replay of the
32K-token and short prompts' decodes, plain and with MTP, over half-lives of 4-64 steps, priors of
0-50 tokens, margins of 0.5-2 and 16-256 swaps per step ranked the settings by the CPU's distinct
experts per step plus the swaps' DMA (about 30 us of DRAM time each, measured). The landscape is flat
near the optimum (half-lives 16-64 within 1%); margin 1.5 against 1.0 was then measured on the engine
(fewer swaps, 32K MTP +11% partly through a different continuation, 32K plain -1.5%).

Before this, `fn_generate` printed "0 cached experts swapped" after a long prompt because the refill
and re-ranking ran at the first decode step, after it read the counters (they ran: 3,300 swaps and a
0.2 s refill); the prompt's routing reached the statistics (streamed chunks record every pair). But
during decoding the cache barely moved: the recent routing was dominated by the prompt's last 8K
tokens, so 256 decoded tokens shifted it by a few percent and the re-rankings swapped 13-17 experts.

### 3.5 Prompts

A prompt of more than 512 tokens runs in big chunks with every routed expert on the GPU (code:
`engine.cpp` "Prompt processing", `prefill.{h,cpp}`, `cuda/expert_stream.cu`, the phased API of
`cuda/experts_batch.cu`). Per chunk:

- **Buffers borrowed from the expert cache.** The cache is one VRAM arena (each layer's pool a part of
  it). A prompt lends the arena's tail to the chunk's activations, the experts' workspace, the
  streaming ring and the conversion slots, and the experts that lived there leave the cache. The
  activation buffers overlay each other by stage (the hyper-connection mixer, PLE, DeltaNet,
  attention and FFN never hold live data at the same time; about 235 KiB per token, the experts'
  workspace included; 3.5 GiB for 8,192 tokens with the 1.5 GiB ring and the conversion slots, 4 GiB
  at 250K depth with the KV staging pool). The buffers stay lent across consecutive prompt
  calls (the server feeds long prompts in pieces of 8,192) and come back at the next decode step or
  draft (`end_prompt`): the lent slots are refilled with the uncached experts that rank highest for
  the cache, copied from pinned RAM and converted on the GPU (about 0.1 ms per expert; 0.19 s after a
  32K-token prompt); without decode adaptation the cache is then re-ranked as after any prompt.
- **Chunk size.** `prefill_chunk = 0` (the default) takes per prompt the largest chunk on a 256-token
  grid, up to 8,192 (`prefill_chunk_max`), whose buffers, a ring of up to 1.5 GiB (at least 256 MiB)
  and, when KV streaming stages the prompt's chunks (section 3.9), the staging pool (2 KiB per
  context token, 512 MiB at 262K; `kv_stage_borrow`) fit in 90% of the cache; the last chunk takes
  the rest. A prompt that stages is lent even when it fits the permanent buffers. A ring of about a layer's
  streamed experts lets the copy engine run on through a layer's dense part (24K at 262K: a 640 MiB
  ring 2,593 tok/s, 1.5 GiB 2,797).
- **Every expert on the GPU.** At load, every layer's CPU copy (`CpuExperts`, now one contiguous,
  page-aligned blob per expert: gate, up, down) is registered with the driver (`cudaHostRegister`,
  per layer; about 0.7 s for 48 layers). A chunk's plan gives each layer's cached experts keys
  0.. (their pool slots) and the others the following keys in expert-id order, in groups of 32 (64 for
  chunks under 4,096 tokens). A copy stream fills the ring with the groups' blobs in that order (one
  DMA per run of consecutive ids; 26.7 GB/s measured), as far ahead as the ring allows, with no host
  waits: the order is known before routing. A conversion stream turns each group's blobs into the
  cache's slot layout (the GPU version of `export_expert` + `pack_expert_rows`, byte-identical; about
  6-8 us per expert) in one of two sets of conversion slots, which frees its ring bytes; so a group
  converts while the previous one computes. Per layer the compute stream groups the pairs by key
  (`experts_phased_begin`), computes the cached experts in one launch, then per group waits for its
  conversion and computes it (`experts_phased_run`); one reduction per layer adds every token's pairs
  in k order. A few event operations per group of 32 experts (WDDM charges about 10 us each). Since
  each pair's result does not depend on its key, launch or where its weights came from, a streamed
  chunk gives bitwise the same output whatever the cache holds or lends.
- **CPU share of small chunks** (optional, `prefill_cpu_share_max`, off by default). In small chunks PCIe is
  the bottleneck, so the experts predicted (from the decayed recent routing) to get the fewest tokens can go
  to the CPU instead: as many as the CPU can finish (60 us per expert plus 10 us per pair) by the time the
  copy engine has delivered the layer's other experts, counting from the end of the layer's dense
  part. A worker thread runs them (`CpuExperts::run_batch`) as soon as the layer's routing reaches the
  host; the stream waits for it in a host function and reads its sums straight from pinned memory (a
  copy would queue behind the ring's). Those pairs carry the CPU's
  numerics (16-bit activations) and the split depends on history, so such chunks are reproducible only
  for the same history. Measured, it helped only 2K chunks at a 262K window (+14%) and slowed 3-4K chunks
  and the 64K window by 6-11%, so it is off.
- **Smaller chunks** (under 1,024 tokens, `prefill_stream_min`) and layers whose RAM could not be
  pinned keep the hybrid path: cached experts on the GPU (`experts_gpu_batch`), the others on the CPU.
- **Dense layers** (Q8_0) on the tensor cores (`cuda/gemm_tc.cu`, `prefill_dense_tc`): the int8 codes
  enter exactly as fp16, each 32-block's scale multiplies the block's partial sum in FP32, and each
  activation enters as two fp16 terms (hi, lo) after a per-row power-of-two scale, with FP32
  accumulation of the hi products; the experts' Fp16x2 arithmetic. On the model's matrices it is 1.6x
  faster than dequantizing for cuBLAS SGEMM and closer to double precision (1-4e-7 relative against
  SGEMM's 3e-7 to 1.3e-6, `test_gemm_tc`); SGEMM stays for long narrow products with few tiles and for
  F32/BF16 weights. DeltaNet's conv runs in parallel tiles of tokens and its delta rule spreads each v
  head's 128 state rows over 16 blocks (a warp per 2 rows, the next token's inputs prefetched; the
  RMSNorm runs afterwards), bitwise the same as the single-block kernels that decode steps use
  (`NINFER_FN_SERIAL_DN=1` forces those).
- **QSA** stores raw keys and selects cells 256 queries at a time, so the raw-key ring is 264 rows and
  the selection scratch 64 MiB at 262K whatever the chunk.
- **Host work** for chunk c+1 (embeddings, the 16 PLE rows per token) runs on a thread during chunk c.
- **MTP.** The MTP layer's catch-up skips the prompt positions that no draft can attend to (all but
  the last 2,051 + 8), in passes of at most 512 tokens, so its K/V ring of 512 + 2,051 rows (section
  3.9) holds whatever the chunk size.

`prefill_lend = false` restores the old scheme (buffers for `prefill_chunk` tokens allocated for good,
experts split between the GPU cache and the CPU). `fn_generate --profile` (or `NINFER_FN_PROFILE=1`)
prints the time per stage of every prompt from CUDA events; `--prefill-runs`, `--pieces`, `--hash-state`
help compare schedules.

### 3.6 Long context: QSA

Every attention layer keeps pooled indexer keys for blocks of 4 tokens; each query selects its 2,051
cells (the best blocks by the indexer score, ties to the lower block, plus its own incomplete block)
and attention is gathered over them (`cuda/qsa.cu`). Given the same inputs, selections are identical to
the FP32 reference. Decode costs 0.05 to 0.1 ms per attention layer even at 262K context. Raw indexer
keys live in a ring of 264 rows (a block needs them only until it is complete; prompt chunks store and
select 256 tokens at a time).

### 3.7 Speculative decoding (MTP)

The MTP head (shared-Q8_0 GGUF) is one more layer fed with the main model's last hidden streams and the
next token's embedding. `draft(next, k)` proposes tokens, `forward({next, drafts...}, true)` verifies
them in one step, and `rollback(n)` keeps the accepted prefix: DeltaNet conv and recurrent states are
kept after each token of a 2 to 4 token step and the PLE history is rebuilt, so rejected drafts are
undone exactly. Before every step the MTP layer catches up on the main model's tokens with their true
hidden states, so drafts always attend to exact entries. A catch-up pass stops once it has stored its
K/V: in a single layer a position's keys and values depend only on that position's input, and the rest
of the pass (attention, experts) was never read (drafts and verify logits bitwise unchanged; after a
long prompt the catch-up of 2,059 positions no longer computes their experts).

The layer's 512 routed experts (Q8_0, 2.7 GB) are computed by the CPU (`CpuExperts::add_layer`: Q8_0
gate/up rows against the 16-bit activations the main layers' CPU experts use, 5.5e-5 from exact math in
`test_cpu_experts --mtp`), through one more `ExpertLink` serviced while the draft pass's graph runs; the
GPU computes the shared expert meanwhile. Their VRAM goes to the expert cache: 13.99 instead of 11.50
GiB at a 64K window (6,256 experts instead of 5,144). A draft costs about 1 ms more (three drafts: 3.0-3.6
ms of CPU experts per step); the drafts were the same tokens on every test (identical draft hashes on
the oracle prompts, acceptance 94.5% on p2 and 73-75% after the 32K prompt, as before).
`EngineOptions::mtp_experts_vram` (`fn_generate --mtp-experts-vram`) keeps them in VRAM. The
layer attends to its last 2,051 positions only (`NINFER_FN_MTP_WINDOW`, an experiment, widens it; section
4.5), so its K/V is a ring of `cap + 2,051` rows (rounded up
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
  only VRAM cost that grows with the window (2 KiB per token, needed during prompts only): the prompt
  planner lends it from the expert cache with the chunk's other buffers (`kv_stage_bytes()`,
  `kv_stage_borrow()`, section 3.5); only with `prefill_lend = false` does the engine keep a pool of
  its own (`kv_stage_cells`, default the window).

### 3.10 Images

The model reads images through Qwen3-VL's vision tower, shipped for llama.cpp as `mmproj-F16.gguf`
(`general.architecture = clip`, `clip.projector_type = qwen3vl_merger`; the same file serves the base and
the Huihui model, whose abliteration does not touch vision). The reference is llama.cpp's mtmd
(`tools/mtmd/models/qwen3vl.cpp`, `mtmd-helper.cpp`) and its `qwen4exp` graph.

- **Encoder** (`vision.{h,cpp}`, `cuda/vision.{h,cu}`): 27 pre-norm ViT layers of 1152 (16 heads of 72,
  2-D rope over the patch row and column, bidirectional attention, GELU-tanh MLP of 4304), a learned
  48 x 48 position embedding resized bilinearly (aligned corners) to the patch grid, and the merger:
  LayerNorm per patch, 2x2 patches as one 4608-wide row, 4608 -> GELU -> 2560. Patches are 16 x 16 with
  the frame repeated (the two temporal kernels side by side as one 1152 x 1536 matrix). FP32 activations,
  F16 weights converted exactly to FP32 for cuBLAS SGEMM (no TF32), an FP32 tiled attention kernel with
  online softmax. The merger's GELU is exact (erf), as the model defines it (llama.cpp uses the tanh
  approximation; on the test images the choice changes nothing measurable next to llama.cpp's own
  rounding).
- **VRAM.** None between images. The weights (0.9 GB, F16) live in pinned RAM. An encode borrows its
  workspace from the expert cache (`Engine::lend_vram`, the same lending prompts use: about 270 MiB plus
  65 KiB per patch, 356 MiB for a 1024-token image) and streams one layer's weights (30 MB) at a time into
  a double-buffered stage, overlapped with the previous layer. The lent experts come back at the next step
  that needs the cache (or the prompt binds its own buffers over the same region).
- **Preprocessing** is the Qwen3.5 frontend's (decode, bicubic resize to multiples of 32 within the pixel
  budget, 2x2-block patch order). The budget is llama.cpp's for this projector: 8 to 4,096 tokens per
  image (8,192 to 4,194,304 pixels). The frontend hands patches over as bf16 of `u / 127.5 - 1`; each
  value maps back to its 8-bit pixel, normalized in FP32 as llama.cpp does.
- **Positions.** The model's rope is interleaved multi-axis (sections `[11, 11, 10, 0]` over the 32
  rotated pairs: pair i turns with the temporal, height or width position, i % 3). An image's tokens share
  the temporal position p of their first token and count rows and columns from p (`t = p, h = p + row,
  w = p + col`); text after the image continues at `p + max(rows, cols)`, the largest position so far plus
  one (mtmd and Qwen3-VL's `get_rope_index` alike). `Engine::forward(tokens, all_logits, ForwardInputs)`
  takes these per token (`[3][n]`) with the image rows that replace the token embeddings; without them,
  text continues after `next_position()`. The KV cell stays the token's index in the sequence (causal
  attention in sequence order, as llama.cpp's 2-D causal mask orders an image's cells raster-wise), and
  QSA blocks are 4 consecutive cells (llama.cpp's `indexer_kpool_by_order`), each block key turned with
  its first member's positions. Kernels read a per-step position table (`ops.h`, `kRopeAxes`; decode
  steps upload theirs with the step record inside the CUDA graph, prompt chunks with their inputs), rows
  -3 .. T-1 so that blocks completed by the step find their first member. The MTP layer turns its queries
  and keys with the same positions.
- **PLE.** An image position carries the token id `qwen4exp.ple.image_token_id` (248056,
  `<|image_pad|>`), which the n-gram hash reads there and in the n-grams of the tokens after it, in
  sequence order (llama.cpp's behaviour when an image is decoded in one ubatch).
- **MTP.** The catch-up feeds image positions the image token's embedding (the MTP head reads token ids,
  as in llama.cpp and Strata); drafts are verified, so this affects only acceptance.
- **State.** `rope_next` (the next text position) travels with snapshots; `restore()` also checks a digest
  of the cells' positions and embedding-row hashes, so a snapshot cannot resume over a different image with
  the same token ids. `rollback()` restores it from the kept tokens.
- **Server** (`flashnext_core.cpp`, `--flashnext-vision <mmproj.gguf>`): images in OpenAI chat
  (`image_url`), Responses and Anthropic messages, through the shared frontend (media cache, decode,
  `<|vision_start|><|image_pad|>...<|vision_end|>`, positions). Before a prompt runs, the images whose
  positions it still computes are encoded (or taken from a 512 MiB cache of encoded images keyed by content
  digest and grid). Prefix reuse compares, besides token ids, a per-position media key (image digest, grid
  and index), so snapshots and the live sequence are reused across turns that resend the same images. Video
  is rejected. The request log reports `vision_tokens` and the encode time (`vision`).

## 4. Measurements (RTX 4090 + Ryzen 9 7950X)

| Configuration | Result |
|---|---:|
| llama.cpp fork: decode with a 68-slot GPU expert cache and MTP 2 | 43.7 tok/s |
| llama.cpp fork: prompt processing | 240 to 280 tok/s |
| This engine: decode, every expert on the CPU | 19.0 tok/s |
| This engine: decode, 15 GiB expert cache, one CUDA graph per step (code prompt, 66 to 69% hits) | 54 to 56 tok/s |
| This engine: decode after a 2,600-token prompt (cache re-ranked, 85 to 92% hits) | 63 to 69 tok/s |
| This engine: greedy decode with MTP, 2 to 3 drafts (chat prompt, 72 to 84% accepted) | **78 tok/s** |
| This engine: prompt processing, 2,600 tokens (first version: 512- and 1,024-token chunks) | 427 and 512 tok/s |
| This engine: prompt processing, 2,600 tokens, every expert on the GPU (section 3.5) | 1,115 tok/s |
| This engine: prompt processing, 8,192 / 24,576 / 32,768 tokens (section 4.2) | 2,783 / **3,055** / **3,058** tok/s |

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

With the streamed prompt path (section 3.5) the same 250,139-token prompt prefills in 85.6 s (2,922
tok/s, default cache, 31 chunks of 8,192) and the answer still names all three facts. With the cache
pinned to 4 GiB: 2,837 tok/s with the K/V streamed (27 chunks staged, 86 GiB of staging DMA over the
prompt) and 2,905 tok/s with `--kv-stream 0`, so staging costs the prompt 2.3% of its time; the prompt
and decode logits are bitwise the same all three ways, and the prompt logits the same as before KV
streaming was merged.

### 4.2 Prompt processing

`fn_generate -n 2`, one run each, idle machine (2026-10-05). Before: the engine before the streamed
prompt path (cached experts on the GPU, the rest on the CPU, cache re-ranked between chunks). After:
the default (`prefill_chunk = 0`, section 3.5). Tokens per second:

| Prompt | Window | Before, 512-token chunks (the default) | Before, other chunks | After |
|---:|---|---:|---:|---:|
| 600 | 262K, MTP | 199 | | 315 |
| 2,048 | 262K, MTP | 337 | | 847 |
| 3,072 | 262K, MTP | 315 | | 1,213 |
| 6,144 | 262K, MTP | 390 | | 2,366 |
| 8,192 | 262K, MTP | 408 | 496 (2,048) | 2,689 |
| 24,576 | 262K, MTP | 419 | 527 (2,048) | **2,988** |
| 32,768 | 262K, MTP | 412 | | **3,046** |
| 2,048 | 64K | 349 | 347 (2,048) | 890 |
| 8,192 | 64K | 463 | 613 (2,048), 605 (4,096) | 2,783 |
| 24,576 | 64K | | 653 (8,192) | **3,055** |
| 32,768 | 64K | | 825 (2,048) | **3,058** |
| 250,139 | 262K | 412 (server, 256K prompt) | 677 (2,048, K/V streamed) | 2,922 |

Fixed chunk sizes after the change (64K window): 2,048 tokens 1,343 tok/s on 8K and 4,096 tokens 1,748
(with the CPU share then on); a 262K window with MTP: 4,096 tokens 1,581 on 8K. Feeding 24K in pieces of
8,192 (as the server does): 2,737 against 2,988 in one call, and 2,957 against 3,004 at 64K without
MTP (the lent buffers stay bound between calls, but a call cannot prepare host data or stream experts
ahead into the next one).

Through the server (`ninfer-serve`, 262K window, MTP 3; the long-context client of section 4.1): a
255,686-token prompt with three planted codenames took 96.7 s to the first token (2.65K tok/s, against
10 min 21 s before), all three codenames retrieved, decode 44.2 tok/s at that depth; the follow-up
turn reused all 255,731 cached tokens and started in 0.8 s; a short request decoded at 55.7 tok/s.

Decode is not slower. With the expert cache pinned to the same size (so the same experts are cached;
64K window, K/V in VRAM, three alternating runs each): code prompt without MTP 52.2 tok/s against 48.5
before (median step 17.7 against 17.9 ms), chat prompt with MTP 2 78.7 against 73.4 tok/s. At the
default size the cache is also a little bigger, since prompt buffers above 512 tokens no longer take
VRAM for good: 14.58 / 11.50 GiB (without / with MTP) at 64K and 14.24 / 11.17 GiB at 262K, against
14.43 / 11.36 and 13.68 / 10.61 with KV streaming alone (section 4.1).

Where a chunk's time goes (`--profile`, 24K at 262K with MTP, 8.2 s on the GPU timeline): routed
experts 26% (22.9% streamed, 2.7% cached; the copy engine moved 161 GiB, 19.5 GiB/s over the prompt, and
the compute stream waited for it 0.3% of the time), DeltaNet layers 25% (input projections 11.5%,
conv 1%, recurrence 8%, output projection 4.3%), attention layers 17% (projections, QSA selection,
attention), hyper-connection mixers 18% (combine and norm 5.7%, down 7.2%, up 2.4%, gate 3%),
embedding upload and PLE 6%, shared expert, router and combine 8%. On the 250K prompt the attention
layers grow to 23% and streamed experts are 21%.

Checks on the merged tree:

- Bitwise: a streamed chunk's output does not depend on the cache. The 250K prompt gives the same
  logits hash with a 14.2 GiB and a 4 GiB cache, K/V streamed or resident, and before and after the
  merge; the 60K needle prompt (128K window, 4 GiB cache) the same state, prompt and decode hashes
  with `--kv-stream 0` and `1`; the 24K prompt at 262K with MTP the same logits hash in one call, in
  pieces of 8,192, with the serial DeltaNet kernels, and before the merge.
- Against the FP32 reference (`--compare-ref`): the 2,600-token prompt's logits differ by 3.9e-2 to
  6.4e-2 relative in every configuration measured, the old engine's included (4.8e-2 with 2,048-token
  chunks): layer 0 agrees to about 1e-6, and the error grows through the layers as expert selections
  at near-ties go the other way (about 5% of the last layers' selections differ); top-1 the same, top-5
  log-probabilities within 0.4. Short prompts: 1.2e-4 (p1) and 8.0e-3 (p3), as before. Greedy decoding
  after the 2,600-token prompt: the same 32 tokens as the old engine.
- Kernels: `test_expert_stream` (GPU conversion byte-identical to `export_expert` + `pack_expert_rows`
  for every format), `test_gemm_tc`, `test_gpu_experts_batch`, `test_qsa --kv-only`.
- Snapshot and restore after a streamed prompt (`--test-snapshot`: 24K at 262K with MTP, in pieces
  of 8,192 and with a 4 GiB cache: identical continuations), and with a 5,000-token detour between the
  snapshot and the restore without an expert cache, K/V resident and streamed (identical, and the same
  verify and draft hashes both ways). With a cache a detour changes which experts are cached (the
  prompt re-ranks it; a lent prompt refills the lent slots with the best-ranked experts) and CPU
  experts round activations to 8 bits, so the continuation differs in the last bits, as it did before
  this work. Follow-ups of 3,072, 6,144 and 500 tokens after 24K: 1,396, 2,555 and 327 tok/s.

### 4.3 Images

Tools: `fn_vision` (`tools/flashnext/fn_vision.cpp`) and llama.cpp's side, `vision_ref`
(`tools/flashnext/llama_ref/`, a CPU build against the llama.cpp checkout; FP32 attention, the whole
prompt in one ubatch). Test images: a photograph (448 x 448, 196 tokens), an editor screenshot with an
error trace (800 x 480, 375), a bar chart (640 x 480, 300), a 1024 x 1024 wallpaper (1,024), all with
sides that are multiples of 32 so that both sides see the same pixels.

The encoder, against a double-precision CPU reference of the same formulas (`fn_vision --cpu-ref`),
relative L2 error of the rows:

| Image | Tokens | Ours | llama.cpp | Ours vs llama.cpp (mean cosine) |
|---|---:|---:|---:|---:|
| chart | 300 | 5.7e-6 | 1.0e-3 | 1.0e-3 (0.9999993) |
| photo | 196 | 2.8e-6 | 1.9e-3 | 1.9e-3 (0.9999984) |
| screenshot | 375 | | | 2.1e-3 (0.9999982) |
| wallpaper | 1,024 | | | 5.6e-3 (0.9999827) |

llama.cpp's error is its own (activations rounded to f16 for the F16 weights and in flash attention, a
table GELU); the merger's GELU variant changes nothing at these digits.

Encode time on the GPU (`fn_vision --repeat 3`, warm, idle machine; the stages from
`NINFER_FN_VISION_PROFILE=1`), and llama.cpp on the CPU (16 threads) for comparison:

| Image | Patches | Tokens | Encode | Attention | Other layers (QKV, MLP, merger) | Workspace lent | llama.cpp CPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 448 x 448 | 784 | 196 | 37 ms | 6 ms | 29 ms | 272 MiB | 1.6 s |
| 640 x 480 | 1,200 | 300 | 52 ms | 11 ms | 39 ms | 283 MiB | 2.7 s |
| 800 x 480 | 1,500 | 375 | 59 ms | 14 ms | 43 ms | 290 MiB | 4.0 s |
| 1024 x 1024 | 4,096 | 1,024 | 204 ms | 100 ms | 100 ms | 356 MiB | 19.5 s |
| 1920 x 1088 | 8,160 | 2,040 | 653 ms | 450 ms | 195 ms | 458 MiB | |
| 2048 x 2048 | 16,384 | 4,096 (the most per image) | 2.07 s | 1.69 s | 0.37 s | 666 MiB | |

Small images are bound by the weights' copy (0.9 GB per image over PCIe, hidden behind the layers from
about 1,000 patches on); large ones by the attention (quadratic in the patches, about 18 TFLOPS on the
CUDA cores; the GEMMs run at about 35 TFLOPS). The first encode after the server starts adds about
40 ms (cuBLAS set-up).

The whole model on image prompts, against llama.cpp on the CPU (`vision_ref case`, then
`fn_vision --case`; first next-token logits and 24 greedy tokens; "llama rows" feeds llama.cpp's image
rows to our engine, isolating the language model):

| Prompt | Tokens (image) | First logits vs llama.cpp: rel L2, top-10 shared | Greedy tokens equal |
|---|---:|---|---|
| text only (control) | 67 (0) | 0.085, 10 | 24 of 24 |
| chart | 325 (300) | 0.32, 8 (llama rows: 0.33, 8) | 18 of 24, a near-tie, then the same (llama rows: 24 of 24) |
| photo + screenshot | 618 (571) | 0.36, 10 (llama rows: 0.34, 10) | 5 of 24, then synonyms ("rugged coastline" for "coastal landscape") and back in step |
| 2,900 tokens of notes + chart (QSA sparse, positions shifted) | 3,288 (300) | 0.20, 8 | 24 of 24 |

Top-1 agrees everywhere. With images the logits sit further from llama.cpp's than with text; our rows
and llama.cpp's give the same continuations, so the differences come from the language model's rounding,
not from the encoder or the positions. Against the FP32 reference model on the same inputs
(`fn_vision --compare-ref`, chart prompt): rel L2 3.4e-2 with top-10 identical (0.15 and 9 of the top 10
with llama.cpp's rows in both), in line with text prompts of that length (section 4.2: expert selections
that flip at near-ties).

Text-only work is unchanged. `fn_generate --hash` with a pinned cache gives bitwise the same prompt, decode,
verify and draft hashes as main for the oracle prompts p1 (with the snapshot test), p2 and p3, the
2,600-token prompt at 262K with MTP 3 and a 7,800-token prompt fed in pieces of 1,536 with MTP. Decode,
`fn_generate -n 256` at 64K, three alternating runs each: code without MTP 53.5 tok/s (main) and 53.7 (this
tree), chat with MTP 2 89.8 and 90.1. The server with `--flashnext-vision` keeps 3 experts fewer in its
cache (11.16 against 11.17 GiB: the cuBLAS handle) and the encoder's 0.9 GB in pinned RAM; text requests
decode and prefill as before (23.6K-token prompt: 2,759 tok/s with the encoder loaded, 2,763 on main).

Through the server (`ninfer-serve --flashnext-vision`, 262K window, MTP 3, temperature 0): a photograph
described, an error screenshot read verbatim (port, exception, quoted value), a chart's values, a follow-up
turn reusing 595 of 628 tokens, an image in the middle of a conversation (402 of 436 reused), two images
told apart, a 1920 x 1080 screenshot resized to 60 x 34 tokens (encode 0.66 s, TTFT 3.5 s), the Anthropic
endpoint, a 66,652-token prompt with an image at its end (both the planted codename and the chart value;
TTFT 24.9 s, decode 44 tok/s), an image across the boundary of the server's 8,192-token prompt pieces, and
a six-turn conversation fed piecewise with images in two turns (MTP on, no invalid drafts): all correct on
the base model and on Huihui. In `fn_vision`, the chart prompt fed in pieces of 100 tokens (the image split
three times) and the 3,288-token prompt in pieces of 1,000 keep all 24 greedy tokens equal to llama.cpp's. An image that a conversation resends is
neither encoded nor computed again: its turn reuses the cached prompt (TTFT 0.3 s) and the encoded-image
cache serves new prompts with it.

### 4.4 Decode after prompts: the cache follows decoding

Same-PC comparison with Strata 0.1.39 (2026-10-05): the same UD-IQ4_XS file and token ids, a 64K
window, 256 greedy tokens; `fn_generate -m <shard 1> --tokens-file <ids> -n 256 --ctx 65536
--routing-stats <fresh copy of fn_ref\routing.bin> [--mtp <shared-Q8_0 head> --draft 3]`, main (3900904b)
and this work alternating, two runs each (means); Strata with `work\scratch\h2h.ps1`'s settings (`--spec
4 --spec-min-p 0.5`, its own expert profile), one run. Prompts: `fn_ref\p2.ids` (15 tokens) and the
first 32,768 ids of `fn_ref\kv_ctx64k.ids` (source code). "Hits": the decode's pairs computed from the
VRAM cache.

| | Ours, main | Ours, now | Strata 0.1.39 |
|---|---|---|---|
| Short prompt, plain | 54.4 tok/s, 67.3% hits | **62.2 tok/s**, 78.9% hits | |
| Short prompt, MTP | 67.0 tok/s, 3.82 tokens per step, 57.1 ms per step, 94.5% accepted, 59.5% hits | **83.0 tok/s**, 3.82 tokens per step, 46.0 ms per step, 94.5% accepted, 76.3% hits | 54.8 tok/s, 3.44 tokens per round, 62.3 ms per round |
| 32K prompt, prefill | 3,054 tok/s | 3,066 tok/s | 3,470 tok/s |
| After the 32K prompt, plain | 46.0 tok/s, 62.8% hits | **53.2 tok/s**, 73.2% hits | |
| After the 32K prompt, MTP | 42.0 tok/s, 3.24 tokens per step, 77.1 ms per step, 74.3% accepted, 36.4% hits | **65.6 tok/s**, 3.24 tokens per step, 49.4 ms per step, 75.1% accepted, 74.4% hits | 81.7 tok/s, 2.51 tokens per round, 30.7 ms per round |

A step with MTP after the 32K prompt, before and after (per step, from the engine's counters):

| | main | now |
|---|---:|---:|
| Drafts (the first one also refills the lent slots, and on main re-ranks the cache and computes the catch-up's experts) | 13.5 ms | 8.4-9.2 ms, of which the MTP experts on the CPU 2.8-3.3 ms |
| Verification of 4 tokens | 63.2 ms | 39.8-40.7 ms |
| - the CPU's experts | 50.3 ms (18.1 distinct experts per layer) | 26-27 ms (8.6 distinct experts per layer) |
| - the CPU waiting for the GPU's layers | 10.0 ms | 10.6-10.8 ms |
| Expert cache | 11.50 GiB, re-ranked once after the prompt | 13.99 GiB, 81 swaps per step in the background |

So a verification step of four tokens costs what its distinct CPU experts cost: the CPU groups a step's
pairs by expert (57-66 us per expert whatever the step's size), the dense part grows from 8 to 10 ms
from one token to four, and the GPU's cached experts (per pair) always finish before the CPU's. Nsight on
verification steps (short prompt, MTP 3, steps 32-40): per step 19.1 ms of the GPU waiting for the CPU,
10.2 ms of cached experts (`k_gate_up` 4.3, `k_down` 5.9; about 29 pairs per layer), 11.7 ms of dense
GEMVs (fused 6.3, Q8_0 3.2, the Q6_K head 1.7, F32 0.5), 1.1 ms of DeltaNet recurrence and about 3 ms of
small kernels; all 408 layer waits were longer than 20 us (median 265 us), so grouping the GPU's pairs by
expert would not shorten a step. With MTP, verification now beats plain decoding at 32K depth by 23%
(65.6 against 53.2 tok/s; it was 9% slower). Strata stops drafting below a draft probability of 0.5
(`--spec-min-p 0.5`), so its rounds verify fewer tokens (2.51 committed per round against our 3.24 per
step) at a lower cost per round; `fn_generate` always verifies four (the server adapts the draft length).

Other prompt shapes (one run each, main against now): the 32K prompt fed in pieces of 1,536 tokens (the
server's way, with a snapshot after each), plain 61.9 against 60.3 tok/s (main re-ranked the cache after
each piece: 75.6% hits; now 80.2%, with 37 swaps per step) and MTP 39.3 against 70.6; follow-ups of 300
and 2,000 tokens after the 32K prompt, then decoding: plain 33.3 (32.0% hits) against 58.5 (80.3%), MTP
36.2 against 65.1.

Checks:

- Bitwise, with the cache pinned to 8 GiB (`--cache-mib 8192 --hash`): with `--no-decode-adapt` (and
  `--mtp-experts-vram` with MTP) this work and main give the same prompt, decode, verification and draft
  hashes on p1 (plain), p3 (MTP 2), the 2,600-token prompt (MTP 3) and the 32K prompt (plain): the
  catch-up's shortening, the transfers through the SMs and the step bookkeeping change nothing. With the
  defaults, the prompt hashes are still main's, the draft hashes too (MTP experts on the CPU), and decode
  and verification logits differ through placement only.
- Against the FP32 reference (`--compare-ref`): p1 1.233e-4 and p3 8.026e-3, as main; with an MTP head
  loaded (the larger cache computes more of the short prompts' pairs on the GPU) 1.258e-4 and 8.024e-3;
  the 2,600-token prompt 6.439e-2 with an MTP head loaded, as main (streamed chunks do not depend on the
  cache).
- Snapshots: `--test-snapshot` on p1 with MTP, and on the 32K prompt fed in pieces of 8,192 with MTP,
  gives bitwise the same continuation with `--no-decode-adapt` and the same tokens with adaptation (the
  cache then changes between the two continuations, so logits differ in the last bits; the test now
  compares logits only without it). With a 5,000-token detour, main's and this work's continuations
  differ in the last bits as before (the detour's prompt re-ranks or adapts the cache); without a cache
  (`--cache-mib 0 --no-decode-adapt`) they are bitwise the same.
- `test_cpu_experts --mtp`: the MTP layer's experts 5.5e-5 from exact math, 4 tokens bitwise equal to
  single-token calls.

### 4.5 Decode at 250K depth

`fn_ref\kv_needle250k.ids` (250,139 tokens of this repository's source with three facts planted at 11%,
45% and 89%, then the question), a 262K window, `fn_generate --ctx 262144 -n 128 [--mtp ... --draft 3]`,
one run each (2026-10-05). All four runs answer "1) 58213 2) Pistachio Thunderbolt 3) 6:47 in the
morning", the three facts (main and this work generate the same 128 tokens, plain and with MTP).

| | main | now |
|---|---|---|
| Expert cache (plain / MTP) | 14.24 / 11.17 GiB | 14.24 / 13.66 GiB |
| Plain | 27.7 tok/s, 21.7% hits | **45.1 tok/s**, 69.3% hits (median step 18.3 ms) |
| MTP 3 | 25.6 tok/s, 2.61 tokens per step, 55.1% accepted | **44.9 tok/s**, 2.56 tokens per step, 53.3% accepted |

Where a plain step's time goes at that depth (now; the engine's counters, and Nsight over 16 steps): the
CPU waiting for the GPU's layers 9.2 ms (8.2 ms after the 32K prompt), the CPU's experts 10.2 ms (3.1
distinct experts per layer), the head and the step's end about 1 ms; the first step also refills the
slots the prompt borrowed (0.21 s, 1.7 ms per token over 128 tokens). On the GPU: dense GEMVs 6.1 ms (fused
3.5, Q8_0 1.7, head 0.6, F32 0.3), the attention layers 1.55 ms (indexer scores over the 62,500 blocks
0.47, top-k selection 0.54, KV page copies 0.17, attention 0.18, page resolution 0.07), cached experts
1.5 ms, DeltaNet 0.3 ms, small kernels about 1.3 ms. With MTP a verification step of four tokens costs 45
ms (CPU experts 28.4 ms, 8.5 distinct per layer; the GPU's part 12.4 ms, of which attention 2.9 ms) for
2.56 tokens: MTP no longer pays at that depth, because the drafts' acceptance falls with depth (94.5% on
the short prompt, 75% after 32K tokens, 53% here).

The MTP layer attends to its last 2,051 positions only. `NINFER_FN_MTP_WINDOW=32768` (an experiment: a
ring of 32K + 512 rows, 68 MB, which snapshots carry too; single-token drafts only) measured, one run each:
after the 32K prompt 81.3% accepted and 68.1 tok/s (75.1% and 65.5 with 2,051), at 250K 56.9% and 45.9
tok/s (53.3% and 45.7), on the short prompt unchanged (94.5%, 81.9 against 83.0 tok/s); each draft's
attention costs about 0.14 ms more.

## 5. Next

1. Prompts: the streamed experts are now about a quarter of a chunk's GPU time and overlap their
   copies; what is left is dense work (DeltaNet and attention projections, the hyper-connection
   mixers) and, at depth, QSA selection. Staging deep chunks' K/V costs 2.3% at 250K (section 4.1);
   staging only the blocks a chunk's selections name would save most of that. Small prompts (600 to
   2,048 tokens) are bound by PCIe (a whole layer's non-cached experts for few tokens): a cache that
   keeps more of the experts a prompt needs, or a CPU share that pays off, would help them.
2. Find why a short request through the server decodes slower than `fn_generate` with the same
   window (46 against 77 tok/s; different prompts, so measure like for like first).
3. Higher-precision expert quantizations within the RAM budget, and an evaluation through Pi.
4. Images: the encoder's attention is FP32 on the CUDA cores and dominates large images (quadratic in
   the patches: about 70% of a 1,024-token image's encode); better register tiling, or tensor cores with
   split fp16 operands, would cut it. The MTP catch-up could feed image rows instead of the image token's
   embedding. Video (the frontend already samples frames) needs the temporal positions and timestamps.
5. Decode at depth (section 4.5). (a) MTP: acceptance falls with depth; a wider MTP window (measured +6
   points of acceptance after 32K tokens, +4 at 250K) and a per-draft probability gate (Strata stops below
   0.5; at 250K a four-token step costs 2.4 plain steps for 2.56 tokens) would let MTP pay at depth again;
   fn_generate verifies a fixed four tokens, the server already adapts the length. (b) Expert misses are
   still half a step: the refill after a long prompt (0.2 s) could overlap the first steps (refilling in the
   background together with the re-ranking measured worse: 32K prompt, MTP, 47.4 against 57.0 tok/s, the
   lent slots queued behind the re-ranking's swaps), and
   the CPU reads experts at 40-46 GB/s of the 55-62 the RAM gives. (c) Attention at 250K is 8% of a step,
   mostly the indexer's scores (FP32 block keys, about 800 GB/s already) and the top-k selection (0.54 ms
   per token over 12 layers), where a faster exact selection (a radix select) could help.
