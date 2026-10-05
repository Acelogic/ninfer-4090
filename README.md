# NInfer-4090 Extreme

One native C++/CUDA inference engine, tuned for one machine: an **RTX 4090** (24 GB), a **Ryzen 9 7950X**
(16 cores, AVX-512) and **192 GB of DDR5**, on **Windows 11**. A single `ninfer-serve.exe` serves three model
families behind OpenAI- and Anthropic-compatible APIs, so coding agents such as [Pi](https://pi.dev) use it directly:

- **Qwen3.8-Flash-Next**: a 125B-parameter mixture of experts. This is the main model. It runs on our own Flash-Next
  engine and needs the whole machine: the dense layers and the busiest experts on the GPU, the remaining experts in
  RAM and on the CPU.
- **Qwen3.8 27B** and **Ternary Bonsai 2 27B**: dense models that fit on the GPU, on the NInfer engine.

Abliterated (uncensored) versions of all three are supported.

## Models and speed

Each row is a ready-made profile. The profile name is also the model id that clients request.

| Profile | Model | Context | Writes | Reads prompts |
|---|---|---|---:|---:|
| `qwen3.8-flash-next` | Qwen3.8-Flash-Next, Unsloth UD-IQ4_XS (94 GB) | 262K, text | 50-60 tok/s; ~44 at 256K deep | 2,500-3,000 tok/s |
| `qwen3.8-flash-next-huihui` | Qwen3.8-Flash-Next, abliterated (Huihui), same recipe | 262K, text | same as above | same |
| `qwen3.8-27b-fast` | Qwen3.8 27B | 131K, text | ~225 tok/s, up to 400 on edits | ~3,600 tok/s |
| `qwen3.8-27b` | Qwen3.8 27B | 208K, text + images | ~160 tok/s | ~3,600 tok/s |
| `qwen3.8-27b-huihui` | Qwen3.8 27B, abliterated (Huihui) | 208K, text + images | same as `qwen3.8-27b` | |
| `bonsai2-27b` | Ternary Bonsai 2 27B (Prism ML) | 262K, text + images | ~245 tok/s, up to 370 on edits | |
| `bonsai2-27b-heretic` | Ternary Bonsai 2 27B, abliterated (Heretic) | 262K, text + images | same as `bonsai2-27b` | |

- All numbers were measured on the machine above, with the GPU also driving the desktop.
- **Flash-Next** writes 50-60 tok/s on short replies through the server, and about 44 tok/s 256K tokens deep. These
  numbers use multi-token prediction with 1-3 drafts, chosen adaptively.
- **Prompt reading:** Flash-Next reads prompts at 2,500-3,000 tok/s, so Pi's ~24K-token system prompt takes about
  11 s and a 256K-token prompt about 97 s. In an agent session, a follow-up turn starts in about 1 s, because the
  conversation's state is reused.
- **No quality cost from speculation:** speculative decoding drafts several tokens ahead and has the full model
  check them, so the output is what the model would have written anyway.
- **Exact numerics:** the Flash-Next engine keeps FP32 activations and the file's exact weights. Its logits match an
  FP32 reference forward pass to a relative 1.6e-4.
- **Bonsai** is a ternary-compressed Qwen3.8 27B: 6.4 GB instead of 19 GB, keeping about 98% of the full model's
  benchmark scores according to Prism ML.

## How Flash-Next runs on one 24 GB GPU

Qwen3.8-Flash-Next has 48 layers:

- 36 Gated DeltaNet layers and 12 gated-attention layers with QSA sparse attention, which reads 2,051 selected
  positions;
- in every layer, a mixture of 512 experts (10 per token) plus a shared expert;
- four-stream hyper-connections, per-layer token embeddings, and an MTP head for drafting.

The 4-bit file is 94 GB, so it can't sit in VRAM. The engine splits the work across the machine:

- **GPU:** every dense layer, attention, and an adaptive cache of the most-used experts (5-6 thousand of the 24,576).
  The cache follows each conversation's routing. One CUDA graph runs per decode step.
- **CPU:** experts that miss the cache run on all 16 cores with AVX-512 VNNI kernels, over lossless repacks of the
  quantized weights. The GPU and the CPU work at the same time and meet through mapped memory.
- **Prompts:** the GPU computes every expert, in chunks of up to 8,192 tokens. The experts that aren't in VRAM
  stream from pinned RAM over PCIe while earlier layers compute. This follows Strata's design, but uses this engine's
  exact kernels, so the results are bit-identical however the experts are cached or streamed.
- **Long context:**
  - QSA keeps attention cost flat with depth.
  - The attention cache lives in pinned RAM, with a page cache in VRAM, also following Strata's design. A 262K window
    costs about 1 GB of VRAM instead of 7 GB, so the expert cache keeps about 10.6 GiB, and decode at a 262K window is
    as fast as at 64K.
  - Facts placed at 10%, 50% and 90% of a 256K-token prompt are all retrieved.
- **Memory:** the experts' CPU copy (about 66 GB) and the attention cache (6 GB at 262K) are pinned in RAM, so the GPU
  can read them directly.

[docs/maintainer/flash-next-engine.md](docs/maintainer/flash-next-engine.md) has the design and the measurements.

## This machine's layout

Everything for the engine lives in one folder, `Developer\ninfer-extreme`, with the models next to it:

```text
Developer\
  ninfer-extreme\      this repository (one branch: main)
    app\               the installed runtime: ninfer-serve.exe, launcher, tray, profiles, logs  (not in git)
    build\             build output                                                             (not in git)
    work\              local tools, test data, comparisons                                      (not in git)
  models\              every model file; app\models points here
    qwen3.8-flash-next\UD-IQ4_XS\...gguf, qwen3.8-flash-next\MTP\...gguf
    qwen3.8-flash-next-huihui\UD-IQ4_XS\...gguf
    qwen3_8_27b_a8.ninfer, qwen3_8_27b_huihui_v3.ninfer, bonsai2_27b_*.ninfer
  build-tools\         CUDA 13.3, vcpkg and its packages
```

## Build and install

You need Visual Studio 2022 Build Tools, plus CUDA 13.3 and vcpkg in `..\build-tools`. From PowerShell 7:

```powershell
pwsh -File windows\Build-NInfer.ps1 -Configure -Install
```

This builds one `ninfer-serve.exe` (all three model families), `ninfer.exe`, `ninfer-perplexity.exe` and the
Flash-Next developer tools. `-Install` then copies the server, the launcher scripts, the tray and the profiles into
`app\`, backing up the previous profiles first. `-Test` runs the test suite. Build details:
[WINDOWS_PORT.md](WINDOWS_PORT.md).

To start a model:

```powershell
pwsh -File app\Start-NInfer.ps1 -Name qwen3.8-flash-next
```

The API is at `http://127.0.0.1:18085/v1`. A tray icon shows the loaded model and VRAM use and can stop the engine.
`app\Stop-NInfer.ps1` stops it from a script.

## Using it from Pi

1. Add the engine as an OpenAI-compatible provider in `~/.pi/agent/models.json`:

   ```json
   "ninfer": {
     "baseUrl": "http://127.0.0.1:18085/v1",
     "api": "openai-completions",
     "apiKey": "local",
     "compat": { "supportsStore": false, "supportsDeveloperRole": false, "supportsReasoningEffort": true,
                 "maxTokensField": "max_tokens", "thinkingFormat": "qwen", "supportsStrictMode": false },
     "models": [
       { "id": "qwen3.8-flash-next", "name": "Qwen3.8 Flash-Next (NInfer Extreme)", "reasoning": true,
         "contextWindow": 262144, "maxTokens": 32768 }
     ]
   }
   ```

2. Add one entry per profile.
3. On this machine, two Pi extensions complete the setup:
   - **Local backend:** starts the right profile when Pi first uses it, and stops it when Pi exits.
   - **Ask an expert:** hands the local model's hard moments to a stronger hosted model. This happens when you ask
     for help, run `/expert`, or the extension notices the model is stuck: repeated failures, very long responses,
     or the same file edited over and over.

Local models rarely ask for help on their own, so the automatic triggers are what make the fallback work. With
Flash-Next, 24 of the extension's 25 stress scenarios pass. In the 25th, a fake reviewer asks for a file the user never
requested; Flash-Next checks the review against the request and declines to invent the requirement.

## Abliterated models

- **Flash-Next (Huihui):** built here from huihui-ai's BF16 release by patching Unsloth's UD-IQ4_XS file.
  - Huihui's abliteration changes only the four tensor kinds that write into the residual stream: attention output,
    DeltaNet output, and the routed and shared experts' down projections. Every other tensor is byte-identical to
    the original model, so it keeps Unsloth's quantization.
  - The changed tensors (82 GB of BF16 out of 354 GB) are quantized to the type the file already has there, IQ4_NL or
    Q8_0. This uses ggml's own quantizer and mradermacher's importance matrix of the abliterated model (statistics
    that guide the rounding).
  - The result has the same formats, size and speed as the original.
  - The tools are in [`tools/flashnext/abliterated/`](tools/flashnext/abliterated):
    - `find_changed.py` compares byte samples of both BF16 releases over HTTP;
    - `download_changed.py` fetches only the changed tensors;
    - `patch_tensors.py` quantizes them into a copy of the original file.
- **Qwen3.8 27B (Huihui)** and **Bonsai (Heretic):** see [windows/README.md](windows/README.md#abliterated-bonsai).

## What this build changes

On top of JGamboa's NInfer-4090 for Windows `v2026.09.27b`:

- **The Flash-Next engine** (`src/flashnext/`): Qwen3.8-Flash-Next GGUF models in `ninfer-serve`, written for this
  machine:
  - GPU kernels for every dense format and expert format in the file;
  - AVX-512 CPU expert kernels;
  - QSA sparse attention up to 262K, with the attention cache paged through RAM;
  - prompts that stream every uncached expert to the GPU;
  - MTP speculative decoding with an adaptive draft length;
  - an adaptive VRAM expert cache, and prefix reuse across agent turns.
- **One build:** Flash-Next is part of the default build; `windows/Build-NInfer.ps1` builds, tests and installs it.
- **Deeper MTP speculation** for the 27B models: up to 7 drafts per round instead of 5.
- **More context on a desktop GPU:** memory is sized from the free VRAM measured before the weights load. Qwen3.8 27B
  fits 208K context with vision while the GPU also drives the desktop. At 229K with vision, Windows pages GPU memory
  and decode drops from ~170 to ~75 tok/s.
- **Windows tooling** in [`windows/`](windows/README.md): the profile launcher, stop script, tray icon and optional
  [Froggeric chat templates](third_party/froggeric).

## Documentation

| Topic | Document |
|---|---|
| Flash-Next engine: design, measurements, next steps | [docs/maintainer/flash-next-engine.md](docs/maintainer/flash-next-engine.md) |
| Launcher, profiles and tray icon | [windows/README.md](windows/README.md) |
| HTTP server and API reference (including Flash-Next options) | [docs/serving.md](docs/serving.md) |
| Full NInfer guide: running by hand, settings, benchmarks, converting models | [docs/guide.md](docs/guide.md) |
| CLI options | [docs/cli.md](docs/cli.md) |
| Windows build | [WINDOWS_PORT.md](WINDOWS_PORT.md) |

## Credits

This engine is the work of a chain of projects. All commits keep their original authors.

| Project | Author | Contribution |
|---|---|---|
| [NInfer](https://github.com/Neroued/ninfer) | Neroued | The engine, built for the RTX 5090 (`sm_120a`) |
| [NInfer-3090](https://github.com/Don-Chad/ninfer-3090) | Don-Chad | The `sm_86` compatibility layer, ReplaySSM integration and Qwen3.8 runtime support |
| [NInfer-4090](https://github.com/sergiuszm/ninfer-4090) | sergiuszm | The RTX 4090 (`sm_89`) port: Ada-tuned prefill attention and serving features |
| [NInfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) | UDPSendToFailed | Rotated and E8-lattice KV cache modes, the vision scratchpad, the Windows VRAM budgeting idea |
| [NInfer-4090](https://github.com/shantanusingh16/ninfer-4090) | shantanusingh16 | llama.cpp-style `timings` on chat completions |
| [NInfer-4090 for Windows](https://github.com/JGamboa/ninfer-4090-windows) | JGamboa | The base of this build: the native Windows build, Ternary Bonsai support, int8 prefill, n-gram speculation, concurrent lanes and DFlash2 verification |
| This repository | Acelogic | The Flash-Next engine and the changes listed above |

Ideas and formats:

- **[Strata](https://github.com/Niko1221/Strata)** (Niko1221, MIT): the design of streaming experts over PCIe while
  reading prompts, and of paging the attention cache through RAM, which this engine adapts.
- **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)** (MIT): the GGUF format and the quantization formats.
  Its `llama-quantize` produces the abliterated Flash-Next file.

Models and templates:

- **Qwen3.8-Flash-Next** and **Qwen3.8 27B:** the Qwen team, Apache-2.0.
- **UD-IQ4_XS quantization:** [Unsloth](https://huggingface.co/unsloth).
- **Abliterated Flash-Next and Qwen3.8 27B:** [huihui-ai](https://huggingface.co/huihui-ai). The 27B was converted to
  NInfer's format by Barding-Defense. The importance matrix for the Flash-Next quantization is from
  [mradermacher](https://huggingface.co/mradermacher).
- **DFlash2 drafter** used by `qwen3.8-27b-fast`: [z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2).
- **Ternary Bonsai 2 27B**, its packings and Hadamard rotation: [Prism ML](https://huggingface.co/prism-ml),
  Apache-2.0. Its abliterated version is converted from OS-Software's
  `Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF`, Apache-2.0.
- **Qwen Fixed Chat Templates v22.5:** Frederic Guigand (froggeric), Apache-2.0, kept in
  [`third_party/froggeric`](third_party/froggeric).

## License

Apache License 2.0, as upstream; see [LICENSE](LICENSE). The Windows release zip also includes the licenses of the
libraries it redistributes (FFmpeg LGPL-2.1, curl, zlib) in `licenses/`.
