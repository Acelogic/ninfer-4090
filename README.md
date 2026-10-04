# NInfer-4090 Extreme

Fast local inference for **Qwen3.8 27B** and **Ternary Bonsai 2 27B** on one **RTX 4090** under
**Windows 11**. It's a native C++/CUDA engine with no WSL or Docker. It serves an OpenAI- and
Anthropic-compatible API, so coding agents such as [Pi](https://pi.dev) and OpenCode, and any
OpenAI client, can use it directly.

This is [JGamboa's NInfer-4090 for Windows](https://github.com/JGamboa/ninfer-4090-windows) with a
few additions ([what this build changes](#what-this-build-changes)). The engine is
[Neroued's NInfer](https://github.com/Neroued/ninfer). See [Credits](#credits) for everyone
involved.

## Models and speed

Each row is a ready-made profile. The profile name is also the model id that clients request.

| Profile | Model | Context | Decode speed |
|---|---|---|---:|
| `qwen3.8-27b-fast` | Qwen3.8 27B | 131K, text only | ~225 tok/s, up to 400 on edits |
| `qwen3.8-27b` | Qwen3.8 27B | 229K, text + images | ~160 tok/s |
| `qwen3.8-27b-huihui` | Qwen3.8 27B, abliterated (Huihui) | 229K, text + images | same as `qwen3.8-27b` |
| `bonsai2-27b` | Ternary Bonsai 2 27B (Prism ML) | 262K, text + images | ~245 tok/s, up to 370 on edits |
| `bonsai2-27b-heretic` | Ternary Bonsai 2 27B, abliterated (Heretic) | 262K, text + images | same as `bonsai2-27b` |

- **Speeds** are the mean over eight coding, writing, math and edit prompts, at temperature 0.6
  with low reasoning effort. They were measured on an RTX 4090 that also drives the desktop.
  Plain prose is the slowest case: about 80 tok/s on Qwen when thinking is off.
- **Long prompts** are fast too: the Qwen profiles read a 90K-token prompt in about 25 s.
- **No quality cost from speculation.** Speculative decoding guesses several tokens ahead and has
  the model check them, so the output is what the model would have written anyway, only faster.
- **Bonsai** is a ternary-compressed Qwen3.8 27B. It's 6.4 GB instead of 19 GB and keeps about 98%
  of the full model's benchmark scores, according to Prism ML.

## Quick start

**Requirements:**
- an RTX 4090 and Windows 11;
- NVIDIA driver 595 or newer;
- the [VC++ 2015-2022 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe);
- PowerShell 7.

1. Download `ninfer-4090-acelogic-<date>-windows-x64.zip` from the
   [latest release](https://github.com/Acelogic/ninfer-4090-extreme/releases/latest) and unzip it, for
   example to `C:\ninfer`.
2. Download a model into its `models` folder, using the
   [`hf` tool](https://huggingface.co/docs/huggingface_hub/guides/cli) or a browser:

   ```bat
   hf download jgamboa/Qwen3.8-27B-NInfer-4090 qwen3_8_27b_a8.ninfer --local-dir C:\ninfer\models
   hf download jgamboa/Ternary-Bonsai-2-27B-NInfer-4090 bonsai2_27b_vl_mtp_q4q5.ninfer --local-dir C:\ninfer\models
   ```

3. Start a profile:

   ```powershell
   pwsh -File C:\ninfer\Start-NInfer.ps1 -Name qwen3.8-27b-fast
   ```

The API is at `http://127.0.0.1:18085/v1`. A tray icon shows the loaded model and VRAM use and can
stop the engine; `Stop-NInfer.ps1` stops it from a script.

[windows/README.md](windows/README.md) has the full profile settings, the tray, how agents can stop
the engine automatically when they exit, and how to set up the two abliterated models.

## What this build changes

On top of JGamboa's `v2026.09.27b`:

- **Deeper MTP speculation:** up to 7 drafts per round instead of 5.
- **More context on a desktop GPU:** on Windows, the engine sizes its memory from the free VRAM
  measured before the weights load, because the reading taken after the load is about 1 GiB low.
  Qwen3.8 27B now fits 229K context with vision instead of 196K.
- **Windows tooling** in [`windows/`](windows/README.md): the profile launcher, stop script, tray
  icon and optional [Froggeric chat templates](third_party/froggeric).
- **Windows fix for the v2→v3 artifact upgrade tool:** `tools/upgrade_ninfer_v2_to_v3.py` now runs
  on Windows.

The previous Acelogic engine, based on UDPSendToFailed's NInfer-4090 with a native Froggeric
renderer, is still available as release
[`v1.2.0-froggeric-v22.5`](https://github.com/Acelogic/ninfer-4090-extreme/releases/tag/v1.2.0-froggeric-v22.5).

## Build from source

From an "x64 Native Tools" prompt, with CUDA 12.8 or newer, CMake 3.28+, Ninja and
[vcpkg](https://github.com/microsoft/vcpkg):

```bat
git clone https://github.com/Acelogic/ninfer-4090-extreme
cd ninfer-4090-extreme
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DVCPKG_TARGET_TRIPLET=x64-windows -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

The build produces `ninfer.exe` (CLI), `ninfer-serve.exe` (server) and `ninfer-perplexity.exe` in
`build\apps`. Build details are in [WINDOWS_PORT.md](WINDOWS_PORT.md). Add
`-DNINFER_WITH_FLASHNEXT=ON` to also serve Qwen3.8-Flash-Next GGUF models (an AVX-512 CPU is
required); see [docs/serving.md](docs/serving.md#flash-next-gguf-models).

## Documentation

| Topic | Document |
|---|---|
| Full guide: running by hand, performance, settings, benchmarks, converting models, serving | [docs/guide.md](docs/guide.md) |
| Launcher, profiles and tray icon | [windows/README.md](windows/README.md) |
| CLI options | [docs/cli.md](docs/cli.md) |
| HTTP server and API reference | [docs/serving.md](docs/serving.md) |
| Windows build and measurements | [WINDOWS_PORT.md](WINDOWS_PORT.md) |
| Everything else | [docs/README.md](docs/README.md) |

## Credits

This engine is the work of a chain of projects. All commits keep their original authors.

| Project | Author | Contribution |
|---|---|---|
| [NInfer](https://github.com/Neroued/ninfer) | Neroued | The engine, built for the RTX 5090 (`sm_120a`) |
| [NInfer-3090](https://github.com/Don-Chad/ninfer-3090) | Don-Chad | The `sm_86` compatibility layer, ReplaySSM integration and Qwen3.8 runtime support |
| [NInfer-4090](https://github.com/sergiuszm/ninfer-4090) | sergiuszm | The RTX 4090 (`sm_89`) port: Ada-tuned prefill attention and serving features (`/metrics`, `/slots`, slot save/restore, long anchors) |
| [NInfer-4090](https://github.com/UDPSendToFailed/ninfer-4090) | UDPSendToFailed | The rotated and E8-lattice KV cache modes and the vision scratchpad; the Windows VRAM budgeting idea ported here; the base of the previous Acelogic engine |
| [NInfer-4090](https://github.com/shantanusingh16/ninfer-4090) | shantanusingh16 | llama.cpp-style `timings` on chat completions |
| [NInfer-4090 for Windows](https://github.com/JGamboa/ninfer-4090-windows) | JGamboa | The base of this build: the native Windows build, Ternary Bonsai support (converter, format, kernels, vision), int8 prefill, n-gram speculation, concurrent lanes and DFlash2 verification |
| This repository | Acelogic | The changes listed above |

JGamboa developed the Bonsai port with [Claude Code](https://claude.com/claude-code). Every kernel
was checked against FP64 oracles, and every performance claim was measured on the RTX 4090.

Models and templates:

- **Ternary Bonsai 2 27B**, its packings and Hadamard rotation: [Prism ML](https://huggingface.co/prism-ml), Apache-2.0.
  Their [llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) defined the formats read here.
  [fraserprice/bonsai-vllm](https://github.com/fraserprice/bonsai-vllm) was a CUDA reference.
- **Qwen3.8 27B:** the Qwen team, Apache-2.0. The DFlash2 drafter used by `qwen3.8-27b-fast`:
  [z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2).
- **Huihui abliterated Qwen3.8 27B:** huihui-ai, converted to NInfer's format by Barding-Defense, Apache-2.0.
- **Heretic abliterated Bonsai:** converted from OS-Software's
  `Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF`, Apache-2.0.
- **Qwen Fixed Chat Templates v22.5:** Frederic Guigand (froggeric), Apache-2.0, kept in
  [`third_party/froggeric`](third_party/froggeric).

## License

Apache License 2.0, as upstream; see [LICENSE](LICENSE). The Windows release zip also includes the
licenses of the libraries it redistributes (FFmpeg LGPL-2.1, curl, zlib) in `licenses/`.
