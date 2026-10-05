# Windows launcher, model profiles and tray icon

These scripts ship in the Windows release zip next to `ninfer-serve.exe`.

- **One launcher** starts any model profile listed in `ninfer-models.json`.
- **One stop script** stops whatever is running.
- **A tray icon** shows whether a model is loaded and lets you free the GPU.

| File | Purpose |
|---|---|
| `ninfer-models.json` | Model profiles (artifact file and server arguments), shared settings and the tray's options |
| `Start-NInfer.ps1 -Name <profile>` | Starts `ninfer-serve` on `127.0.0.1:18085`, records the process in `server-process.json`, and starts the tray. The profile name is the model id clients ask for. |
| `Stop-NInfer.ps1` | Stops the recorded server, after checking that the recorded pid is still that server |
| `NInfer-Tray.ps1`, `Start-NInferTray.ps1` | Tray icon (see below) |
| `templates/froggeric-v22.5/` | Optional chat template, selected per profile with `"template": "froggeric"` |

Use PowerShell 7 (`pwsh`).

## Profiles

All figures were measured on one RTX 4090 that also drives the Windows desktop.

| Profile | Model | Speculation | Context | Measured |
|---|---|---|---|---|
| `qwen3.8-27b-fast` | Qwen3.8 27B (int8 prefill) | DFlash2, 12 drafts | 131K, no vision | ~226 tok/s mean on short replies: 240–270 on math, 340–400 on edits, ~80 on prose |
| `qwen3.8-27b` | Qwen3.8 27B (int8 prefill) | MTP 5 + n-gram | 208K, vision | ~152 tok/s short; ~100 tok/s after a 90K-token prompt; 173K prompt prefilled in 62 s |
| `qwen3.8-27b-huihui` | Huihui abliterated Qwen3.8 27B | MTP 5 + n-gram | 208K, vision | Same weights format as `qwen3.8-27b`, with BF16 prefill |
| `bonsai2-27b` | Ternary Bonsai 2 27B (Prism ML) | MTP 2 + n-gram | 262K, vision | ~247 tok/s mean (~180 on prose); 145 tok/s after a 90K-token prompt |
| `bonsai2-27b-heretic` | Bonsai 2 27B with the Heretic abliteration | MTP 2 + n-gram | 262K, vision | Same as `bonsai2-27b` |
| `qwen3.8-flash-next` | Qwen3.8-Flash-Next (UD-IQ4_XS GGUF), experts in a VRAM cache and on the CPU | MTP up to 3, adaptive | 262K, vision | 50-60 tok/s on short replies, ~44 tok/s 256K tokens deep; prompts read at 2,500-3,000 tok/s (Pi's 24K system prompt in ~11 s, a 256K prompt in ~97 s); a follow-up turn reuses the conversation (TTFT ~1 s); an image costs 40-200 ms of encoding up to 1,000 tokens (0.65 s for a 1080p screenshot, 2,040 tokens) and is reused while the conversation resends it |
| `qwen3.8-flash-next-huihui` | Huihui abliterated Qwen3.8-Flash-Next, quantized here with the UD-IQ4_XS recipe | MTP up to 3, adaptive | 262K, vision | Same as `qwen3.8-flash-next` |

**Flash-Next** is served by the same `ninfer-serve.exe` (Flash-Next is part of the default build). Its files go in `models\qwen3.8-flash-next\` (the three UD-IQ4_XS shards in `UD-IQ4_XS\`, the shared-Q8_0 MTP head in `MTP\`, the vision encoder `mmproj-F16.gguf`) and `models\qwen3.8-flash-next-huihui\`. Both profiles load the same vision encoder (`--flashnext-vision`; the abliteration does not touch vision): images can be pasted into Pi like for the 27B profiles. Its weights stay in pinned RAM (0.9 GB) and each image borrows VRAM from the expert cache only while it is encoded, so text-only work keeps its speed. A profile may instead point `file` (and `exe`) at other locations: absolute paths work and `%USERPROFILE%` expands. Each Flash-Next profile keeps its expert routing statistics (which experts to cache) in its own file next to the launcher.

**Lossless speculation.** Speculative decoding never changes what the model would have written. Drafts are verified by the full model, so speed comes without quality loss.

**Bonsai is a different trade-off.** It is ternary-compressed Qwen3.8 27B, about 98% of the full model's benchmark scores per Prism ML, in a fraction of the memory.

## Setup

1. Extract the release zip, for example to `C:\ninfer`.
2. Put the artifacts named in `ninfer-models.json` into `models\`:

   ```bash
   hf download jgamboa/Qwen3.8-27B-NInfer-4090 qwen3_8_27b_a8.ninfer --local-dir models
   hf download jgamboa/Ternary-Bonsai-2-27B-NInfer-4090 bonsai2_27b_vl_mtp_q4q5.ninfer --local-dir models
   ```

   This build reads v3 artifacts. A v2 groupwise-int artifact, such as the Huihui abliterated conversion, can be upgraded in place with `tools/upgrade_ninfer_v2_to_v3.py`, without downloading the weights again.
3. Run `pwsh -File .\Start-NInfer.ps1 -Name qwen3.8-27b`. The OpenAI-compatible API is at `http://127.0.0.1:18085/v1`.

## Abliterated Bonsai

`bonsai2-27b-heretic` is converted locally from OS-Software's `Ternary-Bonsai-2-27B-Uncensored-Heretic-GGUF` (PTQ1_0, Apache-2.0) with this repository's converter.

- **Text weights:** taken from the GGUF, with the ternary codes kept bit-exact.
- **Tokenizer, template, config and MTP head:** taken from the Qwen3.8 27B artifact.
- **Vision tower:** taken from `mmproj-Q8_0.gguf`.

```bash
python -m tools.convert.bonsai_base --gguf Ternary-Bonsai-2-27B-Uncensored-Heretic-PTQ1_0.gguf --reference qwen3_8_27b.ninfer --mmproj mmproj-Q8_0.gguf --out bonsai2-27b-heretic-vl
python -m tools.convert --model bonsai2-27b-heretic-vl --recipe bonsai2_27b_mtp_q4q5 --components text,vision,mtp --source gguf=Ternary-Bonsai-2-27B-Uncensored-Heretic-PTQ1_0.gguf --source mmproj=mmproj-Q8_0.gguf --source mtp=qwen3_8_27b.ninfer --proposal --name bonsai2-27b-heretic --out models/bonsai2_27b_heretic_vl_mtp_q4q5.ninfer --device cuda
```

## Tray icon

The icon color shows the engine state:

| Color | State |
|---|---|
| Green | A model is served |
| Amber | Loading |
| Grey | Stopped |

Hovering shows the model and VRAM use. The menu stops the engine, starts any profile, toggles auto-stop, and opens the logs folder. The tray exits on its own about 15 seconds after the engine stops. Only one runs per desktop session.

### Auto-stop when a client exits

Closing a terminal kills the client running in it before the client can clean up, which can leave a 20 GB model on the GPU. To avoid that, clients can register a lease:

1. Write `%LOCALAPPDATA%\ninfer-tray\leases\<pid>.json` containing:

   ```json
   {"pid": 1234, "startedMs": 1790000000000, "startedBackend": true}
   ```

   - `startedMs` is the client process's start time in Unix milliseconds. The tray compares it with the live process, so a reused pid is not mistaken for the client.
   - Set `startedBackend` to `true` when this client started the engine.
2. Delete the file on a clean exit.

When at least one lease says the client started the engine, and none of the lease processes is still running for two polls (about 6 seconds), the tray stops the engine. An engine started by hand is never auto-stopped. The [pi](https://pi.dev) extension `pi-local-backend` implements this contract.

## VRAM notes for a 24 GB card that drives the desktop

**How this build budgets VRAM.** Runtime capacity is sized from the free memory measured before the weights are uploaded, minus the bytes uploaded. On WDDM, the figure measured after the upload reads about 1 GiB low.

**What that buys.** Qwen3.8 27B fits 208K context with vision while the card also drives the desktop: 154-185 tok/s on code. At 229K with vision the card is oversubscribed, Windows pages GPU memory, and the same request runs at 66-86 tok/s; at 262K, a few tokens per second.

**Getting more room.** Connecting the monitor to the CPU's integrated graphics frees about 1.3 GB of VRAM and the display's share of GPU time. The engine's own measurements put that share at 5–18 % of decode speed.
