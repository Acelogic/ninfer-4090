# Windows launcher and tray icon

These scripts ship in the Windows release zip next to `ninfer-serve.exe`. They start and stop the server for Qwen3.8 27B, and show a tray icon that tells you whether a model is loaded and lets you free the GPU.

| File | Purpose |
|---|---|
| `Start-Qwen27.ps1` | Starts `ninfer-serve` on `127.0.0.1:18085`, records the process in `server-process.json`, and starts the tray icon |
| `Stop-Qwen27.ps1` | Stops the recorded server, after checking that the recorded pid is still that server |
| `Start-Qwen27-Huihui.ps1` | Example variant: serves another groupwise-int artifact under its own model id |
| `NInfer-Tray.ps1`, `Start-NInferTray.ps1` | Tray icon (see below) |
| `ninfer-tray.json` | Engines listed in the tray's Start menu, and its settings |

## Setup

1. Extract the release zip, for example to `C:\ninfer`.
2. Put the model in `models\`.

   This engine reads groupwise-int artifacts in the v1/v2 container. Upstream NInfer now publishes v3 artifacts, so pin the last v2 revision of the official model:

   ```bash
   hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer --revision 18dfc887423fa5aabf3cb56fac41490e462b3fab --local-dir models
   ```

   The expected SHA-256 is `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e`.
3. Run `pwsh -File .\Start-Qwen27.ps1`. The OpenAI-compatible API is at `http://127.0.0.1:18085/v1` with model id `qwen3.8-27b`.

`Start-Qwen27.ps1` takes these options:

| Option | Default | Choices |
|---|---|---|
| `-Template` | `Froggeric` (native v22.5 template) | `Artifact` (template embedded in the model) |
| `-Context` | 262144 | |
| `-Mtp` | 7 | 0–7 |
| `-KvType` | `rk4v4-e8` | `rk4v4`, `rk8v4`, `int8`, … |
| `-ModelFile`, `-ModelId` | official 27B | another artifact in `models\` and the id to serve it under |
| `-TextOnly` | off | disables vision |

Use PowerShell 7 (`pwsh`).

## Tray icon

The icon color shows the engine state:

| Color | State |
|---|---|
| Green | A model is served |
| Amber | Loading |
| Grey | Stopped |

Hovering shows the model and VRAM use. The menu stops the engine, starts any engine listed in `ninfer-tray.json`, toggles auto-stop, and opens the logs folder. The tray exits on its own about 15 seconds after the engine stops. `Start-Qwen27.ps1` starts it, and only one runs per desktop session.

### Auto-stop when a client exits

Closing a terminal kills the client running in it before the client can clean up, which can leave a 20 GB model on the GPU. To avoid that, clients can register a lease:

1. Write `%LOCALAPPDATA%\ninfer-tray\leases\<pid>.json` containing:

   ```json
   {"pid": 1234, "startedMs": 1790000000000, "startedBackend": true}
   ```

   - `startedMs` is the client process's start time in Unix milliseconds. The tray compares it with the live process, so a reused pid is not mistaken for the client.
   - Set `startedBackend` to `true` when this client started the engine.
2. Delete the file on a clean exit.

When at least one lease says the client started the engine, and none of the lease processes is still running for two polls (about 6 seconds), the tray stops the engine. An engine you started by hand is never auto-stopped. The [pi](https://pi.dev) extension `pi-local-backend` implements this contract.
