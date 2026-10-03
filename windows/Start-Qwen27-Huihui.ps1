# Starts the Huihui abliterated Qwen3.8 27B (groupwise-int artifact) with the same settings as
# Start-Qwen27.ps1, served under its own model id. Extra arguments are passed through.
& (Join-Path $PSScriptRoot 'Start-Qwen27.ps1') -ModelFile 'qwen3_8_27b_huihui_abliterated.ninfer' -ModelId 'qwen3.8-27b-huihui' @args
