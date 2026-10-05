<#
Runs the Qwen3.8-Flash-Next text evaluation (IFBench, AIME 2025, AIME 2026, GPQA-Diamond) on Windows:
starts the installed `qwen3.8-flash-next` profile (app\Start-NInfer.ps1), runs or resumes the suite of
eval/configs/qwen3_8_flash_next_reasoning.yaml, and stops the server at the end.

  pwsh -File eval\run_qwen3_8_flash_next_reasoning.ps1 -Plan             # show the work, no requests
  pwsh -File eval\run_qwen3_8_flash_next_reasoning.ps1                   # a new run
  pwsh -File eval\run_qwen3_8_flash_next_reasoning.ps1 -Resume <run dir> # continue an interrupted run
  pwsh -File eval\run_qwen3_8_flash_next_reasoning.ps1 -Suite smoke      # a few samples of each job first

The server must not be running already (it refuses to share the GPU with a second engine).
#>
param([switch]$Plan, [string]$Resume, [string]$Suite = 'text_full')
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$app = Join-Path $repo 'app'
$python = Join-Path $repo 'eval\.venv\Scripts\python.exe'
$config = Join-Path $repo 'eval\configs\qwen3_8_flash_next_reasoning.yaml'
$profile = 'qwen3.8-flash-next'
$env:PYTHONPATH = Join-Path $repo 'eval'
Set-Location $repo

if ($Plan) {
    & $python -m ninfer_eval plan --config $config --suite $Suite --check-runtime
    exit $LASTEXITCODE
}

$endpoint = (Get-Content -Raw (Join-Path $app 'ninfer-models.json') | ConvertFrom-Json).endpoint
if (Get-NetTCPConnection -State Listen -LocalPort ([Uri]$endpoint).Port -ErrorAction SilentlyContinue) {
    throw "A server is already listening on $endpoint. Stop it first (app\Stop-NInfer.ps1)."
}
& (Join-Path $app 'Start-NInfer.ps1') -Name $profile | Out-Null
try {
    $deadline = (Get-Date).AddMinutes(15)
    while ($true) {
        try { if ((Invoke-RestMethod "$endpoint/v1/models" -TimeoutSec 5).data.id -contains $profile) { break } } catch { }
        if ((Get-Date) -gt $deadline) { throw "$profile did not become ready within 15 minutes" }
        Start-Sleep -Seconds 5
    }
    if ($Resume) { & $python -m ninfer_eval resume --run $Resume }
    else { & $python -m ninfer_eval run --config $config --suite $Suite }
    $code = $LASTEXITCODE
} finally {
    & (Join-Path $app 'Stop-NInfer.ps1') | Out-Null
}
exit $code
