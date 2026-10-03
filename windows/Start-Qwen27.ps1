<#
Starts ninfer-serve for Qwen3.8 27B on 127.0.0.1 and records the process in server-process.json,
which Stop-Qwen27.ps1 and the tray icon use to stop exactly this process.

Expected layout (the Windows release zip): this script next to ninfer-serve.exe, models in
models\, the Froggeric template in templates\froggeric-v22.5\. Any groupwise-int Qwen3.8 27B
artifact works: pass -ModelFile and -ModelId to serve another one. Starts the tray icon
(Start-NInferTray.ps1) when it is present.
#>
param(
    [ValidateSet('Froggeric','Artifact')][string]$Template = 'Froggeric',
    [ValidateRange(4096,262144)][int]$Context = 262144,
    [ValidateRange(0,7)][int]$Mtp = 7,
    [ValidateSet('rk4v4-e8','rk4v4','rk2v4-e8','rk8v4','int8')][string]$KvType = 'rk4v4-e8',
    [ValidateRange(256,8192)][int]$VisionMaxTokens = 4096,
    [string]$ModelFile = 'qwen3_8_27b.ninfer',
    [string]$ModelId = 'qwen3.8-27b',
    [int]$Port = 18085,
    [switch]$TextOnly,
    [switch]$Benchmark
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$exe = Join-Path $root 'ninfer-serve.exe'
$model = Join-Path $root "models\$ModelFile"
$stateFile = Join-Path $root 'server-process.json'
$templatePath = $null
if ($Template -eq 'Froggeric') {
    $templatePath = Join-Path $root 'templates\froggeric-v22.5\chat_template.jinja'
    # The renderer is registered by hash; any other file is rejected by the server anyway.
    if ((Get-FileHash -LiteralPath $templatePath -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'e57684bae4156211a55473c5a63be976a405a37ab5be5ae0e5abf1df5349c4b2') {
        throw 'The Froggeric template file does not match the v22.5 revision this build registers.'
    }
}
foreach ($required in $exe, $model) { if (-not (Test-Path -LiteralPath $required)) { throw "Missing: $required" } }
if (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. Stop the running server first (Stop-Qwen27.ps1)."
}
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$stdout = Join-Path $root "logs\server-$stamp.stdout.log"
$stderr = Join-Path $root "logs\server-$stamp.stderr.log"
$requests = Join-Path $root "logs\server-$stamp.requests.jsonl"
$serveArgs = @(
    ('"' + $model + '"'),
    '--host','127.0.0.1','--port',"$Port",'--model-id',$ModelId,
    '--max-context',"$Context",'--kv-capacity',"$Context",'--max-concurrency','1',
    '--max-pending-requests','16','--pending-timeout-ms','600000',
    '--prefill-chunk','1024','--kv-dtype',$KvType,
    '--preserve-thinking','--reasoning-effort','medium',
    '--no-disk-cache','--request-log-jsonl',('"' + $requests + '"')
)
if ($Mtp -gt 0) { $serveArgs += @('--spec','mtp','--draft-tokens',"$Mtp",'--lm-head-draft') }
if ($templatePath) { $serveArgs += @('--chat-template',('"' + $templatePath + '"')) }
if (-not $TextOnly) { $serveArgs += @('--vision','--vision-max-tokens',"$VisionMaxTokens") }
if ($Benchmark) { $serveArgs += '--no-prefix-reuse' }
$process = Start-Process -FilePath $exe -ArgumentList $serveArgs -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$state = [ordered]@{
    pid = $process.Id; executable = $exe; model = $ModelId; modelFile = $ModelFile; context = $Context; mtp = $Mtp
    kvType = $KvType; vision = (-not [bool]$TextOnly); template = $Template; endpoint = "http://127.0.0.1:$Port/v1"
    stdout = $stdout; stderr = $stderr; requests = $requests; started = $process.StartTime.ToString('o')
}
$state | ConvertTo-Json | Set-Content -LiteralPath $stateFile -Encoding utf8
# Tray icon to watch and stop the engine; a failure here must not fail the start.
$tray = Join-Path $root 'Start-NInferTray.ps1'
if (Test-Path -LiteralPath $tray) { try { & $tray } catch { Write-Warning "Tray icon not started: $($_.Exception.Message)" } }
$state | ConvertTo-Json
