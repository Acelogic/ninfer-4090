<#
Starts ninfer-serve for one profile from ninfer-models.json and records the process in
server-process.json, which Stop-NInfer.ps1 and the tray icon use to stop exactly that process.

Layout (the Windows release zip): this script next to ninfer-serve.exe, model files in models\.
A profile's "file" may instead be an absolute path, and "exe" may name another ninfer-serve.exe (for
example a build with Flash-Next GGUF support); relative paths resolve against this folder. Environment
variables (%USERPROFILE%) expand in "file", "exe" and "args".
The profile name (-Name, alias -Profile) is also the model id the server reports, so clients select it by that name.
Extra arguments after -Name are appended to the profile's server arguments. A profile with
"template": "froggeric" serves froggeric's Qwen Fixed Chat Templates v22.5 instead of the artifact's template.
Starts the tray icon (Start-NInferTray.ps1) when it is present.
#>
param(
    [Parameter(Mandatory)][Alias('Profile')][string]$Name,
    [string]$Config = (Join-Path $PSScriptRoot 'ninfer-models.json'),
    [Parameter(ValueFromRemainingArguments)][string[]]$Extra
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$cfg = Get-Content -Raw -LiteralPath $Config | ConvertFrom-Json
$entry = $cfg.profiles.$Name
if (-not $entry) { throw "Unknown profile '$Name'. Profiles: $(($cfg.profiles.PSObject.Properties.Name) -join ', ')" }
$port = ([Uri]$cfg.endpoint).Port
function Resolve-ProfilePath([string]$path, [string]$base) {
    $path = [Environment]::ExpandEnvironmentVariables($path)
    if (-not [IO.Path]::IsPathRooted($path)) { $path = Join-Path $base $path }
    [IO.Path]::GetFullPath($path)  # Stop-NInfer.ps1 matches the process path exactly
}
$exe = if ($entry.exe) { Resolve-ProfilePath $entry.exe $root } else { Join-Path $root 'ninfer-serve.exe' }
$model = Resolve-ProfilePath $entry.file (Join-Path $root 'models')
# Profile arguments may carry paths: expand variables and quote any argument with spaces.
$profileArgs = @($entry.args | ForEach-Object {
    $arg = [Environment]::ExpandEnvironmentVariables([string]$_)
    if ($arg -match '\s') { '"' + $arg + '"' } else { $arg }
})
foreach ($required in $exe, $model) { if (-not (Test-Path -LiteralPath $required)) { throw "Missing: $required" } }
if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) {
    throw "Port $port is already in use. Stop the running server first (Stop-NInfer.ps1)."
}
New-Item -ItemType Directory -Force -Path (Join-Path $root 'logs') | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$log = Join-Path $root "logs\server-$Name-$stamp"
$serveArgs = @(('"' + $model + '"'), '--host', '127.0.0.1', '--port', "$port", '--model-id', $Name) +
    @($cfg.common) + $profileArgs + @($Extra | Where-Object { $_ }) +
    $(if ($entry.template) { @('--chat-template', ('"' + (Join-Path $root $cfg.templates.($entry.template)) + '"')) } else { @() }) +
    @('--request-log-jsonl', ('"' + "$log.requests.jsonl" + '"'))
$process = Start-Process -FilePath $exe -ArgumentList $serveArgs -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput "$log.stdout.log" -RedirectStandardError "$log.stderr.log" -PassThru
$state = [ordered]@{
    pid = $process.Id; executable = $exe; profile = $Name; model = $Name; modelFile = $entry.file
    endpoint = "http://127.0.0.1:$port/v1"; stderr = "$log.stderr.log"; started = $process.StartTime.ToString('o')
}
$state | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $root 'server-process.json') -Encoding utf8
# Tray icon to watch and stop the engine; a failure here must not fail the start.
$tray = Join-Path $root 'Start-NInferTray.ps1'
if (Test-Path -LiteralPath $tray) { try { & $tray } catch { Write-Warning "Tray icon not started: $($_.Exception.Message)" } }
$state | ConvertTo-Json
