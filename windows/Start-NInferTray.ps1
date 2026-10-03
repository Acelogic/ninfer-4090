# Starts the NInfer tray icon in its own hidden process, so it outlives the terminal that started
# it. Safe to call repeatedly: the tray exits at once if one is already running.
$tray = Join-Path $PSScriptRoot 'NInfer-Tray.ps1'
$pwsh = (Get-Command pwsh.exe -ErrorAction SilentlyContinue).Source
if (-not $pwsh) { $pwsh = Join-Path $env:ProgramFiles 'PowerShell\7\pwsh.exe' }
Start-Process -FilePath $pwsh -WindowStyle Hidden -ArgumentList @(
    '-NoProfile', '-STA', '-WindowStyle', 'Hidden', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $tray + '"')
)
