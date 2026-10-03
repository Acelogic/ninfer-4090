# Stops the server recorded by Start-NInfer.ps1, after checking the recorded pid is still that process.
$ErrorActionPreference = 'Stop'
$stateFile = Join-Path $PSScriptRoot 'server-process.json'
if (-not (Test-Path -LiteralPath $stateFile)) { Write-Output 'No saved NInfer server process.'; return }
$state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
$process = Get-Process -Id $state.pid -ErrorAction SilentlyContinue
if (-not $process) { Write-Output 'NInfer server is already stopped.'; return }
if ($process.Path -ne $state.executable -or [Math]::Abs(($process.StartTime - [datetime]$state.started).TotalSeconds) -gt 1) { throw 'Saved PID identity changed; refusing to stop it.' }
Stop-Process -Id $process.Id
if (-not $process.WaitForExit(15000)) { throw 'NInfer server did not exit.' }
Write-Output "NInfer server stopped ($($state.profile))."
