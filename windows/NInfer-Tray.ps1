<#
NInfer tray icon.

Shows whether the local NInfer engine is serving a model and lets you stop it to free the GPU,
or start one of the engines listed in ninfer-tray.json. The icon is green while a model is
served, amber while one is loading, and grey when the engine is stopped.

Auto-stop: Pi sessions that use the engine leave lease files in %LOCALAPPDATA%\ninfer-tray\leases
(written by the pi-local-backend extension). When a Pi session started the engine and every Pi
session holding a lease is gone, the tray stops the engine. This covers terminals closed without
/quit, where Pi gets no chance to clean up. An engine you started yourself is never auto-stopped.

Run with: pwsh -STA -WindowStyle Hidden -File NInfer-Tray.ps1   (Start-NInferTray.ps1 does this)
#>
param([string]$Config = (Join-Path $PSScriptRoot 'ninfer-tray.json'))
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms, System.Drawing

$mutex = New-Object System.Threading.Mutex($false, 'Local\NInferTray')
if (-not $mutex.WaitOne(0)) { return } # one tray per desktop session

$configDir = Split-Path -Parent (Resolve-Path -LiteralPath $Config)
$cfg = Get-Content -Raw -LiteralPath $Config | ConvertFrom-Json
$port = ([Uri]$cfg.endpoint).Port
$stateDir = Join-Path $env:LOCALAPPDATA 'ninfer-tray'
$leaseDir = Join-Path $stateDir 'leases'
$logFile = Join-Path $stateDir 'tray.log'
New-Item -ItemType Directory -Force -Path $leaseDir | Out-Null

function Write-Log([string]$text) { Add-Content -LiteralPath $logFile -Value "[$(Get-Date -Format s)] $text" }
function Resolve-ConfigPath([string]$path) { if ([IO.Path]::IsPathRooted($path)) { $path } else { Join-Path $configDir $path } }

# 'running' with the served model ids, 'loading' while the port is up but not answering, or 'stopped'.
function Get-EngineState {
    try {
        $ids = @((Invoke-RestMethod "$($cfg.endpoint)/v1/models" -TimeoutSec 2).data | ForEach-Object id)
        return @{ state = 'running'; models = $ids }
    } catch {
        $listening = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
        return @{ state = $(if ($listening) { 'loading' } else { 'stopped' }); models = @() }
    }
}

function Get-VramText {
    try {
        $used, $total = (& nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | Select-Object -First 1) -split ',\s*'
        return '{0:N1}/{1:N0} GB' -f ([double]$used / 1024), ([double]$total / 1024)
    } catch { return '' }
}

# Leases whose Pi process is still the same process (pid reuse is checked against start time).
function Get-Leases {
    Get-ChildItem -LiteralPath $leaseDir -Filter '*.json' -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $lease = Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json
            $proc = Get-Process -Id $lease.pid -ErrorAction SilentlyContinue
            $started = [DateTimeOffset]::FromUnixTimeMilliseconds([int64]$lease.startedMs).LocalDateTime
            $alive = $proc -and $proc.ProcessName -eq 'node' -and [Math]::Abs(($proc.StartTime - $started).TotalSeconds) -lt 10
            [pscustomobject]@{ file = $_.FullName; alive = [bool]$alive; startedBackend = [bool]$lease.startedBackend }
        } catch { }
    }
}

function Clear-Leases { Get-ChildItem -LiteralPath $leaseDir -Filter '*.json' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue }

function Stop-Engine([string]$reason) {
    Write-Log "stopping engine: $reason"
    foreach ($engine in $cfg.engines) {
        # Each stop script only stops the process its own launcher recorded.
        try { & (Resolve-ConfigPath $engine.stop) | Out-Null } catch { Write-Log "stop $($engine.label): $($_.Exception.Message)" }
    }
    Clear-Leases
    $notify.ShowBalloonTip(3000, 'NInfer stopped', $reason, [System.Windows.Forms.ToolTipIcon]::Info)
}

function Start-Engine($engine) {
    if ((Get-EngineState).state -ne 'stopped') { Stop-Engine "Switching to $($engine.label)"; Start-Sleep -Seconds 2 }
    Write-Log "starting $($engine.label)"
    try { & (Resolve-ConfigPath $engine.start) | Out-Null } catch { Write-Log "start $($engine.label): $($_.Exception.Message)"; $notify.ShowBalloonTip(5000, 'NInfer start failed', $_.Exception.Message, [System.Windows.Forms.ToolTipIcon]::Error) }
}

function New-DotIcon([System.Drawing.Color]$color) {
    $bmp = New-Object System.Drawing.Bitmap 16, 16
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = 'AntiAlias'
    $g.FillEllipse((New-Object System.Drawing.SolidBrush $color), 1, 1, 14, 14)
    $g.DrawEllipse((New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(60, 60, 60)), 1.5), 1, 1, 14, 14)
    $g.Dispose()
    [System.Drawing.Icon]::FromHandle($bmp.GetHicon())
}
$icons = @{
    running = New-DotIcon ([System.Drawing.Color]::FromArgb(46, 204, 113))
    loading = New-DotIcon ([System.Drawing.Color]::FromArgb(243, 156, 18))
    stopped = New-DotIcon ([System.Drawing.Color]::FromArgb(149, 165, 166))
}

$notify = New-Object System.Windows.Forms.NotifyIcon
$notify.Icon = $icons.stopped
$notify.Text = 'NInfer'
$notify.Visible = $true

$menu = New-Object System.Windows.Forms.ContextMenuStrip
$statusItem = $menu.Items.Add('NInfer')
$statusItem.Enabled = $false
[void]$menu.Items.Add('-')
$stopItem = $menu.Items.Add('Stop engine (free GPU)')
$stopItem.add_Click({ Stop-Engine 'Stopped from the tray.' })
$startMenu = New-Object System.Windows.Forms.ToolStripMenuItem 'Start'
foreach ($engine in $cfg.engines) {
    $item = $startMenu.DropDownItems.Add($engine.label)
    $item.Tag = $engine
    $item.add_Click({ Start-Engine $this.Tag })
}
[void]$menu.Items.Add($startMenu)
$autoItem = New-Object System.Windows.Forms.ToolStripMenuItem 'Auto-stop when Pi exits'
$autoItem.CheckOnClick = $true
$autoItem.Checked = [bool]$cfg.autoStopWhenPiExits
[void]$menu.Items.Add($autoItem)
$logsItem = $menu.Items.Add('Open logs folder')
$logsItem.add_Click({ Start-Process explorer.exe (Resolve-ConfigPath $cfg.logs) })
[void]$menu.Items.Add('-')
$exitItem = $menu.Items.Add('Exit tray (engine keeps running)')
$exitItem.add_Click({ $timer.Stop(); $notify.Visible = $false; [System.Windows.Forms.Application]::Exit() })
$notify.ContextMenuStrip = $menu

$script:deadPolls = 0
$script:stoppedSince = $null
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 3000
$timer.add_Tick({
    try {
        $engine = Get-EngineState
        $label = if ($engine.models.Count) { $engine.models -join ', ' } else { '' }
        $vram = if ($engine.state -ne 'stopped') { Get-VramText } else { '' }
        $notify.Icon = $icons[$engine.state]
        $text = ("NInfer: $($engine.state) $label $vram").Trim() -replace '\s+', ' '
        $notify.Text = $text.Substring(0, [Math]::Min(63, $text.Length)) # Windows caps tooltips at 63 chars
        $statusItem.Text = $text
        $stopItem.Enabled = $engine.state -ne 'stopped'

        if ($engine.state -eq 'stopped') {
            $script:deadPolls = 0
            Clear-Leases
            if (-not $script:stoppedSince) { $script:stoppedSince = Get-Date }
            if ($cfg.exitWhenStopped -and ((Get-Date) - $script:stoppedSince).TotalSeconds -gt 15) {
                $timer.Stop(); $notify.Visible = $false; [System.Windows.Forms.Application]::Exit()
            }
            return
        }
        $script:stoppedSince = $null
        $leases = @(Get-Leases)
        $piStarted = @($leases | Where-Object startedBackend).Count -gt 0
        $anyAlive = @($leases | Where-Object alive).Count -gt 0
        # Two polls in a row (about 6 s) so a Pi session that is just restarting doesn't trigger it.
        if ($autoItem.Checked -and $piStarted -and -not $anyAlive) { $script:deadPolls++ } else { $script:deadPolls = 0 }
        if ($script:deadPolls -ge 2) { $script:deadPolls = 0; Stop-Engine 'Every Pi session using it has exited.' }
    } catch { Write-Log "tick: $($_.Exception.Message)" }
})
$timer.Start()
Write-Log "tray started (pid $PID)"
[System.Windows.Forms.Application]::Run()
$notify.Dispose()
$mutex.ReleaseMutex()
Write-Log 'tray exited'
