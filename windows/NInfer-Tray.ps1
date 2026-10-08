<#
NInfer tray icon.

Shows whether the local NInfer engine is serving a model and lets you stop it to free the GPU,
or start any profile listed in ninfer-models.json. The icon is green while a model is
served, amber while one is loading, and grey when the engine is stopped. While the engine is
generating (a request is being processed, per its /slots endpoint) the green dot flashes;
"flashWhenGenerating": false in ninfer-models.json keeps it solid.

Auto-stop: Pi and Hermes sessions that use the engine leave lease files in %LOCALAPPDATA%\ninfer-tray\leases
(written by Pi's pi-local-backend extension and Hermes' ninfer-backend plugin). When a session started the
engine and every session holding a lease is gone, the tray stops the engine. This covers terminals closed
without /quit, where Pi or Hermes gets no chance to clean up. An engine you started yourself is never auto-stopped.

Run with: pwsh -STA -WindowStyle Hidden -File NInfer-Tray.ps1   (Start-NInferTray.ps1 does this)
#>
param([string]$Config = (Join-Path $PSScriptRoot 'ninfer-models.json'))
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

# Whether the server Start-NInfer.ps1 recorded is still that process. A big model loads for minutes before its
# port opens, and the tray must not take that time for a stopped engine.
function Test-EngineProcess {
    try {
        $rec = Get-Content -Raw -LiteralPath (Join-Path $configDir 'server-process.json') | ConvertFrom-Json
        $p = Get-Process -Id $rec.pid -ErrorAction Stop
        return $p.Path -eq $rec.executable
    } catch { return $false }
}

# 'running' with the served model ids, 'loading' while the server process runs but does not answer yet, or 'stopped'.
function Get-EngineState {
    try {
        $ids = @((Invoke-RestMethod "$($cfg.endpoint)/v1/models" -TimeoutSec 2).data | ForEach-Object id)
        return @{ state = 'running'; models = $ids }
    } catch {
        $listening = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
        return @{ state = $(if ($listening -or (Test-EngineProcess)) { 'loading' } else { 'stopped' }); models = @() }
    }
}

function Get-VramText {
    try {
        $used, $total = (& nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | Select-Object -First 1) -split ',\s*'
        return '{0:N1}/{1:N0} GB' -f ([double]$used / 1024), ([double]$total / 1024)
    } catch { return '' }
}

# Leases whose session process is still the same process (pid reuse is checked against start time). A lease names
# its process (Hermes runs in python); one without a name is Pi's, which runs in node.
function Get-Leases {
    Get-ChildItem -LiteralPath $leaseDir -Filter '*.json' -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $lease = Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json
            $proc = Get-Process -Id $lease.pid -ErrorAction SilentlyContinue
            $started = [DateTimeOffset]::FromUnixTimeMilliseconds([int64]$lease.startedMs).LocalDateTime
            $expected = if ($lease.process) { [string]$lease.process } else { 'node' }
            $why = if (-not $proc) { 'no such process' } elseif ($proc.ProcessName -ne $expected) { "pid is now $($proc.ProcessName)" } else {
                # StartTime can be unreadable (another user's or an elevated process); then the pid and name have to do
                try { if ([Math]::Abs(($proc.StartTime - $started).TotalSeconds) -ge 10) { "pid reused (started $($proc.StartTime))" } } catch { $null }
            }
            [pscustomobject]@{ file = $_.FullName; alive = -not $why; why = $why; startedBackend = [bool]$lease.startedBackend }
        } catch { }
    }
}

function Clear-Leases { Get-ChildItem -LiteralPath $leaseDir -Filter '*.json' -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue }

# Only the leases of Pi sessions that are gone. A live session's lease must survive the moments the engine is down
# while Pi switches models or the next one loads: Pi writes it once when it starts the engine, so a lease cleared
# then is never written again, and killing that Pi afterwards left the engine loaded.
function Remove-DeadLeases {
    Get-Leases | Where-Object { -not $_.alive } | ForEach-Object {
        Write-Log "lease $(Split-Path -Leaf $_.file) removed: $($_.why)"
        Remove-Item -Force -LiteralPath $_.file -ErrorAction SilentlyContinue
    }
}

function Stop-Engine([string]$reason) {
    Write-Log "stopping engine: $reason"
    # The stop script only stops the process its launcher recorded.
    try { & (Join-Path $configDir 'Stop-NInfer.ps1') | Out-Null } catch { Write-Log "stop: $($_.Exception.Message)" }
    Clear-Leases
    $notify.ShowBalloonTip(3000, 'NInfer stopped', $reason, [System.Windows.Forms.ToolTipIcon]::Info)
}

function Start-Engine([string]$name) {
    $label = $cfg.profiles.$name.label
    if ((Get-EngineState).state -ne 'stopped') { Stop-Engine "Switching to $label"; Start-Sleep -Seconds 2 }
    Clear-Leases # an engine started from the tray is the user's, never auto-stopped
    Write-Log "starting $name"
    try { & (Join-Path $configDir 'Start-NInfer.ps1') -Name $name | Out-Null } catch { Write-Log "start $($name): $($_.Exception.Message)"; $notify.ShowBalloonTip(5000, 'NInfer start failed', $_.Exception.Message, [System.Windows.Forms.ToolTipIcon]::Error) }
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
    dimmed  = New-DotIcon ([System.Drawing.Color]::FromArgb(24, 92, 54))   # the "off" phase of the generating flash
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
foreach ($name in $cfg.profiles.PSObject.Properties.Name) {
    $item = $startMenu.DropDownItems.Add($cfg.profiles.$name.label)
    $item.Tag = $name
    $item.add_Click({ Start-Engine $this.Tag })
}
[void]$menu.Items.Add($startMenu)
$autoItem = New-Object System.Windows.Forms.ToolStripMenuItem 'Auto-stop when Pi or Hermes exits'
$autoItem.CheckOnClick = $true
$autoItem.Checked = [bool]$cfg.autoStopWhenPiExits
[void]$menu.Items.Add($autoItem)
$logsItem = $menu.Items.Add('Open logs folder')
$logsItem.add_Click({ Start-Process explorer.exe (Join-Path $configDir 'logs') })
[void]$menu.Items.Add('-')
$exitItem = $menu.Items.Add('Exit tray (engine keeps running)')
$exitItem.add_Click({ $timer.Stop(); $flash.Stop(); $notify.Visible = $false; [System.Windows.Forms.Application]::Exit() })
$notify.ContextMenuStrip = $menu

$script:deadPolls = 0
$script:stoppedSince = $null
$script:engineState = 'stopped'
$script:generating = $false
$script:flashOn = $true

function Set-TrayIcon {
    $dim = $script:engineState -eq 'running' -and $script:generating -and -not $script:flashOn
    $notify.Icon = if ($dim) { $icons.dimmed } else { $icons[$script:engineState] }
}

# Generating = any slot of the engine's /slots is processing a request. Polled about once a second with a
# non-blocking request (a busy engine must not freeze the tray menu); the dot toggles every half second.
$http = [System.Net.Http.HttpClient]::new()
$http.Timeout = [TimeSpan]::FromSeconds(3)
$script:slotsCall = $null
$script:flashTicks = 0
$flash = New-Object System.Windows.Forms.Timer
$flash.Interval = 500
$flash.add_Tick({
    try {
        $script:flashTicks++
        if ($script:slotsCall -and $script:slotsCall.IsCompleted) {
            $script:generating = $script:slotsCall.Status -eq 'RanToCompletion' -and
                [bool](@($script:slotsCall.Result | ConvertFrom-Json) | Where-Object is_processing)
            $script:slotsCall = $null
        }
        if ($script:engineState -ne 'running') { $script:generating = $false }
        elseif (-not $script:slotsCall -and $script:flashTicks % 2 -eq 0) { $script:slotsCall = $http.GetStringAsync("$($cfg.endpoint)/slots") }
        $script:flashOn = -not ($script:generating -and $script:flashOn)
        Set-TrayIcon
    } catch { $script:slotsCall = $null }
})
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 3000
$timer.add_Tick({
    try {
        $engine = Get-EngineState
        $label = if ($engine.models.Count) { $engine.models -join ', ' } else { '' }
        $vram = if ($engine.state -ne 'stopped') { Get-VramText } else { '' }
        $script:engineState = $engine.state
        Set-TrayIcon
        $busy = if ($engine.state -eq 'running' -and $script:generating) { 'generating' } else { '' }
        $text = ("NInfer: $($engine.state) $busy $label $vram").Trim() -replace '\s+', ' '
        $notify.Text = $text.Substring(0, [Math]::Min(63, $text.Length)) # Windows caps tooltips at 63 chars
        $statusItem.Text = $text
        $stopItem.Enabled = $engine.state -ne 'stopped'

        if ($engine.state -eq 'stopped') {
            $script:deadPolls = 0
            Remove-DeadLeases
            if (-not $script:stoppedSince) { $script:stoppedSince = Get-Date }
            # A live Pi session may be between two models: stay for it (the tray Pi launches after its start would
            # find this one still holding the mutex and quit, and nothing would watch the next engine).
            if (@(Get-Leases | Where-Object alive).Count) { $script:stoppedSince = Get-Date }
            if ($cfg.exitWhenStopped -and ((Get-Date) - $script:stoppedSince).TotalSeconds -gt 15) {
                Write-Log 'exiting: the engine is stopped and no session holds a lease'
                $timer.Stop(); $flash.Stop(); $notify.Visible = $false; [System.Windows.Forms.Application]::Exit()
            }
            return
        }
        $script:stoppedSince = $null
        $leases = @(Get-Leases)
        $piStarted = @($leases | Where-Object startedBackend).Count -gt 0
        $anyAlive = @($leases | Where-Object alive).Count -gt 0
        # Two polls in a row (about 6 s) so a Pi session that is just restarting doesn't trigger it.
        if ($autoItem.Checked -and $piStarted -and -not $anyAlive) { $script:deadPolls++ } else { $script:deadPolls = 0 }
        if ($script:deadPolls -ge 2) {
            $script:deadPolls = 0
            Write-Log ('leases: ' + (($leases | ForEach-Object { "$(Split-Path -Leaf $_.file) ($($_.why))" }) -join ', '))
            Stop-Engine 'Every Pi or Hermes session using it has exited.'
        }
    } catch { Write-Log "tick: $($_.Exception.Message)" }
})
$timer.Start()
if ($cfg.flashWhenGenerating -ne $false) { $flash.Start() }
Write-Log "tray started (pid $PID)"
[System.Windows.Forms.Application]::Run()
$notify.Dispose()
$mutex.ReleaseMutex()
Write-Log 'tray exited'
