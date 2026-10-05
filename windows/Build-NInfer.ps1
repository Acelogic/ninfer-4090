<#
Builds NInfer-4090 Extreme on Windows: one ninfer-serve.exe for Qwen3.8 27B, Ternary Bonsai 2 27B and
Qwen3.8-Flash-Next, plus the CLI, the perplexity tool and the Flash-Next developer tools.

  pwsh -File windows\Build-NInfer.ps1 [-Configure] [-Install] [-Test] [-Targets ninfer-serve,fn_generate] [-Jobs 16]

-Configure  (re)runs CMake; implied when build\ has no build.ninja yet.
-Install    copies the server, CLI, runtime DLLs, launcher scripts, templates and model profiles into app\
            (the folder Pi and the tray start models from). Existing profiles are backed up first.
-Test       runs ctest after building.

Toolchain: Visual Studio 2022 Build Tools, CUDA 13.3 and vcpkg in ..\build-tools (next to this checkout), with
the vcpkg packages in build-tools\vcpkg_installed-ninfer. nvcc fails silently when TEMP contains a space and
mishandles spaces in paths, so the checkout and the toolchain are reached through temporary drive letters.
#>
[CmdletBinding(PositionalBinding = $false)]
param([switch]$Configure, [switch]$Install, [switch]$Test, [string[]]$Targets, [int]$Jobs = 16,
      [string]$Tools = (Join-Path (Split-Path (Split-Path $PSScriptRoot)) 'build-tools'),
      [string]$RootDrive = 'N', [string]$ToolsDrive = 'Q')
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Split-Path $PSScriptRoot)).Path
$Tools = (Resolve-Path $Tools).Path
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
cmd /d /c "`"$vcvars`" >nul && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($Matches[1])" $Matches[2] } }

$made = @()
function Map-Drive([string]$letter, [string]$path) {
    $line = (subst) | Where-Object { $_ -like "${letter}:\*" }
    if (-not $line) { subst "${letter}:" $path; $script:made += "${letter}:"; return }
    if (-not $line.EndsWith("=> $path")) { throw "${letter}: is already mapped elsewhere: $line" }
}
try {
    Map-Drive $RootDrive $root
    $R = "${RootDrive}:"
    Map-Drive $ToolsDrive (Split-Path $Tools)
    $t = "${ToolsDrive}:/$(Split-Path $Tools -Leaf)"
    New-Item -ItemType Directory -Force "$R\build\tmp" | Out-Null
    $env:TEMP = "$R\build\tmp"; $env:TMP = $env:TEMP
    $ninja = (Get-Command ninja -ErrorAction SilentlyContinue).Source
    if (-not $ninja) { $ninja = (Get-ChildItem "$Tools" -Recurse -Filter ninja.exe -ErrorAction SilentlyContinue | Select-Object -First 1).FullName }
    if (-not $ninja) { throw 'ninja.exe not found (install it with: python -m pip install ninja)' }

    if ($Configure -or -not (Test-Path "$R\build\build.ninja")) {
        cmake -S $R/ -B $R/build -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 `
            "-DCMAKE_TOOLCHAIN_FILE=$t/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows `
            "-DVCPKG_INSTALLED_DIR=$t/vcpkg_installed-ninfer" -DVCPKG_MANIFEST_INSTALL=OFF `
            "-DCMAKE_CUDA_COMPILER=$t/cuda-13.3/bin/nvcc.exe" "-DCUDAToolkit_ROOT=$t/cuda-13.3" `
            -DNINFER_WITH_FLASHNEXT=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=OFF *> "$R\build\configure.log"
        $code = $LASTEXITCODE
        Get-Content "$R\build\configure.log" | Select-Object -Last 8
        if ($code) { throw "configure failed ($code); see build\configure.log" }
    }
    $targetArgs = if ($Targets) { @('--target') + $Targets } else { @() }
    cmake --build $R/build -j $Jobs @targetArgs *> "$R\build\build.log"
    $code = $LASTEXITCODE
    Get-Content "$R\build\build.log" | Select-String -Pattern ': error|error:|FAILED|fatal error|LNK[0-9]' | Select-Object -First 60 | ForEach-Object { $_.Line }
    if ($code) { throw "build failed ($code); see build\build.log" }
    'BUILD OK'

    if ($Test) {
        ctest --test-dir $R/build --output-on-failure -j 1 *> "$R\build\ctest.log"
        $code = $LASTEXITCODE
        Get-Content "$R\build\ctest.log" | Select-Object -Last 15
        if ($code) { throw "tests failed ($code); see build\ctest.log" }
    }

    if ($Install) {
        $app = Join-Path $root 'app'
        New-Item -ItemType Directory -Force $app, "$app\logs", "$app\templates" | Out-Null
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
        foreach ($exe in 'ninfer-serve.exe', 'ninfer.exe', 'ninfer-perplexity.exe') { Copy-Item "$R\build\apps\$exe" $app -Force }
        Copy-Item "$R\build\apps\*.dll" $app -Force
        foreach ($s in 'Start-NInfer.ps1', 'Stop-NInfer.ps1', 'NInfer-Tray.ps1', 'Start-NInferTray.ps1') { Copy-Item "$R\windows\$s" $app -Force }
        if (Test-Path "$app\ninfer-models.json") { Copy-Item "$app\ninfer-models.json" "$app\ninfer-models.json.bak_$stamp" }
        Copy-Item "$R\windows\ninfer-models.json" $app -Force
        if (-not (Test-Path "$app\templates\froggeric-v22.5")) { Copy-Item "$R\third_party\froggeric" "$app\templates\froggeric-v22.5" -Recurse }
        New-Item -ItemType Directory -Force "$app\tools" | Out-Null
        Copy-Item "$R\tools\upgrade_ninfer_v2_to_v3.py" "$app\tools" -Force
        $info = [ordered]@{ built = (Get-Date -Format s); commit = (git -C $root rev-parse --short HEAD); flashnext = $true }
        $info | ConvertTo-Json | Set-Content "$app\BUILD-INFO.json"
        "INSTALLED into $app (previous profiles: ninfer-models.json.bak_$stamp)"
    }
} finally {
    foreach ($d in $made) { subst $d /D }
}
