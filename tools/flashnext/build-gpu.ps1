# Builds the GPU Flash-Next tools into build-flashnext with nvcc (sm_89) and MSVC.
param([string]$Cuda = "$env:USERPROFILE\Developer\build-tools\cuda-13.3")
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
cmd /d /c "`"$vcvars`" >nul && set" | ForEach-Object { if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($Matches[1])" $Matches[2] } }

# nvcc exits silently with code 2 when TEMP contains a space, and mishandles spaces in other paths,
# so the tree and the toolkit are reached through temporary drive letters.
$made = @()
function Map-Drive([string]$letter, [string]$path) {
    $line = (subst) | Where-Object { $_ -like "${letter}:\*" }
    if (-not $line) { subst "${letter}:" $path; $script:made += "${letter}:"; return }
    if (-not $line.EndsWith("=> $path")) { throw "${letter}: is already mapped elsewhere: $line" }
}
try {
    Map-Drive R $root
    Map-Drive S (Split-Path $Cuda)
    $nvccExe = "S:\$(Split-Path $Cuda -Leaf)\bin\nvcc.exe"
    New-Item -ItemType Directory -Force R:\build-flashnext\tmp | Out-Null
    $env:TEMP = 'R:\build-flashnext\tmp'
    $env:TMP = $env:TEMP

    $flags = @('-O3', '-std=c++20', '-arch=sm_89', '-lineinfo', '-Xcompiler=/O2,/EHsc,/arch:AVX512,/utf-8,/Zc:preprocessor', '-I', 'R:\src')
    $core = @('R:\src\flashnext\gguf.cpp', 'R:\src\flashnext\quants.cpp')
    $targets = [ordered]@{
        test_gpu_gemv = @('R:\src\flashnext\cuda\gemv.cu', 'R:\tools\flashnext\test_gpu_gemv.cu')
    }
    foreach ($t in $targets.Keys) {
        & $nvccExe @flags @core @($targets[$t]) -o "R:\build-flashnext\$t.exe" 2>&1 | Where-Object { $_ -notmatch '^\S+\.(cu|cpp)$|cudafe1\.cpp$' }
        if ($LASTEXITCODE) { throw "build of $t failed" }
    }
    'BUILD OK'
} finally {
    foreach ($d in $made) { subst $d /D }
}
