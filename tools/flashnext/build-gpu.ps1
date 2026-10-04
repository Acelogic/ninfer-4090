# Builds the GPU Flash-Next tools into build-flashnext with nvcc (sm_89) and MSVC.
# Usage: build-gpu.ps1 [target ...] [-Cuda <toolkit dir>]   (no target: all)
[CmdletBinding(PositionalBinding = $false)]
param([string]$Cuda = "$env:USERPROFILE\Developer\build-tools\cuda-13.3",
      [Parameter(ValueFromRemainingArguments)][string[]]$Only)
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
    $engine = @('R:\src\flashnext\cpu_experts.cpp', 'R:\src\flashnext\reference.cpp', 'R:\src\flashnext\engine.cpp',
                'R:\src\flashnext\cuda\gemv.cu', 'R:\src\flashnext\cuda\ops.cu', 'R:\src\flashnext\cuda\experts.cu',
                'R:\src\flashnext\cuda\gemm.cu', 'R:\src\flashnext\cuda\qsa.cu', 'cublas.lib')
    $targets = [ordered]@{
        test_gpu_gemv = @('R:\src\flashnext\cuda\gemv.cu', 'R:\tools\flashnext\test_gpu_gemv.cu')
        test_qsa      = @('R:\src\flashnext\cuda\ops.cu', 'R:\src\flashnext\cuda\experts.cu', 'R:\src\flashnext\cuda\qsa.cu',
                          'R:\src\flashnext\reference.cpp', 'R:\tools\flashnext\test_qsa.cu')
        test_gpu_experts = @('R:\src\flashnext\cuda\experts.cu', 'R:\tools\flashnext\test_gpu_experts.cu')
        fn_generate   = $engine + @('R:\tools\flashnext\fn_generate.cpp')
    }
    if ($Only) { foreach ($k in @($targets.Keys)) { if ($Only -notcontains $k) { $targets.Remove($k) } } }
    # the engine's prompt path uses cuBLAS: ship its DLLs next to the executables
    foreach ($dll in @('cublas64_13.dll', 'cublasLt64_13.dll')) {
        $src = Join-Path $Cuda "bin\x64\$dll"
        if ((Test-Path $src) -and -not (Test-Path "R:\build-flashnext\$dll")) { Copy-Item $src "R:\build-flashnext\$dll" }
    }
    foreach ($t in $targets.Keys) {
        & $nvccExe @flags @core @($targets[$t]) -o "R:\build-flashnext\$t.exe" 2>&1 | Where-Object { $_ -notmatch '^\S+\.(cu|cpp)$|cudafe1\.cpp$' }
        if ($LASTEXITCODE) { throw "build of $t failed" }
    }
    'BUILD OK'
} finally {
    foreach ($d in $made) { subst $d /D }
}
