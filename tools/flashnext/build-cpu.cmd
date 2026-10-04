@echo off
rem Builds the CPU-only Flash-Next tools into ..\..\build-flashnext. ggml is linked only by the format check.
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "SRC=%~dp0..\.."
set "OUT=%SRC%\build-flashnext"
set "F=%USERPROFILE%\Developer\qwen-flash-next-native"
if not exist "%OUT%" mkdir "%OUT%"
set "CXX=cl /nologo /O2 /Ob3 /std:c++20 /EHsc /arch:AVX512 /W3 /I"%SRC%\src" /Fo"%OUT%\\""
set "CORE="%SRC%\src\flashnext\gguf.cpp" "%SRC%\src\flashnext\quants.cpp""
%CXX% /I"%F%\source\ggml\include" %CORE% "%SRC%\tools\flashnext\check_formats.cpp" /Fe:"%OUT%\check_formats.exe" /link /LIBPATH:"%F%\build\ggml\src" ggml-base.lib
if errorlevel 1 exit /b 1
%CXX% %CORE% "%SRC%\src\flashnext\cpu_experts.cpp" "%SRC%\tools\flashnext\test_cpu_experts.cpp" /Fe:"%OUT%\test_cpu_experts.exe"
if errorlevel 1 exit /b 1
rem FP32 reference forward pass and its greedy generator: no ggml.
%CXX% %CORE% "%SRC%\src\flashnext\reference.cpp" "%SRC%\tools\flashnext\ref_generate.cpp" /Fe:"%OUT%\ref_generate.exe"
if errorlevel 1 exit /b 1
echo BUILD OK
