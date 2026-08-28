@echo off
title Zynkbot - Starting...
color 0B
setlocal enabledelayedexpansion

REM Change to the directory where this script is located
cd /d "%~dp0"

echo ========================================
echo     ZYNKBOT - Privacy-First AI
echo     Pure Rust Desktop App
echo ========================================
echo.

REM ============================================================
REM Step 1: Check Project Files
REM ============================================================
echo [1/3] Checking project files...
if not exist zynkbot_rust (
    echo [ERROR] zynkbot_rust folder not found!
    echo Please ensure you're running this from the Zynkbot project root.
    pause
    exit /b 1
)
if not exist zynkbot_rust\src-tauri (
    echo [ERROR] Rust backend not found!
    echo Please ensure the Tauri project is properly set up.
    pause
    exit /b 1
)
echo [OK] Project files found
echo.

REM ============================================================
REM Step 2: Check Environment
REM ============================================================
echo [2/3] Checking environment...

REM ============================================================
REM Step 3: Clean Up Old Processes
REM ============================================================
echo [3/3] Cleaning up old processes...
REM Kill React dev server on port 3000
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :3000 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)
REM Kill Tauri app
taskkill /F /IM app.exe >nul 2>&1
taskkill /F /IM zynkbot_rust.exe >nul 2>&1
timeout /t 2 /nobreak >nul
echo [OK] Old processes cleaned up
echo.

REM ============================================================
REM Check Node.js
REM ============================================================
where node >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Node.js is not installed
    echo.
    echo Install Node.js from: https://nodejs.org/
    pause
    exit /b 1
)
echo [OK] Node.js found

REM ============================================================
REM Check Rust/Cargo
REM ============================================================
where cargo >nul 2>&1
if %errorLevel% neq 0 (
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    where cargo >nul 2>&1
    if !errorLevel! neq 0 (
        echo [ERROR] Rust is not installed
        echo.
        echo Install Rust from: https://rustup.rs
        echo After installing, restart this window and try again.
        pause
        exit /b 1
    )
)
echo [OK] Rust/Cargo found
echo.

REM ============================================================
REM Check Visual Studio Build Tools (C++) - needed to compile llama.cpp on first launch
REM ============================================================
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS_CPP="
if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%i in (`"%VSWHERE%" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_CPP=%%i"
)
if not defined VS_CPP (
    echo [WARNING] Visual Studio Build Tools ^(C++^) not detected, or install is incomplete.
    echo    If this is the FIRST launch, the Rust build will fail when compiling llama.cpp.
    echo    Install https://aka.ms/vs/17/release/vs_BuildTools.exe ^("Desktop development
    echo    with C++"^), reboot, then run this again. ^(Already-built installs can ignore this.^)
    echo.
) else (
    echo [OK] Visual Studio Build Tools found

    REM Set NVCC_CCBIN for CUDA compilation (nvcc needs to find cl.exe)
    for /f "delims=" %%i in ('dir /b /ad "%VS_CPP%\VC\Tools\MSVC" 2^>nul ^| sort /r') do (
        set "MSVC_VERSION=%%i"
        goto :found_msvc
    )
    :found_msvc
    if defined MSVC_VERSION (
        set "NVCC_CCBIN=%VS_CPP%\VC\Tools\MSVC\!MSVC_VERSION!\bin\Hostx64\x64"
        echo [OK] CUDA compiler configured

)

REM ============================================================
REM CUDA toolkit path (bindgen_cuda / CMake)
REM ============================================================
REM bindgen_cuda searches only the PARENT install directory
REM ("C:/Program Files/NVIDIA GPU Computing Toolkit") and never the versioned
REM subdirectory the Windows CUDA installer actually creates, so it panics with
REM "Could not find CUDA in standard locations" unless CUDA_PATH points at the
REM full versioned path. Derive that from wherever nvcc really is rather than
REM hardcoding a version: RTX 50-series (Blackwell, sm_120) needs CUDA 12.8+, so
REM a pinned v12.6 silently left newer cards building on CPU.
if defined CUDA_PATH goto cuda_path_ready
for /f "delims=" %%i in ('where nvcc 2^>nul') do set "NVCC_EXE=%%i"
if not defined NVCC_EXE goto cuda_path_ready
for %%i in ("%NVCC_EXE%") do set "NVCC_BIN_DIR=%%~dpi"
set "NVCC_BIN_DIR=%NVCC_BIN_DIR:~0,-1%"
for %%i in ("%NVCC_BIN_DIR%") do set "CUDA_PATH=%%~dpi"
set "CUDA_PATH=%CUDA_PATH:~0,-1%"
:cuda_path_ready
if not defined CUDA_PATH goto cuda_env_done
set "CUDA_HOME=%CUDA_PATH%"
REM bindgen_cuda also honours CUDA_ROOT, so set both and either lookup succeeds.
set "CUDA_ROOT=%CUDA_PATH%"
echo [OK] CUDA_PATH configured: %CUDA_PATH%
:cuda_env_done

REM ============================================================
REM Check .env file
REM ============================================================
if not exist "zynkbot_rust\src-tauri\.env" (
    echo [WARNING] .env file not found
    echo    API features (Anthropic/OpenAI/xAI^) will not work
    echo    Local offline features will still work
    echo    Run install.bat to create .env, or create it manually
    echo.
)

REM ============================================================
REM Create models directory if missing
REM ============================================================
if not exist "zynkbot_rust\src-tauri\models\user" (
    echo [INFO] Creating models directory...
    mkdir "zynkbot_rust\src-tauri\models\user"
    echo [OK] Created: zynkbot_rust\src-tauri\models\user
    echo      You can place GGUF models here for local inference
    echo.
)

REM ============================================================
REM Install npm dependencies if missing
REM ============================================================
if not exist "zynkbot_rust\node_modules" (
    echo [INFO] Installing npm dependencies...
    cd zynkbot_rust
    call npm install
    cd ..
    echo [OK] npm dependencies installed
    echo.
)

REM ============================================================
REM Start Rust Desktop App
REM ============================================================
echo ========================================
echo   Zynkbot is ready!
echo   Database: SQLite (embedded)
echo   Backend: Pure Rust (Candle^)
echo.
echo   Close this window to stop Zynkbot
echo ========================================
echo.
if not exist "zynkbot_rust\src-tauri\target\debug\app.exe" (
    echo [WARNING] Binary not found -- Rust backend has not been compiled yet.
    echo           Run install.bat first. Starting anyway in 10 seconds...
    timeout /t 10 /nobreak >nul
)

REM Detect CUDA and set features flag
set "FEATURES_FLAG="
where nvcc >nul 2>&1
if !errorLevel! equ 0 (
    where nvidia-smi >nul 2>&1
    if !errorLevel! equ 0 (
        set "FEATURES_FLAG=--features cuda"
        echo [OK] CUDA detected - building with GPU acceleration
    )
)

REM Add Vosk DLL to PATH for offline dictation
if exist "%~dp0zynkbot_rust\src-tauri\lib\vosk\libvosk.dll" (
    set "PATH=%~dp0zynkbot_rust\src-tauri\lib\vosk;!PATH!"
)

cd zynkbot_rust
call npm run tauri -- dev !FEATURES_FLAG!

REM ============================================================
REM Cleanup on Exit
REM ============================================================
cd ..
echo.
echo [INFO] Shutting down Zynkbot...
REM Kill React dev server on port 3000
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :3000 ^| findstr LISTENING') do (
    echo [INFO] Killing Node.js dev server (PID %%a^)
    taskkill /F /PID %%a >nul 2>&1
)
REM Kill Tauri app
taskkill /F /IM app.exe >nul 2>&1
taskkill /F /IM zynkbot_rust.exe >nul 2>&1
echo [OK] Zynkbot stopped
pause
