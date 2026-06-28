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

    REM Set CUDA_PATH for CMake CUDA detection
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" (
        set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
        set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
        echo [OK] CUDA_PATH configured
        set "CMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"
        set "CMAKE_GENERATOR_TOOLSET=cuda=12.6"
        echo [OK] CMake CUDA configuration complete
    )
)

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
    echo [!] FIRST-TIME BUILD DETECTED
    echo     Zynkbot is compiling its Rust backend for the first time.
    echo     This takes 10-15 minutes and only happens once.
    echo.
    echo     In the 700s, compilation will appear to pause or freeze for
    echo     several minutes. This is normal -- do NOT close this window.
    echo     Let it complete. The app will open automatically when done.
    echo.
    timeout /t 5 /nobreak >nul
)

cd zynkbot_rust
call npm run tauri:dev

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
