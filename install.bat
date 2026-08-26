@echo off
REM Zynkbot One-Click Installation Script for Windows
REM Tested on: Windows 10/11
REM Usage: Run as Administrator (right-click -> Run as administrator)

setlocal enabledelayedexpansion

REM Add Rust to PATH if it exists (admin sessions do not inherit user PATH)
if exist "%USERPROFILE%\.cargo\bin" (
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)
title Zynkbot Installation
color 0B

echo =========================================
echo    Zynkbot Automated Installation
echo =========================================
echo.
echo *** IMPORTANT: ANTIVIRUS WARNING ***
echo.
echo Before proceeding, please DISABLE or add exceptions for:
echo   - Norton, McAfee, Windows Defender, or other antivirus
echo   - Windows Firewall
echo.
echo Why? Antivirus may block or quarantine:
echo   - Rust installer (rustup-init.exe)
echo   - Development tools (LLVM, CMake)
echo.
echo After installation completes, you can re-enable antivirus.
echo.
echo =========================================
echo.
set /p ANTIVIRUS_CONFIRM="Have you disabled antivirus? (y/n): "
if /i not "!ANTIVIRUS_CONFIRM!"=="y" (
    echo.
    echo [WARNING] Installation may fail if antivirus is blocking downloads.
    echo Press Ctrl+C to exit, or any key to continue anyway...
    pause >nul
)
echo.
echo This script will:
echo   1. Check and install dependencies
echo   2. Install Rust toolchain
echo   3. Detect GPU hardware and configure CUDA
echo   4. Configure environment
echo   5. Install Node dependencies
echo   6. Create model directories
echo   7. Download system models (embeddings, safety, entity extraction)
echo   8. Download user LLM models (optional)
echo.
echo Note: No database server required. Zynkbot uses an embedded SQLite
echo       database created automatically on first launch.
echo.
echo Starting installation in 3 seconds...
timeout /t 3 /nobreak >nul
echo.

REM ============================================
REM Step 1: Check Admin Rights
REM ============================================
echo =========================================
echo Step 1: Checking Administrator Rights
echo =========================================
echo.

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] This script requires Administrator privileges!
    echo.
    echo Please:
    echo   1. Right-click install.bat
    echo   2. Select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo [OK] Running with Administrator rights
echo.

REM ============================================
REM Change to script directory (fixes admin mode issue)
REM ============================================
cd /d "%~dp0"

REM ============================================
REM Add Windows Defender Exclusion
REM ============================================
echo Adding Windows Defender exclusion for project folder...
powershell -Command "Add-MpPreference -ExclusionPath '%~dp0'" 2>nul
if %errorLevel% equ 0 (
    echo [OK] Windows Defender exclusion added
) else (
    echo [INFO] Could not add Defender exclusion ^(may already exist or Defender not active^)
)
echo.

REM ============================================
REM Step 2: Check Dependencies
REM ============================================
echo =========================================
echo Step 2: Checking Dependencies
echo =========================================
echo.

REM Check if Chocolatey is installed
where choco >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Chocolatey not found. Installing Chocolatey...
    echo.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"

    REM Refresh environment
    call refreshenv.cmd >nul 2>&1 || (
        echo [WARNING] Please close and reopen this window, then run install.bat again
        pause
        exit /b 1
    )
    echo [OK] Chocolatey installed
) else (
    echo [OK] Chocolatey already installed
)
echo.

REM Check Node.js
echo Checking Node.js...
where node >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Installing Node.js...
    choco install -y nodejs
    call refreshenv.cmd
    echo [OK] Node.js installed
) else (
    for /f "tokens=*" %%v in ('node --version') do set NODE_VERSION=%%v
    echo [OK] Node.js already installed: !NODE_VERSION!
)
echo.

REM Check npm
where npm >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] npm not found even after Node.js installation
    exit /b 1
)
echo [OK] npm available
echo.

REM Check Git
echo Checking Git...
where git >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Installing Git...
    choco install -y git
    call refreshenv.cmd
    echo [OK] Git installed
) else (
    for /f "tokens=*" %%v in ('git --version') do set GIT_VERSION=%%v
    echo [OK] !GIT_VERSION!
)
echo.

REM Check wget (for downloading models)
where wget >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Installing wget...
    choco install -y wget
    call refreshenv.cmd
    echo [OK] wget installed
) else (
    echo [OK] wget available
)
echo.

REM Check LLVM/Clang (required for Rust bindgen)
echo Checking LLVM/Clang...
where clang.exe >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing LLVM ^(required for Rust compilation^)...
    choco install -y llvm
    call refreshenv.cmd >nul 2>&1
    echo [OK] LLVM installed
) else (
    echo [OK] LLVM already installed
)
echo.

REM Check CMake (required for building native dependencies)
echo Checking CMake...
where cmake.exe >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing CMake ^(required for native builds^)...
    choco install -y cmake
    call refreshenv.cmd >nul 2>&1
    echo [OK] CMake installed
) else (
    echo [OK] CMake already installed
)
echo.

REM ============================================
REM Visual Studio Build Tools (C++) - manual PREREQUISITE
REM Required to compile llama.cpp from source for local models. We do NOT
REM auto-install it: it is multi-GB and needs a reboot to finalize, which breaks
REM automation (an incomplete install reports no usable instance and CMake then
REM fails with "could not find any instance of Visual Studio"). Instead we detect
REM a usable (complete) C++ instance via vswhere and stop with instructions.
REM ============================================
echo Checking Visual Studio Build Tools (C++ workload)...
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VS_CPP="
if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%i in (`"%VSWHERE%" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_CPP=%%i"
)
if not defined VS_CPP (
    echo.
    echo [ERROR] Visual Studio Build Tools with "Desktop development with C++" was
    echo         not found, or an existing install is incomplete ^(e.g. pending a reboot^).
    echo.
    echo This is the one dependency we cannot auto-install, because it requires a
    echo reboot. Zynkbot compiles llama.cpp from source for local models, which
    echo needs the Microsoft C++ compiler. Please:
    echo.
    echo   1. Download:  https://aka.ms/vs/17/release/vs_BuildTools.exe
    echo   2. In the installer, check "Desktop development with C++"
    echo   3. Install, then REBOOT if prompted
    echo   4. Re-run install.bat - it handles everything else automatically
    echo.
    pause
    exit /b 1
)
echo [OK] Visual Studio Build Tools ^(C++^): %VS_CPP%
echo.

REM ============================================
REM Step 3: Install Rust
REM ============================================
echo =========================================
echo Step 3: Installing Rust Toolchain
echo =========================================
echo.

REM Rust counts as installed only if cargo actually runs. Written with goto labels
REM rather than one large if/else block on purpose: cmd.exe parses a parenthesised
REM block in a single pass and expands every %VAR% inside it up front, so an
REM %errorLevel% read within such a block reports the value from before the block
REM started. "if errorlevel N" is evaluated at the moment it runs, so it is immune
REM to that, and goto keeps each outcome on its own straight path.

REM An elevated shell does not inherit the user PATH, so also look in rustup's home.
if exist "%USERPROFILE%\.cargo\bin" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

cargo --version >nul 2>&1
if not errorlevel 1 goto rust_ready

echo [INFO] Rust not found - installing...
echo.
powershell -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://win.rustup.rs' -OutFile '%~dp0rustup-init.exe' -UseBasicParsing"
if not exist "%~dp0rustup-init.exe" curl -sS --ssl-no-revoke -o "%~dp0rustup-init.exe" https://win.rustup.rs
if not exist "%~dp0rustup-init.exe" goto rust_download_failed

echo [OK] Download complete. Running the Rust installer, this takes a few minutes...
"%~dp0rustup-init.exe" -y
del "%~dp0rustup-init.exe" >nul 2>&1

set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
cargo --version >nul 2>&1
if errorlevel 1 goto rust_install_failed

REM rustup edits the user PATH only; an elevated session needs it in the system PATH.
echo [INFO] Adding Rust to system PATH...
powershell -ExecutionPolicy Bypass -File "%~dp0add_rust_to_system_path.ps1"
goto rust_ready

:rust_download_failed
echo [ERROR] Could not download rustup-init.exe
echo.
echo Antivirus or a firewall is almost always the cause. Either:
echo   1. Add an exclusion for %~dp0 and re-run install.bat, or
echo   2. Install Rust yourself from https://rustup.rs and re-run install.bat
echo.
pause
exit /b 1

:rust_install_failed
echo [ERROR] The Rust installer ran but cargo still will not start.
echo.
echo Check your antivirus quarantine for files under
echo %USERPROFILE%\.cargo, then re-run install.bat.
echo.
pause
exit /b 1

:rust_ready
for /f "tokens=*" %%v in ('rustc --version') do set "RUST_VERSION=%%v"
echo [OK] Rust ready: %RUST_VERSION%
echo.
echo.

REM ============================================
REM Step 4: Detect GPU and Configure CUDA
REM ============================================
echo =========================================
echo Step 4: Detecting GPU Hardware
echo =========================================
echo.

REM CUDA_AVAILABLE is the single answer to "can we build GPU code?", and the
REM pre-compile step below is the only consumer. nvcc is the test that matters: a
REM machine can have a driver (nvidia-smi) with no toolkit, and without the toolkit
REM there is nothing to compile CUDA code with, so that case belongs on CPU.
set "CUDA_AVAILABLE=0"

where nvidia-smi >nul 2>&1
if errorlevel 1 goto cuda_none

echo [INFO] NVIDIA GPU detected:
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

where nvcc >nul 2>&1
if errorlevel 1 goto cuda_no_toolkit

set "CUDA_AVAILABLE=1"
echo [OK] CUDA toolkit found - GPU acceleration will be enabled
nvcc --version | findstr "release"

REM Build Tools does not ship the CUDA MSBuild integration that the full VS IDE
REM gets, and CMake needs it. Both sides are discovered rather than hardcoded: the
REM NVIDIA installer sets CUDA_PATH, VS_CPP came from vswhere in step 2, and the
REM toolset directory is globbed so this is not pinned to one CUDA or VS version.
if not defined CUDA_PATH goto cuda_ready
if not defined VS_CPP goto cuda_ready
set "CUDA_MSBUILD_SRC=%CUDA_PATH%\extras\visual_studio_integration\MSBuildExtensions"
if not exist "%CUDA_MSBUILD_SRC%" goto cuda_ready
set "VS_BUILDCUSTOM="
for /d %%d in ("%VS_CPP%\MSBuild\Microsoft\VC\v*") do set "VS_BUILDCUSTOM=%%d\BuildCustomizations"
if not defined VS_BUILDCUSTOM goto cuda_ready
if not exist "%VS_BUILDCUSTOM%" goto cuda_ready
echo [INFO] Installing CUDA MSBuild integration for Build Tools...
copy /Y "%CUDA_MSBUILD_SRC%\*" "%VS_BUILDCUSTOM%\" >nul 2>&1
if errorlevel 1 echo [WARNING] Could not copy CUDA MSBuild integration - CMake may not find CUDA.
if not errorlevel 1 echo [OK] CUDA integration installed for CMake compatibility
goto cuda_ready

:cuda_no_toolkit
echo [WARNING] NVIDIA GPU found but the CUDA toolkit ^(nvcc^) is not installed.
echo           Building for CPU. Install the CUDA Toolkit and re-run install.bat
echo           to enable GPU: https://developer.nvidia.com/cuda-downloads
goto cuda_ready

:cuda_none
echo [INFO] No NVIDIA GPU detected - building for CPU mode.

:cuda_ready
echo.

REM ============================================
REM Step 5: Configure Environment
REM ============================================
echo =========================================
echo Step 5: Configuring Environment
echo =========================================
echo.

set ENV_FILE=zynkbot_rust\src-tauri\.env

if exist "%ENV_FILE%" (
    echo [WARNING] .env file already exists, backing up...
    copy "%ENV_FILE%" "%ENV_FILE%.backup.%date:~-4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
)

echo Creating .env file...
(
    echo # LLM Backend
    echo ZYNK_MODEL_BACKEND=local
    echo # LOCAL_MODEL_PATH is not needed - models are auto-discovered from models/user/
    echo.
    echo # API Keys (add via UI later^)
    echo OPENAI_API_KEY=
    echo ANTHROPIC_API_KEY=
    echo XAI_API_KEY=
    echo.
    echo # Safety
    echo ZYNK_CONTAINMENT_MODE=guardian
    echo.
    echo # ZynkSync
    echo ZYNKSYNC_AUTO_SYNC=true
    echo ZYNKSYNC_SYNC_INTERVAL=60
) > "%ENV_FILE%"

echo [OK] Environment configured
echo      Database: embedded SQLite - no configuration needed
echo.

REM ============================================
REM Step 6: Install Node Dependencies
REM ============================================
echo =========================================
echo Step 6: Installing Node Dependencies
echo =========================================
echo.

cd zynkbot_rust
echo Running npm install...
call npm install

REM Tell Git to ignore auto-generated package-lock.json changes (prevents merge conflicts for testers)
git update-index --skip-worktree package-lock.json 2>nul

echo [OK] Node dependencies installed
echo.

REM Return to project root before model steps
cd /d "%~dp0"

REM ============================================
REM Step 7: Create Models Directories
REM ============================================
echo =========================================
echo Step 7: Creating Models Directories
echo =========================================
echo.

set USER_MODELS_DIR=%~dp0zynkbot_rust\src-tauri\models\user
set SYSTEM_MODELS_DIR=%~dp0zynkbot_rust\src-tauri\models\system

if not exist "%USER_MODELS_DIR%" mkdir "%USER_MODELS_DIR%"
echo [OK] User models directory created: zynkbot_rust\src-tauri\models\user

if not exist "%SYSTEM_MODELS_DIR%" mkdir "%SYSTEM_MODELS_DIR%"
echo [OK] System models directory created: zynkbot_rust\src-tauri\models\system
echo.

REM ============================================
REM Step 8: Download System Models (Required)
REM ============================================
echo =========================================
echo Step 8: Download System Models (Required)
echo =========================================
echo.
echo Downloading internal models for embeddings, safety, and entity extraction...
echo.

REM Download all-MiniLM-L6-v2 (embeddings)
echo Downloading embeddings model (all-MiniLM-L6-v2^)...
set EMBED_DIR=%SYSTEM_MODELS_DIR%\all-MiniLM-L6-v2
if not exist "%EMBED_DIR%" mkdir "%EMBED_DIR%"

if not exist "%EMBED_DIR%\config.json" (
    wget -q --show-progress -O "%EMBED_DIR%\config.json" "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json"
)
if not exist "%EMBED_DIR%\tokenizer.json" (
    wget -q --show-progress -O "%EMBED_DIR%\tokenizer.json" "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
)
if not exist "%EMBED_DIR%\model.safetensors" (
    wget -q --show-progress -O "%EMBED_DIR%\model.safetensors" "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors"
)
echo [OK] Embeddings model downloaded
echo.

REM Download toxic-bert (safety)
echo Downloading safety classifier (toxic-bert^)...
set SAFETY_DIR=%SYSTEM_MODELS_DIR%\toxic-bert
if not exist "%SAFETY_DIR%" mkdir "%SAFETY_DIR%"

if not exist "%SAFETY_DIR%\config.json" (
    wget -q --show-progress -O "%SAFETY_DIR%\config.json" "https://huggingface.co/unitary/toxic-bert/resolve/main/config.json"
)
if not exist "%SAFETY_DIR%\vocab.txt" (
    wget -q --show-progress -O "%SAFETY_DIR%\vocab.txt" "https://huggingface.co/unitary/toxic-bert/resolve/main/vocab.txt"
)
if not exist "%SAFETY_DIR%\model.safetensors" (
    wget -q --show-progress -O "%SAFETY_DIR%\model.safetensors" "https://huggingface.co/unitary/toxic-bert/resolve/main/model.safetensors"
)
echo [OK] Safety classifier downloaded
echo.

REM Download bert-base-NER (entity extraction)
echo Downloading entity extraction model (BERT NER^)...
set NER_DIR=%SYSTEM_MODELS_DIR%\bert-base-NER
if not exist "%NER_DIR%" mkdir "%NER_DIR%"

if not exist "%NER_DIR%\config.json" (
    wget -q --show-progress -O "%NER_DIR%\config.json" "https://huggingface.co/dslim/bert-base-NER/resolve/main/config.json"
)
if not exist "%NER_DIR%\vocab.txt" (
    wget -q --show-progress -O "%NER_DIR%\vocab.txt" "https://huggingface.co/dslim/bert-base-NER/resolve/main/vocab.txt"
)
if not exist "%NER_DIR%\model.safetensors" (
    wget -q --show-progress -O "%NER_DIR%\model.safetensors" "https://huggingface.co/dslim/bert-base-NER/resolve/main/model.safetensors"
)
echo [OK] BERT NER model downloaded
echo.
echo [OK] All system models downloaded successfully!
echo.

REM ============================================
REM Step 9: Download User Models (Optional)
REM ============================================
echo =========================================
echo Step 9: Download User Models (Optional)
echo =========================================
echo.
echo Would you like to download local LLM models for offline inference?
echo.
echo Available models:
echo   1. Qwen3 8B (5.0GB^)                        - Best all-around; recommended for new users
echo   2. DeepSeek R1 Distill Llama 8B (4.7GB^)  - Reasoning model; analytical tasks
echo   3. Llama 3.1 8B Lexi Uncensored (4.9GB^)  - Creative, unfiltered responses
echo.
echo Enter model numbers to download (space-separated^), or press Enter to skip
echo Example: 1 2 3 (for all^), or just 1 (for Llama^)
echo.
set /p MODEL_CHOICES="Your choice: "

if not "!MODEL_CHOICES!"=="" (
    cd /d "%USER_MODELS_DIR%"

    for %%m in (!MODEL_CHOICES!) do (
        if "%%m"=="1" (
            echo.
            echo Downloading Qwen3 8B (5.0GB^)...
            wget -c "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf" -O "Qwen3-8B-Q4_K_M.gguf"
            if !errorLevel! equ 0 (
                echo [OK] Qwen3 8B downloaded
            ) else (
                echo [ERROR] Failed to download Qwen3 8B
            )
        )

        if "%%m"=="2" (
            echo.
            echo Downloading DeepSeek R1 Distill Llama 8B (4.7GB^)...
            wget -c "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf" -O "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
            if !errorLevel! equ 0 (
                echo [OK] DeepSeek R1 Distill Llama 8B downloaded
            ) else (
                echo [ERROR] Failed to download DeepSeek R1 Distill Llama 8B
            )
        )

        if "%%m"=="3" (
            echo.
            echo Downloading Llama 3.1 8B Lexi Uncensored (4.9GB^)...
            wget -c "https://huggingface.co/bartowski/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF/resolve/main/Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf" -O "Llama-3.1-8B-Lexi-Uncensored-V2-Q4_K_M.gguf"
            if !errorLevel! equ 0 (
                echo [OK] Llama 3.1 8B Lexi Uncensored downloaded
            ) else (
                echo [ERROR] Failed to download Llama 3.1 8B Lexi Uncensored
            )
        )
    )

    echo.
    echo [OK] Model downloads complete
) else (
    echo [INFO] Skipping model downloads
    echo       You can download models later - see docs/MODELS.md
)
echo.

cd /d "%~dp0"

REM ============================================
REM Create Start Menu Shortcut
REM ============================================
echo =========================================
echo Creating Start Menu Shortcut
echo =========================================
echo.

powershell -Command "$WshShell = New-Object -ComObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut([System.Environment]::GetFolderPath('Programs') + '\Zynkbot.lnk'); $Shortcut.TargetPath = '%~dp0START_ZYNKBOT.bat'; $Shortcut.WorkingDirectory = '%~dp0'; $Shortcut.IconLocation = '%~dp0zynkbot_rust\src-tauri\icons\icon.ico'; $Shortcut.Description = 'AI Assistant with Memory'; $Shortcut.Save()"

if %errorLevel% equ 0 (
    echo [OK] Start Menu shortcut created - Zynkbot now appears in your Start Menu
) else (
    echo [WARNING] Could not create Start Menu shortcut - you can still launch via START_ZYNKBOT.bat
)
echo.

REM ============================================
REM Download Vosk Windows SDK (offline dictation)
REM ============================================
echo =========================================
echo Downloading Vosk Speech Recognition Library
echo =========================================
echo.

set "VOSK_LIB_DIR=%~dp0zynkbot_rust\src-tauri\lib\vosk"
if not exist "%VOSK_LIB_DIR%" mkdir "%VOSK_LIB_DIR%"

if exist "%VOSK_LIB_DIR%\libvosk.lib" (
    echo [OK] Vosk library already present - skipping download
) else (
    echo [INFO] Downloading Vosk Windows SDK ^(~24MB^)...
    set "VOSK_VER=0.3.45"
    set "VOSK_ZIP=%TEMP%\vosk-win64.zip"
    set "VOSK_EXTRACT=%TEMP%\vosk_extract"
    set "VOSK_URL=https://github.com/alphacep/vosk-api/releases/download/v!VOSK_VER!/vosk-win64-!VOSK_VER!.zip"

    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '!VOSK_URL!' -OutFile '!VOSK_ZIP!' -UseBasicParsing"
    if !errorLevel! equ 0 (
        echo [INFO] Extracting Vosk library files...
        REM Every path here is expanded by cmd (!VAR!), never by PowerShell. A
        REM '$env:TEMP' in single quotes is a literal string, so PowerShell read
        REM $env: as a drive qualifier and failed with "Cannot find drive".
        powershell -Command "Expand-Archive -Path '!VOSK_ZIP!' -DestinationPath '!VOSK_EXTRACT!' -Force; Copy-Item '!VOSK_EXTRACT!\vosk-win64-!VOSK_VER!\libvosk.lib' -Destination '!VOSK_LIB_DIR!\libvosk.lib' -Force; Copy-Item '!VOSK_EXTRACT!\vosk-win64-!VOSK_VER!\libvosk.dll' -Destination '!VOSK_LIB_DIR!\libvosk.dll' -Force; Remove-Item '!VOSK_EXTRACT!' -Recurse -Force; Remove-Item '!VOSK_ZIP!' -Force"
        if !errorLevel! equ 0 (
            echo [OK] Vosk library installed - offline dictation enabled
        ) else (
            echo [WARNING] Vosk extraction failed - offline dictation unavailable
            echo           You can still use OpenAI Whisper for cloud dictation
        )
    ) else (
        echo [WARNING] Vosk download failed - offline dictation unavailable
        echo           You can still use OpenAI Whisper for cloud dictation
    )
)
echo.

REM ============================================
REM Pre-compile Rust Backend (one-time build)
REM ============================================
echo =========================================
echo Pre-compiling Rust Backend
echo =========================================
echo.
echo [INFO] Building Zynkbot for the first time.
echo        This takes 10-20 minutes. The build may appear frozen -- this is normal.
echo        Do NOT close this window.
echo.

REM Step 4 already decided this - reuse it instead of re-running the detection.
set "PRECOMPILE_FEATURES="
if "%CUDA_AVAILABLE%"=="1" set "PRECOMPILE_FEATURES=--features cuda"
if "%CUDA_AVAILABLE%"=="1" echo [INFO] CUDA detected - compiling with GPU acceleration
if not "%CUDA_AVAILABLE%"=="1" echo [INFO] No CUDA toolkit - compiling for CPU

cd zynkbot_rust\src-tauri
cargo build !PRECOMPILE_FEATURES!
if !errorLevel! equ 0 (
    echo.
    echo [OK] Rust backend compiled successfully
) else (
    echo.
    echo [WARNING] Build failed - see errors above.
    echo           Fix the issue and re-run install.bat, or run START_ZYNKBOT.bat
    echo           manually ^(it will compile on first launch^).
)
cd ..\..
echo.

REM ============================================
REM Installation Complete
REM ============================================
echo =========================================
echo    [OK] Installation Complete!
echo =========================================
echo.
echo Next Steps:
echo.
echo 1. Start Zynkbot:
echo    Double-click START_ZYNKBOT.bat
echo.
echo 2. Add API keys (optional, for cloud models^):
echo    Click Settings (gear icon^) -^> API Keys in the app
echo    - OpenAI, Anthropic, or xAI keys
echo    - Not required - local models work offline
echo.
echo 3. Complete onboarding:
echo    Click "Get to Know You" button to personalize your experience
echo.
echo 4. Add documents to Knowledge Base (optional^):
echo    Settings -^> Knowledge Base -^> Upload Documents
echo    - Supports: txt, md, json, code files (PDF: coming soon^)
echo    - Searchable via semantic similarity
echo.
echo =========================================
echo  Ready to use Zynkbot!
echo =========================================
echo.
pause
