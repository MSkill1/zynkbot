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

where cargo >nul 2>&1
if %errorLevel% neq 0 (
    echo [INFO] Installing Rust...
    echo.
    echo [DEBUG] Testing network connectivity to rustup.rs...
    ping -n 1 win.rustup.rs >nul 2>&1
    if %errorLevel% equ 0 (
        echo [DEBUG] Network connectivity: OK
    ) else (
        echo [DEBUG] Network connectivity: FAILED - Cannot reach win.rustup.rs
    )
    echo.

    echo [DEBUG] Attempting download via PowerShell...
    echo [DEBUG] Target: %~dp0rustup-init.exe
    powershell -Command "try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; [System.Net.ServicePointManager]::CheckCertificateRevocationList = $false; Write-Host '[DEBUG] PowerShell: Starting download...'; Invoke-WebRequest -Uri 'https://win.rustup.rs' -OutFile '%~dp0rustup-init.exe' -UseBasicParsing; Write-Host '[DEBUG] PowerShell: Download complete'; exit 0 } catch { Write-Host '[DEBUG] PowerShell ERROR:' $_.Exception.Message -ForegroundColor Red; exit 1 }"
    set PS_EXIT=%errorLevel%
    echo [DEBUG] PowerShell exit code: %PS_EXIT%

    REM If PowerShell failed, try curl as fallback with relaxed SSL
    if not exist "%~dp0rustup-init.exe" (
        echo.
        echo [DEBUG] PowerShell download failed, trying curl...
        echo [DEBUG] Using --ssl-no-revoke flag to bypass certificate checks
        curl --ssl-no-revoke -v -o "%~dp0rustup-init.exe" https://win.rustup.rs
        set CURL_EXIT=%errorLevel%
        echo [DEBUG] curl exit code: !CURL_EXIT!
    )

    echo.
    echo [DEBUG] Checking if file exists: %~dp0rustup-init.exe
    if not exist "%~dp0rustup-init.exe" (
        echo [DEBUG] File does NOT exist
        echo.
        echo ========================================
        echo DOWNLOAD FAILED - Diagnostic Information
        echo ========================================
        echo PowerShell exit code: %PS_EXIT%
        echo curl exit code: !CURL_EXIT!
        echo.
        echo Possible causes:
        echo  1. Windows Defender quarantining the file after download
        echo  2. Antivirus blocking the download
        echo  3. Firewall blocking HTTPS connections
        echo  4. Network connectivity issues
        echo.
        echo Please try:
        echo  1. Add exclusion for: %~dp0
        echo  2. Temporarily disable Windows Defender real-time protection
        echo  3. OR manually download from: https://rustup.rs
        echo.
        pause
        exit /b 1
    )

    REM File exists - success!
    for %%A in ("%~dp0rustup-init.exe") do echo [DEBUG] File size: %%~zA bytes
    echo [DEBUG] Download successful!
    echo [OK] Download complete
    echo.
    echo Running Rust installer with default options...
    echo (This may take a few minutes...)
    "%~dp0rustup-init.exe" -y

    if %errorLevel% neq 0 (
        echo [ERROR] Rust installer failed
        echo.
        echo This is usually caused by:
        echo   1. Antivirus quarantining rustup-init.exe
        echo   2. Network/firewall blocking downloads
        echo.
        echo Please:
        echo   1. Check your antivirus quarantine and restore rustup-init.exe
        echo   2. Disable antivirus temporarily
        echo   3. Run install.bat again
        echo.
        pause
        exit /b 1
    )

    del "%~dp0rustup-init.exe" 2>nul

    REM Add Rust to PATH for this session (rustup adds it permanently, but we need it now)
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

    REM Verify Rust was actually installed by checking for cargo
    if not exist "%USERPROFILE%\.cargo\bin\cargo.exe" (
        echo [ERROR] Rust installation failed - cargo.exe not found
        echo.
        echo This means the Rust installer ran but didn't install files.
        echo Usually caused by antivirus blocking/deleting files.
        echo.
        echo Please:
        echo   1. Check antivirus logs and quarantine
        echo   2. Disable antivirus completely
        echo   3. Run install.bat again
        echo.
        pause
        exit /b 1
    )

    echo [OK] Rust installed successfully
    echo [OK] Cargo found at: %USERPROFILE%\.cargo\bin\cargo.exe

    REM Add Rust to SYSTEM PATH permanently so admin sessions can find it
    echo [INFO] Adding Rust to system PATH...
    powershell -ExecutionPolicy Bypass -File "%~dp0add_rust_to_system_path.ps1"
    echo [OK] Rust added to system PATH
) else (
    for /f "tokens=*" %%v in ('rustc --version') do set RUST_VERSION=%%v
    echo [OK] Rust already installed: !RUST_VERSION!
    REM Make sure PATH is set even if Rust was already installed
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)
echo.

REM Verify Rust is accessible in current session
cargo --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Cargo command not working even though files exist
    echo.
    echo This means PATH wasn't updated properly.
    echo Attempting to continue with explicit path...
    echo.
    REM Set explicit path for remaining commands
    set "CARGO=%USERPROFILE%\.cargo\bin\cargo.exe"
) else (
    echo [OK] Rust/Cargo is accessible and working
    set "CARGO=cargo"
)
echo.

REM ============================================
REM Step 4: Detect GPU and Configure CUDA
REM ============================================
echo =========================================
echo Step 4: Detecting GPU Hardware
echo =========================================
echo.

where nvidia-smi >nul 2>&1
if %errorLevel% equ 0 (
    echo [INFO] NVIDIA GPU detected:
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    where nvcc >nul 2>&1
    if !errorLevel! equ 0 (
        echo [OK] CUDA toolkit found - GPU acceleration will be enabled by START_ZYNKBOT.bat
        nvcc --version | findstr "release"

        REM Copy CUDA MSBuild integration files for Visual Studio Build Tools
        REM (Full VS IDE gets these automatically, Build Tools needs manual copy)
        set "CUDA_INTEGRATION_SRC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\extras\visual_studio_integration\MSBuildExtensions"
        set "VS_BUILDTOOLS_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations"

        if exist "!CUDA_INTEGRATION_SRC!" (
            if exist "!VS_BUILDTOOLS_DIR!" (
                if not exist "!VS_BUILDTOOLS_DIR!\CUDA 12.6.props" (
                    echo [INFO] Installing CUDA MSBuild integration for Build Tools...
                    copy /Y "!CUDA_INTEGRATION_SRC!\*" "!VS_BUILDTOOLS_DIR!\" >nul 2>&1
                    if !errorLevel! equ 0 (
                        echo [OK] CUDA integration installed for CMake compatibility
                    )
                )
            )
        )
    ) else (
        echo [WARNING] NVIDIA GPU found but CUDA toolkit ^(nvcc^) is not installed.
        echo           Building for CPU. Install the CUDA Toolkit and re-run install.bat
        echo           to enable GPU: https://developer.nvidia.com/cuda-downloads
    )
) else (
    echo [INFO] No NVIDIA GPU detected - building for CPU mode.
)
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

set "PRECOMPILE_FEATURES="
where nvcc >nul 2>&1
if !errorLevel! equ 0 (
    where nvidia-smi >nul 2>&1
    if !errorLevel! equ 0 (
        set "PRECOMPILE_FEATURES=--features cuda"
        echo [INFO] CUDA detected - compiling with GPU acceleration
    )
)

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
