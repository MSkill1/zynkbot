@echo off
setlocal enabledelayedexpansion
REM =============================================================================
REM ZYNKBOT DATA UNINSTALL — Windows
REM =============================================================================
REM Removes Zynkbot's database, configuration, downloaded models, and build
REM artifacts. Leaves Rust, Node.js, and all system packages in place —
REM this machine will still be a working dev environment after running.
REM
REM Use this to wipe your Zynkbot data and start fresh without reinstalling
REM system dependencies.
REM =============================================================================

title Zynkbot Data Uninstall
color 0C

echo =============================================
echo    Zynkbot Data Uninstall -- Windows
echo =============================================
echo.
echo This will remove:
echo   - Zynkbot SQLite database and app data
echo   - Environment configuration (.env)
echo   - Downloaded system models
echo   - node_modules and frontend build
echo   - Rust build cache (src-tauri\target)
echo.
echo This will NOT remove:
echo   - Rust toolchain
echo   - Node.js / npm
echo   - cmake, LLVM, or other system packages
echo   - Visual Studio Build Tools
echo   - The project folder
echo.
set /p CONFIRM="Continue? (y/n): "
if /i not "!CONFIRM!"=="y" exit /b 0
echo.

echo =============================================
echo Step 1: Stopping Zynkbot processes
echo =============================================
taskkill /F /IM app.exe 2>nul
taskkill /F /IM zynkbot_rust.exe 2>nul
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :3000 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr :57963 ^| findstr LISTENING') do taskkill /F /PID %%a 2>nul
echo [OK] Processes stopped

echo.
echo =============================================
echo Step 2: Removing SQLite database and app data
echo =============================================
set APPDATA_DIR=%LOCALAPPDATA%\zynkbot
if exist "!APPDATA_DIR!" (
    rmdir /s /q "!APPDATA_DIR!"
    echo [OK] Removed app data directory
) else (
    echo [--] No app data directory found
)

echo.
echo =============================================
echo Step 3: Removing configuration
echo =============================================
set ENV_FILE=%~dp0zynkbot_rust\src-tauri\.env
if exist "!ENV_FILE!" (
    del "!ENV_FILE!"
    echo [OK] Removed .env
) else (
    echo [--] No .env found
)

echo.
echo =============================================
echo Step 4: Removing downloaded models
echo =============================================
if exist "%~dp0zynkbot_rust\src-tauri\models" (
    rmdir /s /q "%~dp0zynkbot_rust\src-tauri\models"
    echo [OK] Removed models directory
) else (
    echo [--] No models directory found
)

echo.
echo =============================================
echo Step 5: Removing build artifacts
echo =============================================
if exist "%~dp0zynkbot_rust\node_modules" (
    rmdir /s /q "%~dp0zynkbot_rust\node_modules"
    echo [OK] Removed node_modules
)
if exist "%~dp0zynkbot_rust\build" (
    rmdir /s /q "%~dp0zynkbot_rust\build"
    echo [OK] Removed frontend build
)
if exist "%~dp0zynkbot_rust\src-tauri\target" (
    echo Removing Rust build cache (this may take a moment)...
    rmdir /s /q "%~dp0zynkbot_rust\src-tauri\target"
    echo [OK] Removed Rust target directory
)

echo.
echo =============================================
echo    Zynkbot Data Uninstall Complete
echo =============================================
echo.
echo To reinstall, run:  install.bat
echo.
pause
