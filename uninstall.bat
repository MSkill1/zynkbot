@echo off
REM Zynkbot Uninstall Script for Windows
REM Removes Zynkbot and optionally clears your memory database and Rust toolchain.

setlocal EnableDelayedExpansion

echo =========================================
echo    Zynkbot Uninstall
echo =========================================
echo.
echo This script will:
echo   - Stop any running Zynkbot processes
echo   - Remove the Start Menu shortcut
echo   - Optionally remove your memory database
echo   - Optionally remove the Rust toolchain
echo   - Optionally remove the Zynkbot project folder
echo.
echo System packages (VS Build Tools, Node.js) are NOT removed.
echo They may be used by other applications.
echo.
set /p CONFIRM="Continue? [y/N]: "
if /i not "%CONFIRM%"=="y" (
    echo Uninstall cancelled.
    pause
    exit /b 0
)
echo.

REM ============================================
REM Stop running processes
REM ============================================
echo Stopping any running Zynkbot processes...
taskkill /f /im "zynkbot.exe" >nul 2>&1 || true
taskkill /f /im "app.exe" >nul 2>&1 || true
timeout /t 1 /nobreak >nul
echo Done.
echo.

REM ============================================
REM Remove Start Menu shortcut
REM ============================================
set SHORTCUT="%APPDATA%\Microsoft\Windows\Start Menu\Programs\Zynkbot.lnk"
if exist %SHORTCUT% (
    del /f %SHORTCUT%
    echo Removed Start Menu shortcut.
) else (
    echo No Start Menu shortcut found.
)
echo.

REM ============================================
REM Memory database
REM ============================================
set DB_DIR=%LOCALAPPDATA%\zynkbot
if exist "%DB_DIR%" (
    echo Your memory database is stored at: %DB_DIR%
    echo This contains all memories Zynkbot has learned about you.
    echo.
    set /p DEL_DB="Delete your memory database? This cannot be undone. [y/N]: "
    if /i "!DEL_DB!"=="y" (
        rmdir /s /q "%DB_DIR%"
        echo Memory database deleted.
    ) else (
        echo Memory database kept at: %DB_DIR%
        echo You can delete it manually at any time.
    )
) else (
    echo No memory database found.
)
echo.

REM ============================================
REM Rust toolchain
REM ============================================
where rustup >nul 2>&1
if %errorlevel%==0 (
    echo The Rust toolchain ^(rustup + cargo^) is installed on this machine.
    echo Rust may be used by other projects. Only remove it if you installed
    echo it solely for Zynkbot.
    echo.
    set /p DEL_RUST="Remove the Rust toolchain? [y/N]: "
    if /i "!DEL_RUST!"=="y" (
        rustup self uninstall -y
        echo Rust toolchain removed.
    ) else (
        echo Rust toolchain kept.
    )
) else (
    echo Rust toolchain not found.
)
echo.

REM ============================================
REM Project folder
REM ============================================
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
echo The Zynkbot project folder is at: %SCRIPT_DIR%
echo This contains the app, your downloaded models, and configuration.
echo.
set /p DEL_PROJ="Delete the entire project folder? [y/N]: "
if /i "!DEL_PROJ!"=="y" (
    echo Scheduling deletion of project folder...
    REM Use a detached cmd to delete after this script exits
    start "" /b cmd /c "timeout /t 2 /nobreak >nul && rmdir /s /q ""%SCRIPT_DIR%"""
    echo Project folder will be deleted in a moment.
) else (
    echo Project folder kept.
    echo You can delete it manually in File Explorer.
)
echo.

echo =========================================
echo    Zynkbot has been uninstalled.
echo =========================================
echo.
echo Thank you for trying Zynkbot!
echo GitHub: https://github.com/MSkill1/zynkbot
echo.
pause
