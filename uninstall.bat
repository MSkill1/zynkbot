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
echo   - Optionally remove your memory database, API keys and device identity
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
REM No '|| true' here: that is a shell idiom, not a cmd command, and taskkill
REM returns non-zero whenever the process is simply not running -- so every run
REM printed "'true' is not recognized as an internal or external command" twice.
taskkill /f /im "zynkbot.exe" >nul 2>&1
taskkill /f /im "app.exe" >nul 2>&1
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
REM Memory database and device identity
REM ============================================
REM Zynkbot stores state in TWO places, not one. Removing only the first leaves
REM .zynk_user_id and .zynk_device_id behind, so a later install is recognised as
REM the same user and a "fresh install" is not fresh -- which silently invalidates
REM any first-run or onboarding test.
set DB_DIR=%LOCALAPPDATA%\zynkbot
set ID_DIR=%APPDATA%\zynkbot
REM WebView2 profile: localStorage (dictation source, preferred model, wake word,
REM TTS, onboarding flags), cookies and cache. Keyed by the Tauri identifier, not
REM the app name, so it was missed until 2026-09-12 (KI-048): a "fresh" install
REM came back with the old preferences, e.g. dictation defaulting to OpenAI.
set WV_DIR=%LOCALAPPDATA%\ai.containai.zynkbot
set FOUND_DATA=0
if exist "%DB_DIR%" set FOUND_DATA=1
if exist "%ID_DIR%" set FOUND_DATA=1
if exist "%WV_DIR%" set FOUND_DATA=1
if "%FOUND_DATA%"=="1" (
    echo Zynkbot keeps your data in three locations:
    echo.
    echo   %DB_DIR%
    echo     memory database, API keys, ZynkSync TLS identity
    echo   %ID_DIR%
    echo     user id and device id
    echo   %WV_DIR%
    echo     saved preferences (dictation source, preferred model, wake word, onboarding)
    echo.
    echo Keeping these means a future install is recognised as the SAME user.
    echo Delete all three if you are testing a first-run or new-user install.
    echo.
    set /p DEL_DB="Delete ALL Zynkbot data? This cannot be undone. [y/N]: "
    if /i "!DEL_DB!"=="y" (
        if exist "%DB_DIR%" rmdir /s /q "%DB_DIR%"
        if exist "%ID_DIR%" rmdir /s /q "%ID_DIR%"
        if exist "%WV_DIR%" rmdir /s /q "%WV_DIR%"
        echo All Zynkbot data deleted - the next install will behave as a new user.
    ) else (
        echo Data kept in all three locations.
        echo NOTE: the next install will NOT behave as a new user.
        echo To clear it later, run these three commands:
        echo   rmdir /s /q "%DB_DIR%"
        echo   rmdir /s /q "%ID_DIR%"
        echo   rmdir /s /q "%WV_DIR%"
    )
) else (
    echo No Zynkbot data found.
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
