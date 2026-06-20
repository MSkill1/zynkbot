@echo off
title Fix Windows Defender Exclusions
color 0B

echo ========================================
echo  Add Windows Defender Exclusions
echo ========================================
echo.
REM Get the current directory (Zynkbot project root)
set ZYNK_PROJECT=%CD%

echo This will add the following folders to Windows Defender exclusions:
echo   - %ZYNK_PROJECT% (entire project)
echo   - C:\Program Files\Git (Git installation)
echo   - %USERPROFILE%\.cargo (Rust cargo)
echo   - %USERPROFILE%\.rustup (Rust toolchain)
echo.
echo This prevents Windows Defender from scanning these folders,
echo which can cause significant slowdowns.
echo.
set /p CONTINUE="Continue? (y/n): "
if /i not "%CONTINUE%"=="y" exit /b 0

echo.
echo [INFO] Adding exclusions (requires Administrator)...
echo.

REM Check if running as admin
net session >nul 2>&1
if errorlevel 1 (
    echo [ERROR] This script requires Administrator privileges.
    echo.
    echo Please:
    echo   1. Right-click this file
    echo   2. Select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo [1/4] Adding Zynkbot project folder...
powershell -Command "Add-MpPreference -ExclusionPath '%ZYNK_PROJECT%'"
echo [OK] Added %ZYNK_PROJECT%

echo [2/4] Adding Git folder...
powershell -Command "Add-MpPreference -ExclusionPath 'C:\Program Files\Git'" 2>nul
echo [OK] Added Git folder

echo [3/4] Adding Cargo folder...
powershell -Command "Add-MpPreference -ExclusionPath '%USERPROFILE%\.cargo'"
echo [OK] Added .cargo folder

echo [4/4] Adding Rustup folder...
powershell -Command "Add-MpPreference -ExclusionPath '%USERPROFILE%\.rustup'"
echo [OK] Added .rustup folder

echo.
echo ========================================
echo  Verifying Exclusions
echo ========================================
echo.
powershell -Command "Get-MpPreference | Select-Object -ExpandProperty ExclusionPath"
echo.
echo ========================================
echo  Success!
echo ========================================
echo.
echo Windows Defender exclusions have been added.
echo Zynkbot should now run much faster.
echo.
echo NEXT STEPS:
echo   1. Close this window
echo   2. Run START_ZYNKBOT.bat
echo   3. Test if the 12-second delay is gone
echo.
pause
