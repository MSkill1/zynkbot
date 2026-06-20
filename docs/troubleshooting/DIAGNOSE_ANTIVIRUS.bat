@echo off
title Antivirus/Firewall Diagnostic
color 0E

echo ========================================
echo  Zynkbot Antivirus/Firewall Diagnostic
echo ========================================
echo.

echo [1/6] Checking Windows Defender status...
powershell -Command "Get-MpComputerStatus | Select-Object RealTimeProtectionEnabled, IoavProtectionEnabled, BehaviorMonitorEnabled, AntivirusEnabled"
echo.

echo [2/6] Checking Windows Defender exclusions...
powershell -Command "Get-MpPreference | Select-Object -ExpandProperty ExclusionPath"
echo.

echo [3/6] Checking for Norton/Symantec processes...
powershell -Command "Get-Process | Where-Object {$_.ProcessName -like '*Norton*' -or $_.ProcessName -like '*Symantec*' -or $_.ProcessName -like '*ccSvc*'}"
if errorlevel 1 (
    echo [OK] No Norton processes found
) else (
    echo [WARNING] Norton processes still running! Consider using Norton Removal Tool.
)
echo.

echo [4/6] Checking firewall rules for Zynkbot...
powershell -Command "Get-NetFirewallApplicationFilter | Where-Object {$_.Program -like '*zynkbot*' -or $_.Program -like '*app.exe*'}"
if errorlevel 1 (
    echo [WARNING] No firewall rules found for Zynkbot
) else (
    echo [OK] Firewall rules exist
)
echo.

echo [5/6] Checking network connectivity...
ping -n 2 github.com
if errorlevel 1 (
    echo [ERROR] Cannot reach GitHub
) else (
    echo [OK] GitHub reachable
)
echo.

echo [6/6] Checking recent Windows Defender detections...
powershell -Command "Get-MpThreatDetection | Select-Object -First 5 | Format-Table -AutoSize"
echo.

echo ========================================
echo  Diagnostic Complete
echo ========================================
echo.
echo RECOMMENDATIONS:
echo.
echo If Real-time Protection is ON and causing slowness:
echo   1. Add exclusions (see below)
echo   2. Or temporarily disable for testing
echo.
echo If Norton processes found:
echo   1. Download Norton Removal Tool
echo   2. https://support.norton.com/sp/en/us/home/current/solutions/v60392881
echo.
echo If no firewall rules found:
echo   1. Run: scripts\windows_firewall_exception.bat
echo.
echo If GitHub unreachable:
echo   1. Check internet connection
echo   2. Run: ipconfig /flushdns
echo   3. Try: git remote set-url origin https://github.com/MSkill1/zynkbot.git
echo.
pause
