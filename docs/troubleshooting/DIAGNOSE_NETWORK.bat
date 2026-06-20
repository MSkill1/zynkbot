@echo off
title Network Diagnostics for Zynkbot
color 0B

REM Create log file
set LOGFILE=%CD%\network_diagnostic_results.txt
echo Network Diagnostic Results > "%LOGFILE%"
echo Generated: %date% %time% >> "%LOGFILE%"
echo ======================================== >> "%LOGFILE%"
echo. >> "%LOGFILE%"

echo ========================================
echo  Network Diagnostics
echo ========================================
echo.
echo This will test:
echo   1. Internet connectivity
echo   2. DNS resolution
echo   3. GitHub accessibility
echo   4. Download speeds
echo.
echo Results will be saved to: network_diagnostic_results.txt
echo.
pause

REM ============================================
REM Test 1: Basic Internet Connectivity
REM ============================================
echo.
echo =========================================
echo Test 1: Basic Internet Connectivity
echo =========================================
echo.
echo Testing connection to Google DNS (8.8.8.8)...
ping -n 4 8.8.8.8

if %errorLevel% neq 0 (
    echo [FAIL] No internet connection detected
    echo.
    echo Possible causes:
    echo   - WiFi/Ethernet disconnected
    echo   - Router offline
    echo   - ISP outage
    echo.
    goto :check_adapter
) else (
    echo [OK] Internet connection active
)
echo.

REM ============================================
REM Test 2: DNS Resolution
REM ============================================
echo =========================================
echo Test 2: DNS Resolution
echo =========================================
echo.
echo Testing DNS lookup for github.com...
nslookup github.com

if %errorLevel% neq 0 (
    echo [FAIL] DNS resolution failed
    echo.
    echo Trying alternate DNS (Google 8.8.8.8)...
    nslookup github.com 8.8.8.8
    echo.
    echo If this works, your DNS server might be slow.
) else (
    echo [OK] DNS resolution working
)
echo.

REM ============================================
REM Test 3: GitHub Connectivity
REM ============================================
echo =========================================
echo Test 3: GitHub Connectivity
echo =========================================
echo.
echo Testing connection to github.com...
ping -n 4 github.com

if %errorLevel% neq 0 (
    echo [FAIL] Cannot reach GitHub
    echo This could be a firewall or DNS issue
) else (
    echo [OK] GitHub is reachable
)
echo.

REM ============================================
REM Test 4: Download Speed Test
REM ============================================
echo =========================================
echo Test 4: Download Speed Test
echo =========================================
echo.
echo Downloading a 1MB test file from GitHub...
echo.

set START_TIME=%time%
echo Start time: %START_TIME%

REM Download a small file from GitHub (using a known public repo)
curl -L -o "%TEMP%\speedtest.tmp" "https://github.com/git/git/archive/refs/tags/v2.43.0.zip" --max-time 30 --progress-bar

set END_TIME=%time%
echo End time: %END_TIME%

if %errorLevel% neq 0 (
    echo [FAIL] Download failed or too slow (timeout after 30s)
    echo.
    echo This indicates very slow internet speeds or connection issues.
) else (
    echo [OK] Download completed
    echo.

    REM Show file size
    for %%A in ("%TEMP%\speedtest.tmp") do set SIZE=%%~zA
    echo Downloaded file size: %SIZE% bytes

    REM Clean up
    del "%TEMP%\speedtest.tmp" 2>nul
)
echo.

REM ============================================
REM Test 5: Network Adapter Info
REM ============================================
:check_adapter
echo =========================================
echo Test 5: Network Adapter Information
echo =========================================
echo.
ipconfig | findstr /C:"Wireless" /C:"Ethernet" /C:"IPv4" /C:"Default Gateway"
echo.

REM ============================================
REM Test 6: Internet Speed via PowerShell
REM ============================================
echo =========================================
echo Test 6: Approximate Download Speed
echo =========================================
echo.
echo Measuring download speed (10 second test)...
echo.

powershell -Command "$progressPreference = 'silentlyContinue'; $start = Get-Date; try { $webClient = New-Object System.Net.WebClient; $data = $webClient.DownloadData('https://speed.cloudflare.com/__down?bytes=10000000'); $end = Get-Date; $duration = ($end - $start).TotalSeconds; $sizeMB = $data.Length / 1MB; $speedMbps = ($sizeMB * 8) / $duration; Write-Host \"Downloaded: $([math]::Round($sizeMB, 2)) MB\"; Write-Host \"Time: $([math]::Round($duration, 2)) seconds\"; Write-Host \"Speed: $([math]::Round($speedMbps, 2)) Mbps\"; if ($speedMbps -lt 1) { Write-Host \"`n[WARNING] Very slow connection (under 1 Mbps)\" -ForegroundColor Red; Write-Host \"This will cause git/installation issues\" -ForegroundColor Red } elseif ($speedMbps -lt 5) { Write-Host \"`n[WARNING] Slow connection (under 5 Mbps)\" -ForegroundColor Yellow; Write-Host \"Downloads may timeout\" -ForegroundColor Yellow } else { Write-Host \"`n[OK] Connection speed acceptable\" -ForegroundColor Green } } catch { Write-Host \"[FAIL] Speed test failed - connection too slow or unstable\" -ForegroundColor Red }"

echo.

REM ============================================
REM Summary and Recommendations
REM ============================================
echo ========================================
echo  Diagnostic Complete
echo ========================================
echo.
echo RECOMMENDATIONS:
echo.
echo If speeds are under 1 Mbps:
echo   - Restart your router
echo   - Contact your ISP
echo   - Try ethernet cable instead of WiFi
echo   - Wait for better connectivity before installation
echo.
echo If GitHub is unreachable:
echo   - Check firewall settings
echo   - Try: ipconfig /flushdns
echo   - Restart router
echo.
echo If DNS is slow:
echo   - Switch to Google DNS (8.8.8.8, 8.8.4.4)
echo   - Or Cloudflare DNS (1.1.1.1, 1.0.0.1)
echo.
echo For Zynkbot installation to succeed, you need:
echo   - Stable connection (no dropouts)
echo   - At least 5 Mbps download speed
echo   - GitHub.com accessible
echo.
echo ========================================
echo.
echo Opening results file...
timeout /t 2 /nobreak >nul
notepad "%LOGFILE%"
pause
