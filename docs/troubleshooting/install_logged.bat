@echo off
echo =========================================
echo    Running Zynkbot Installation
echo =========================================
echo.
echo Running installation and capturing log...
echo You can watch progress in this window.
echo.

REM Change to the directory where this script is located
cd /d "%~dp0"

REM Delete old log if it exists
if exist install_complete.log del install_complete.log

REM Run installation with full output capture
call install.bat 2>&1 | powershell -Command "$input | ForEach-Object { Write-Host $_; $_ | Out-File -FilePath 'install_complete.log' -Append -Encoding UTF8 }"

echo.
echo =========================================
echo Installation finished - Exit code: %errorlevel%
echo =========================================
echo.
echo Log saved to install_complete.log
echo.
echo Press any key to close this window...
pause >nul
