@echo off
setlocal
powershell.exe -ExecutionPolicy Bypass -File "%~dp0publish_reports_to_github.ps1"
if errorlevel 1 (
  echo.
  echo Publish failed.
  exit /b 1
)
echo.
echo Publish completed.
exit /b 0
