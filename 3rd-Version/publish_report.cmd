@echo off
setlocal
powershell.exe -ExecutionPolicy Bypass -File "C:\Users\Kerem\LaevitasMorningReportRuntime\publish_reports_to_github.ps1"
if errorlevel 1 (
  echo.
  echo Publish failed.
  exit /b 1
)
echo.
echo Publish completed.
exit /b 0
