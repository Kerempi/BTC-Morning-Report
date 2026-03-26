@echo off
setlocal
powershell.exe -ExecutionPolicy Bypass -File "C:\Users\Kerem\LaevitasMorningReportRuntime\run_laevitas_morning_report.ps1"
exit /b %errorlevel%
