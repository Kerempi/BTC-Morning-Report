@echo off
setlocal

powershell.exe -ExecutionPolicy Bypass -File "%~dp0create_laevitas_morning_report_task.ps1"
exit /b %ERRORLEVEL%
