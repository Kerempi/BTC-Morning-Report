@echo off
setlocal

set TASK_NAME=LaevitasBTCMorningReport
set TASK_TIME=08:00
set TARGET_CMD="C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -ExecutionPolicy Bypass -File "C:\Users\Kerem\run_laevitas_morning_report.ps1"

schtasks /Create /TN "%TASK_NAME%" /SC DAILY /ST %TASK_TIME% /TR %TARGET_CMD% /F
if errorlevel 1 (
  echo Failed to create scheduled task.
  exit /b 1
)

echo Task created: %TASK_NAME% at %TASK_TIME%
exit /b 0
