# Laevitas Recovery Pack

This document is for restoring the current operating model on a new Windows machine without changing how you work.

## Target model

1. The report is generated locally every morning at `08:00`.
2. The scheduled task calls a user-level wrapper:
   - `C:\Users\Kerem\run_laevitas_morning_report.ps1`
3. That wrapper delegates into the stable runtime folder:
   - `C:\Users\Kerem\LaevitasMorningReportRuntime`
4. GitHub publish stays manual:
   - `C:\Users\Kerem\publish_laevitas_report.cmd`

## Required software
- Windows
- Python 3.12
- Git
- GitHub push access
- Laevitas API key

## Quick recovery steps

### 1. Get the code
Clone the repo and switch to the recovery branch if needed.

### 2. Install Python dependencies
```powershell
cd "C:\path\to\BTC-Morning-Report\3rd-Version"
python -m pip install -r requirements.txt
```

### 3. Create `.env`
```powershell
Copy-Item .env.example .env
```

Add:
```text
LAEVITAS_API_KEY=YOUR_REAL_KEY
```

### 4. Create stable runtime folder
Create:
```text
C:\Users\Kerem\LaevitasMorningReportRuntime
```

Copy these into it:
- `laevitas_morning_report.py`
- `laevitas_api.py`
- `.env`
- `.env.example`
- `requirements.txt`
- `publish_reports_to_github.ps1`
- `run_laevitas_morning_report.ps1`
- `reports\`
- `data\`

### 5. Test a local run
```powershell
powershell.exe -ExecutionPolicy Bypass -File "C:\Users\Kerem\LaevitasMorningReportRuntime\run_laevitas_morning_report.ps1"
```

Expected outputs:
- `C:\Users\Kerem\LaevitasMorningReportRuntime\reports\btc_morning_report.html`
- `C:\Users\Kerem\LaevitasMorningReportRuntime\reports\btc_morning_report_en.html`
- `C:\Users\Kerem\LaevitasMorningReportRuntime\data\laevitas\btc_daily_archive.csv`

### 6. Recreate the scheduled task
```powershell
.\create_laevitas_morning_report_task.cmd
```

### 7. Manual publish stays manual
```powershell
C:\Users\Kerem\publish_laevitas_report.cmd
```

## Fast diagnosis
1. `C:\Users\Kerem\LaevitasMorningReportRuntime\laevitas_morning_report.log`
2. `schtasks /Query /TN LaevitasBTCMorningReport /V /FO LIST`
3. Check that `C:\Users\Kerem\run_laevitas_morning_report.ps1` exists
4. Check that runtime files exist under `C:\Users\Kerem\LaevitasMorningReportRuntime`
5. Check `.env` has a real key

## Codex prompt
Use:
- `README_laevitas_quickstart.md`
- `CODEX_LAEVITAS_BOOTSTRAP_PROMPT.txt`

## Note
This does not move you to a cloud service.
It preserves the current model, but makes it easier to restore quickly.
