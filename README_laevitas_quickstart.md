# Laevitas 5-Minute Quickstart

Use this on a new Windows machine to restore the current model quickly.

## Target behavior
- Local automatic update every morning at `08:00`
- Manual GitHub publish only
- Turkish main link stays the default
- English report stays on a separate link

## One sentence for Codex
```text
Check the `laevitas-recovery-clean` branch on GitHub. Use `3rd-Version/README_laevitas_quickstart.md` and `3rd-Version/CODEX_LAEVITAS_BOOTSTRAP_PROMPT.txt` to restore the Laevitas morning report system without changing its current operating model.
```

## Minimal setup
```powershell
cd "C:\path\to\BTC-Morning-Report\3rd-Version"
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Put your real key into `.env`:
```text
LAEVITAS_API_KEY=YOUR_REAL_KEY
```

Create:
```text
C:\Users\Kerem\LaevitasMorningReportRuntime
```

Then run:
```powershell
powershell.exe -ExecutionPolicy Bypass -File "C:\Users\Kerem\LaevitasMorningReportRuntime\run_laevitas_morning_report.ps1"
.\create_laevitas_morning_report_task.cmd
```

## Manual publish when needed
```powershell
C:\Users\Kerem\publish_laevitas_report.cmd
```

## Public links
- Turkish main:
  - `https://kerempi.github.io/BTC-Morning-Report/`
- English separate:
  - `https://kerempi.github.io/BTC-Morning-Report/btc_morning_report_en.html`

## If something fails
1. `README_laevitas_recovery.md`
2. `CODEX_LAEVITAS_BOOTSTRAP_PROMPT.txt`
3. `C:\Users\Kerem\LaevitasMorningReportRuntime\laevitas_morning_report.log`
