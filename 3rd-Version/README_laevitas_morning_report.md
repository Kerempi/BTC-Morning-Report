# Laevitas Morning Report

## Purpose
Generate the BTC morning options note locally every morning, while keeping GitHub publish manual.

## Runtime model
- Stable runtime folder:
  - `C:\Users\Kerem\LaevitasMorningReportRuntime`
- Morning scheduled task calls:
  - `C:\Users\Kerem\run_laevitas_morning_report.ps1`
- That wrapper delegates into the stable runtime folder, so branch changes in this repo do not break the 08:00 update.

## What stays the same
- Local automatic update every morning at `08:00`
- Manual GitHub publish only when you choose
- Turkish main report stays the default public link
- English report stays on a separate link

## Manual local run
```powershell
powershell.exe -ExecutionPolicy Bypass -File "C:\Users\Kerem\LaevitasMorningReportRuntime\run_laevitas_morning_report.ps1"
```

## Manual publish
```powershell
C:\Users\Kerem\publish_laevitas_report.cmd
```

## Main outputs
- `C:\Users\Kerem\LaevitasMorningReportRuntime\reports\btc_morning_report.html`
- `C:\Users\Kerem\LaevitasMorningReportRuntime\reports\btc_morning_report_en.html`
- `C:\Users\Kerem\LaevitasMorningReportRuntime\reports\laevitas\*.png`
- `C:\Users\Kerem\LaevitasMorningReportRuntime\data\laevitas\btc_daily_archive.csv`
- `C:\Users\Kerem\LaevitasMorningReportRuntime\laevitas_morning_report.log`

## Recovery
- Fast rebuild / recovery guide: `README_laevitas_recovery.md`
- Five-minute quickstart: `README_laevitas_quickstart.md`
- Copy-paste prompt for Codex on a new machine: `CODEX_LAEVITAS_BOOTSTRAP_PROMPT.txt`
