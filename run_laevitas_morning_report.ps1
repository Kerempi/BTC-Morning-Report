$ErrorActionPreference = 'Stop'
$runtimeScript = 'C:\Users\Kerem\LaevitasMorningReportRuntime\run_laevitas_morning_report.ps1'

if (-not (Test-Path $runtimeScript)) {
  throw "Runtime wrapper not found: $runtimeScript"
}

& $runtimeScript
exit $LASTEXITCODE
