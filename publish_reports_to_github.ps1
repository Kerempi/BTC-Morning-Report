$ErrorActionPreference = 'Stop'
$runtimeScript = 'C:\Users\Kerem\LaevitasMorningReportRuntime\publish_reports_to_github.ps1'

if (-not (Test-Path $runtimeScript)) {
  throw "Runtime publish script not found: $runtimeScript"
}

& $runtimeScript
exit $LASTEXITCODE
