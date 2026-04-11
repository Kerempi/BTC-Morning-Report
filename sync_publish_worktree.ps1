$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceReports = Join-Path $repoRoot "reports"
$publishRoot = Join-Path $repoRoot ".codex_publish_main"
$targetReports = Join-Path $publishRoot "reports"

if (-not (Test-Path $sourceReports)) {
    throw "Source reports folder not found: $sourceReports"
}

if (-not (Test-Path $publishRoot)) {
    throw "Publish worktree not found: $publishRoot"
}

Write-Host "Syncing reports -> publish worktree..."
$null = robocopy $sourceReports $targetReports /MIR /FFT /R:1 /W:1 /NFL /NDL /NJH /NJS /NP

Write-Host ""
Write-Host "Changed files in .codex_publish_main:"
git -C $publishRoot status --short -- reports
