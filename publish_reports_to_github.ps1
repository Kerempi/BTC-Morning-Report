$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$publishRoot = Join-Path $repoRoot ".codex_publish_main"
$sourceReports = Join-Path $repoRoot "reports"
$targetReports = Join-Path $publishRoot "reports"

if (-not (Test-Path $sourceReports)) {
    throw "Local reports folder not found: $sourceReports"
}

if (-not (Test-Path $publishRoot)) {
    throw "Publish worktree not found: $publishRoot"
}

Write-Host "Syncing local reports to publish worktree..."
$null = robocopy $sourceReports $targetReports /MIR /FFT /R:1 /W:1 /NFL /NDL /NJH /NJS /NP

Push-Location $publishRoot
try {
    $status = git status --porcelain -- reports
    if (-not $status) {
        Write-Host "No report changes to publish."
        exit 0
    }

    git add reports
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    git commit -m "Publish local reports $timestamp"
    git push origin HEAD:main
}
finally {
    Pop-Location
}
