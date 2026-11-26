# Refresh the report from whatever the clusters have produced so far.
#
#   powershell -ExecutionPolicy Bypass -File scripts\refresh_report.ps1
#
# One pass: pull data -> regenerate the loop figures -> rebuild the pdf -> commit.
# Deliberately a single foreground process (no scheduling, no spawning), because
# an earlier auto-rebuild task ran away and spawned hundreds of processes.
param(
    [switch]$NoCommit,
    [switch]$SkipSync
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot

if (-not $SkipSync) {
    powershell -ExecutionPolicy Bypass -File (Join-Path $repo "analysis_update\sync_from_cluster.ps1")
}

Push-Location $repo
try {
    $scripts = @(
        "pack_loop_loglog_models.py",
        "pack_loop_saturation.py",
        "pack_loop_layers.py",
        "pack_loop_intermittency.py",
        "pack_loop_selfreg.py",
        "pack_loop_metrics.py",
        "pack_loop_finitetime.py"
    )
    foreach ($s in $scripts) {
        Write-Host "=== $s"
        python (Join-Path $repo "analysis_update\$s")
    }
    Copy-Item (Join-Path $repo "update_figures\*.png") (Join-Path $repo "notebook\update_figures") -Force

    Push-Location (Join-Path $repo "notebook")
    python ".preview\build.py"
    Pop-Location
}
finally {
    Pop-Location
}

if (-not $NoCommit) {
    Push-Location $repo
    git add -A
    git -c user.name="Research Agent" -c user.email="agent@local" commit -m "Refresh report from long-campaign data"
    Pop-Location
}
Write-Host "done: notebook/update.pdf"
