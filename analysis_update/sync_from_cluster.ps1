# Pull campaign results from x11 and Spark into the local checkout.
#
#   powershell -ExecutionPolicy Bypass -File analysis_update\sync_from_cluster.ps1
#   ... -Hosts x11 -Heavy          (also the multi-GB longloop trees)
#
# Remote side: scripts/sync_pack.sh writes /tmp/chaos_sync.tar (uncompressed);
# we scp it and untar over results/.  Idempotent, nothing is deleted.
param(
    [string[]]$Hosts = @("x11", "spark"),
    [switch]$Heavy
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$tmp = Join-Path $env:TEMP "chaos_sync"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$mode = if ($Heavy) { "heavy" } else { "" }

foreach ($h in $Hosts) {
    $tar = Join-Path $tmp "$h.tar"
    Write-Host "=== $h"
    ssh $h "cd ~/LLM/chaos_llm_2026 && bash scripts/sync_pack.sh $mode"
    if ($LASTEXITCODE -ne 0) { Write-Host "    pack failed"; continue }
    scp "${h}:/tmp/chaos_sync.tar" $tar
    if (-not (Test-Path $tar) -or (Get-Item $tar).Length -lt 1000) {
        Write-Host "    nothing to sync"; continue
    }
    tar xf $tar -C $repo
    Write-Host ("    merged {0:N1} MB" -f ((Get-Item $tar).Length / 1MB))
}
Write-Host "done. Now run: python analysis_update/run_all.py"
