# Pull the long-campaign results from x11 and Spark into the local checkout.
#
#   powershell -ExecutionPolicy Bypass -File analysis_update\sync_from_cluster.ps1
#   powershell ... -File analysis_update\sync_from_cluster.ps1 -Only longtf
#
# Both clusters write the same layout:
#   results/longloop/<tag>/T<T>/loop/softloop.npz      closed loop
#   results/longloop/<tag>/T<T>/replay/softloop.npz    network-only replay
#   results/longloop/<tag>/T1.0/crn|ind/softloop.npz   sampled-feedback controls
#   results/longtf/<tag>/a1.npz                        long teacher-forced
# so a plain merge into the local results/ tree is what we want.
param(
    [string]$Only = "",
    [string[]]$Hosts = @("x11", "spark")
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$remote = "~/LLM/chaos_llm_2026"
$tmp = Join-Path $env:TEMP "chaos_sync"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

$targets = if ($Only) { @($Only) } else { @("longloop", "longtf") }

foreach ($h in $Hosts) {
    foreach ($t in $targets) {
        $tgz = Join-Path $tmp "$h-$t.tgz"
        Write-Host "=== $h : results/$t"
        $exists = ssh $h "test -d $remote/results/$t && echo yes || echo no"
        if ($exists.Trim() -ne "yes") {
            Write-Host "    nothing on $h"
            continue
        }
        ssh $h "cd $remote && tar czf /tmp/chaos_$t.tgz results/$t"
        scp "${h}:/tmp/chaos_$t.tgz" $tgz
        if ((Get-Item $tgz).Length -lt 100) {
            Write-Host "    empty archive, skipped"
            continue
        }
        tar xzf $tgz -C $repo
        Write-Host ("    merged {0:N1} MB" -f ((Get-Item $tgz).Length / 1MB))
    }
}
Write-Host "done. Now run: python analysis_update/run_all.py"
