$procs = Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -eq 'python.exe' -and $_.CommandLine -like '*build.py*') -or
    ($_.Name -eq 'cmd.exe' -and $_.CommandLine -like '*run_forever*')
}
$procs | Select-Object ProcessId, Name, CommandLine | Format-List
foreach ($p in $procs) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 4
"--- after kill"
netstat -ano | Select-String ":8899"
Start-Process -FilePath "wscript.exe" -ArgumentList "`"C:\Users\hjaca\research\chaos_llm_2026\notebook\.preview\run_hidden.vbs`""
Start-Sleep -Seconds 10
"--- after start"
netstat -ano | Select-String ":8899"
