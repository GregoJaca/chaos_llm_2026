$dirs = @('a1_inter_w500','a1_childhood_w100','a1_cyber_w100')
foreach ($d in $dirs) {
  New-Item -ItemType Directory -Force ("C:\Users\hjaca\research\chaos_spark_wt\results_big\ds14\" + $d) | Out-Null
  scp -q ("spark:~/LLM/chaos_llm_2026/results_big/ds14/" + $d + "/a1.npz") ("spark:~/LLM/chaos_llm_2026/results_big/ds14/" + $d + "/meta.json") ("C:\Users\hjaca\research\chaos_spark_wt\results_big\ds14\" + $d + "\")
  Write-Output ("pulled " + $d)
}
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\ds14\state_w100 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/ds14/state_w100/state2.npz spark:~/LLM/chaos_llm_2026/results_big/ds14/state_w100/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\ds14\state_w100\
Write-Output "pulled state_w100"
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\ds14\traj_w4096 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/ds14/traj_w4096/*.npz C:\Users\hjaca\research\chaos_spark_wt\results_big\ds14\traj_w4096\
Write-Output "pulled trajectories"
Write-Output PHASE2_PULL_DONE
