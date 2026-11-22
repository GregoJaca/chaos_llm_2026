$dirs = @('a1_inter_w200_fixed','a1_inter_w300_fixed','a1_inter_w500_fixed','a1_inter_w100_posearly_0-22','a1_inter_w100_posearly_22-44','a1_inter_w100_posearly_44-65')
foreach ($d in $dirs) {
  New-Item -ItemType Directory -Force ("C:\Users\hjaca\research\chaos_spark_wt\results_big\ds14\" + $d) | Out-Null
  scp -q ("spark:~/LLM/chaos_llm_2026/results_big/ds14/" + $d + "/a1.npz") ("spark:~/LLM/chaos_llm_2026/results_big/ds14/" + $d + "/meta.json") ("C:\Users\hjaca\research\chaos_spark_wt\results_big\ds14\" + $d + "\")
  Write-Output ("pulled " + $d)
}
