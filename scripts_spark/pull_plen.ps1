$dirs = @('a1_short_prompt_w4096','a1_short_prompt_w50_fixed','a1_short_prompt_w100_fixed','a1_short_prompt_w200_fixed','a1_short_prompt_w300_fixed','a1_short_prompt_w500_fixed','a1_long_prompt_w4096','a1_long_prompt_w50_fixed','a1_long_prompt_w100_fixed','a1_long_prompt_w200_fixed','a1_long_prompt_w300_fixed','a1_long_prompt_w500_fixed')
foreach ($d in $dirs) {
  New-Item -ItemType Directory -Force ("C:\Users\hjaca\research\chaos_spark_wt\results_big\ds14\" + $d) | Out-Null
  scp -q ("spark:~/LLM/chaos_llm_2026/results_big/ds14/" + $d + "/a1.npz") ("spark:~/LLM/chaos_llm_2026/results_big/ds14/" + $d + "/meta.json") ("C:\Users\hjaca\research\chaos_spark_wt\results_big\ds14\" + $d + "\")
  Write-Output ("pulled " + $d)
}
Write-Output PSCAN_PULL_DONE
