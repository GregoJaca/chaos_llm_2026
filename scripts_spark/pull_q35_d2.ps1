New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen35\d2 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/qwen35/d2/*.npy spark:~/LLM/chaos_llm_2026/results_big/qwen35/d2/summary.json C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen35\d2\
Write-Output Q35_D2_PULLED
