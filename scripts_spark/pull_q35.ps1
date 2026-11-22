New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen35\a1_inter_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/qwen35/a1_inter_w0/a1.npz spark:~/LLM/chaos_llm_2026/results_big/qwen35/a1_inter_w0/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen35\a1_inter_w0\
Write-Output "a1 pulled"
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen35\free_inter_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/qwen35/free_inter_w0/free_run.npz spark:~/LLM/chaos_llm_2026/results_big/qwen35/free_inter_w0/summary.json C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen35\free_inter_w0\
Write-Output "free pulled"
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen35\calib | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/qwen35/calib/calib.npz spark:~/LLM/chaos_llm_2026/results_big/qwen35/calib/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen35\calib\
Write-Output "calib pulled"
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\assets\tok_qwen35 | Out-Null
scp -q spark:~/LLM/models/Qwen3.5-9B/tokenizer.json spark:~/LLM/models/Qwen3.5-9B/tokenizer_config.json C:\Users\hjaca\research\chaos_spark_wt\assets\tok_qwen35\
Write-Output "tokenizer pulled"
Write-Output Q35_PULL_DONE
