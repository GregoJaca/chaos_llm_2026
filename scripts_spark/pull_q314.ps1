New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\a1_inter_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/qwen314/a1_inter_w0/a1.npz spark:~/LLM/chaos_llm_2026/results_big/qwen314/a1_inter_w0/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\a1_inter_w0\
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\free_inter_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/qwen314/free_inter_w0/free_run.npz spark:~/LLM/chaos_llm_2026/results_big/qwen314/free_inter_w0/summary.json C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\free_inter_w0\
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\calib | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/qwen314/calib/calib.npz spark:~/LLM/chaos_llm_2026/results_big/qwen314/calib/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\calib\
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\assets\tok_qwen314 | Out-Null
scp -q spark:~/LLM/models/Qwen3-14B/tokenizer.json spark:~/LLM/models/Qwen3-14B/tokenizer_config.json C:\Users\hjaca\research\chaos_spark_wt\assets\tok_qwen314\
Write-Output Q314_CORE_PULLED
