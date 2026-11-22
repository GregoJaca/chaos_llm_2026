New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\a1_inter_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/phi4/a1_inter_w0/a1.npz spark:~/LLM/chaos_llm_2026/results_big/phi4/a1_inter_w0/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\a1_inter_w0\
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\free_inter_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/phi4/free_inter_w0/free_run.npz spark:~/LLM/chaos_llm_2026/results_big/phi4/free_inter_w0/summary.json C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\free_inter_w0\
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\calib | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/phi4/calib/calib.npz spark:~/LLM/chaos_llm_2026/results_big/phi4/calib/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\calib\
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\assets\tok_phi4 | Out-Null
scp -q spark:~/LLM/models/phi-4/tokenizer.json spark:~/LLM/models/phi-4/tokenizer_config.json C:\Users\hjaca\research\chaos_spark_wt\assets\tok_phi4\
Write-Output PHI4_CORE_PULLED
