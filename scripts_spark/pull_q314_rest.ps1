New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\d2 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/phi4/d2/* C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\d2\
Write-Output "phi4 d2 pulled"

New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\d2 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/qwen314/d2/* C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\d2\
Write-Output "qwen314 d2 pulled"

New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\state_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/qwen314/state_w0/state2.npz spark:~/LLM/chaos_llm_2026/results_big/qwen314/state_w0/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\state_w0\
Write-Output "qwen314 state_w0 pulled"

New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\depth_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/qwen314/depth_w0/depth.npz spark:~/LLM/chaos_llm_2026/results_big/qwen314/depth_w0/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\depth_w0\
Write-Output "qwen314 depth_w0 pulled"

New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\nonnorm_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/qwen314/nonnorm_w0/nonnorm.npz spark:~/LLM/chaos_llm_2026/results_big/qwen314/nonnorm_w0/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\nonnorm_w0\
Write-Output "qwen314 nonnorm_w0 pulled"

New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\a1_childhood_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/qwen314/a1_childhood_w0/a1.npz spark:~/LLM/chaos_llm_2026/results_big/qwen314/a1_childhood_w0/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\a1_childhood_w0\
Write-Output "qwen314 childhood pulled"

New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\a1_cyber_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/qwen314/a1_cyber_w0/a1.npz spark:~/LLM/chaos_llm_2026/results_big/qwen314/a1_cyber_w0/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\qwen314\a1_cyber_w0\
Write-Output "qwen314 cyber pulled"

Write-Output "ALL_REST_PULLED"
