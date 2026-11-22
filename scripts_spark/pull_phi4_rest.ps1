New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\state_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/phi4/state_w0/state2.npz spark:~/LLM/chaos_llm_2026/results_big/phi4/state_w0/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\state_w0\
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\depth_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/phi4/depth_w0/depth.npz spark:~/LLM/chaos_llm_2026/results_big/phi4/depth_w0/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\depth_w0\
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\nonnorm_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/phi4/nonnorm_w0/nonnorm.npz spark:~/LLM/chaos_llm_2026/results_big/phi4/nonnorm_w0/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\nonnorm_w0\
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\a1_childhood_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/phi4/a1_childhood_w0/a1.npz spark:~/LLM/chaos_llm_2026/results_big/phi4/a1_childhood_w0/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\a1_childhood_w0\
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\a1_cyber_w0 | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/phi4/a1_cyber_w0/a1.npz spark:~/LLM/chaos_llm_2026/results_big/phi4/a1_cyber_w0/meta.json C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\a1_cyber_w0\
New-Item -ItemType Directory -Force C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\traj | Out-Null
scp -q spark:~/LLM/chaos_llm_2026/results_big/phi4/traj/*.npz C:\Users\hjaca\research\chaos_spark_wt\results_big\phi4\traj\
Write-Output PHI4_REST_PULLED
