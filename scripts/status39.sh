#!/bin/bash
cd ~/LLM/chaos_llm_2026
echo "--- procs"
ps -eo pid,etime,args | grep -E "queue39|model_suite2|run_thresholds" | grep -v grep | head -5
echo "--- gpu"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- queue39 tail"
tail -12 results/queue39.log
echo "--- dirs"
ls results/models/qwen25_3b 2>/dev/null
ls results/models/qwen3_4b/bigr 2>/dev/null
