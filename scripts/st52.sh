#!/bin/bash
cd ~/LLM/chaos_llm_2026
echo "--- procs"
ps -eo pid,etime,args | grep -E "watchdog52|queue52|run_a1|run_transport|run_thresholds" | grep -v grep | head -5
echo "--- gpu"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- log"
tail -4 results/queue52.log
echo "--- dirs"
ls results/sens/deepseek_p1 2>/dev/null
