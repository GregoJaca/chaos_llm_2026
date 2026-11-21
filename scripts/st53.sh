#!/bin/bash
cd ~/LLM/chaos_llm_2026
echo "--- procs"
ps -eo pid,etime,args | grep -E "queue52|queue53|watchdog52|run_thresholds|run_a1|run_transport" | grep -v grep
echo "--- gpu"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- last outputs"
ls -lt results/sens/*/r*/free_run.npz | head -3
echo "--- tail"
tail -2 results/queue52.log
