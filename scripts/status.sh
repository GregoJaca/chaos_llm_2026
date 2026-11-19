#!/bin/bash
cd ~/LLM/chaos_llm_2026
echo "--- processes"
ps -eo pid,etime,args | grep -E "chaos_llm|model_suite" | grep -v grep | head -8
echo "--- gpu"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "--- suite log tail"
tail -3 results/models_qwen3_4b.log
echo "--- dirs"
ls results/models/qwen3_4b
echo "--- python"
which python3.11 python3.10 conda micromamba 2>/dev/null
