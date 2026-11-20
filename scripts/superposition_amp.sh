#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CHAOS_DTYPE=float16
PY39=python3

pkill -f "watchdog52.sh"; pkill -f "bash scripts/queue52.sh"; pkill -f "queue53.sh"; pkill -f "run_thresholds"
sleep 8
nvidia-smi --query-gpu=memory.used --format=csv,noheader

for r in 0.003 0.01 0.03; do
  $PY39 -m chaos_llm.run_superposition \
    --model ~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B \
    --run results/sens/deepseek_p1/geom --trials 2 --r $r --layers 1,7,14,last \
    --dtype float32 --out results/superposition/deepseek_r$r || echo "FAILED sup $r"
  sleep 5
done

nohup setsid bash scripts/watchdog52.sh > results/watchdog52.log 2>&1 < /dev/null &
nohup setsid bash scripts/queue53.sh > results/queue53.log 2>&1 < /dev/null &
sleep 3
echo "resumed"
