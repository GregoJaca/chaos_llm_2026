#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CHAOS_DTYPE=float16
PY39=python3
PY=~/LLM/venv_tf/bin/python

# pause the big queue (it is resumable) and the chained queue
pkill -f "watchdog52.sh"
pkill -f "bash scripts/queue52.sh"
pkill -f "queue53.sh"
pkill -f "run_thresholds"
sleep 8
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo "SUPERPOSITION START $(date)"
$PY39 -m chaos_llm.run_superposition \
  --model ~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B \
  --run results/sens/deepseek_p1/geom --trials 3 --r 0.1 --layers 1,7,14,last \
  --dtype float32 --out results/superposition/deepseek || echo "FAILED sup deepseek"
sleep 8

$PY -m chaos_llm.run_superposition \
  --model ~/LLM/models--Qwen--Qwen3-4B-Instruct-2507 \
  --run results/sens/qwen3_4b_p1/geom --trials 3 --r 0.1 --layers 1,7,14,last \
  --dtype float16 --out results/superposition/qwen3_4b || echo "FAILED sup qwen3_4b"
echo "SUPERPOSITION DONE $(date)"

# resume the queues
nohup setsid bash scripts/watchdog52.sh > results/watchdog52.log 2>&1 < /dev/null &
nohup setsid bash scripts/queue53.sh > results/queue53.log 2>&1 < /dev/null &
sleep 5
echo "queues resumed"
