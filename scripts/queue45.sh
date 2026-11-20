#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
export CHAOS_DTYPE=float16

while pgrep -f "queue44.sh" > /dev/null; do sleep 60; done
echo "QUEUE45 START $(date)"

# 300 independent tangent starts per model
[ -f results/depth2/deepseek1000/depth.npz ] || \
  $PY39 -m chaos_llm.run_depth2 \
    --model ~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B \
    --run results/big4/a1_all_layers/prompts \
    --tokens $(seq -s, 0 1 299) --k 8 --dtype float32 \
    --out results/depth2/deepseek1000 || echo "FAILED deepseek1000"

[ -f results/depth2/qwen3_4b256/depth.npz ] || \
  $PY -m chaos_llm.run_depth2 \
    --model ~/LLM/models--Qwen--Qwen3-4B-Instruct-2507 \
    --run results/models/qwen3_4b/a1 \
    --tokens $(seq -s, 0 1 299) --k 8 --dtype float16 \
    --out results/depth2/qwen3_4b256 || echo "FAILED qwen3_4b256"

echo "QUEUE45 DONE $(date)"
touch results/QUEUE45_DONE
