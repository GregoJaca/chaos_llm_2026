#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python

pkill -f queue36.sh >/dev/null 2>&1
pkill -f dim_dataset.py >/dev/null 2>&1
sleep 5
echo "QUEUE37 START $(date)"

export CHAOS_DTYPE=float32
$PY39 scripts/dim_dataset.py --tag deepseek --out results/dim/deepseek \
  --model ~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B --dtype float32 || echo "FAILED dim deepseek"

export CHAOS_DTYPE=float16
$PY scripts/dim_dataset.py --tag qwen3_4b --out results/dim/qwen3_4b \
  --model ~/LLM/models--Qwen--Qwen3-4B-Instruct-2507 --dtype float16 || echo "FAILED dim qwen3_4b"
$PY scripts/dim_dataset.py --tag phi4mini --out results/dim/phi4mini \
  --model ~/LLM/models--microsoft--Phi-4-mini-instruct --dtype float16 || echo "FAILED dim phi4mini"
$PY311 scripts/dim_dataset.py --tag qwen3_5 --out results/dim/qwen3_5 \
  --model ~/LLM/models--Qwen--Qwen3.5-2B --dtype float16 || echo "FAILED dim qwen3_5"
echo "DIM3 DONE $(date)"

bash scripts/queue35.sh > results/queue35.log 2>&1
echo "QUEUE37 DONE $(date)"
