#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
export CHAOS_DTYPE=float16
echo "QUEUE50 START $(date)"

run_bs () {
  local M=$1 RUN=$2 TAG=$3 BIN=$4 DT=$5
  [ -f results/block_sigma/$TAG/block_sigma.npz ] && return
  $BIN -m chaos_llm.run_block_sigma --model "$M" --run "$RUN" \
    --tokens 16,50,100,143 --iters 20 --dtype "$DT" --out results/block_sigma/$TAG \
    || echo "FAILED bs $TAG"
  sleep 10
}

run_bs ~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B \
  results/big4/a1_all_layers/prompts deepseek $PY39 float32
run_bs ~/LLM/models--Qwen--Qwen3-4B-Instruct-2507 \
  results/models/qwen3_4b/a1 qwen3_4b $PY float16
run_bs ~/LLM/models--microsoft--Phi-4-mini-instruct \
  results/models/phi4mini/a1 phi4mini $PY float16
run_bs ~/LLM/models--Qwen--Qwen3.5-2B \
  results/models/qwen3_5/a1 qwen3_5 $PY311 float16

echo "QUEUE50 DONE $(date)"
touch results/QUEUE50_DONE
