#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
export CHAOS_DTYPE=float16
echo "QUEUE48 START $(date)"

while pgrep -f "queue47.sh" > /dev/null; do sleep 60; done
sleep 20

Q4=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PH=~/LLM/models--microsoft--Phi-4-mini-instruct
Q35=~/LLM/models--Qwen--Qwen3.5-2B
DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B

run_bs () {
  local M=$1 RUN=$2 TAG=$3 BIN=$4 DT=$5
  [ -f results/block_sigma/$TAG/block_sigma.npz ] && return
  $BIN -m chaos_llm.run_block_sigma --model "$M" --run "$RUN" \
    --tokens 16,50 --iters 20 --dtype "$DT" --out results/block_sigma/$TAG \
    || echo "FAILED bs $TAG"
  sleep 15
}

run_bs $DS results/big4/a1_all_layers/prompts deepseek $PY39 float32
run_bs $Q4 results/models/qwen3_4b/a1 qwen3_4b $PY float16
run_bs $PH results/models/phi4mini/a1 phi4mini $PY float16
run_bs $Q35 results/models/qwen3_5/a1 qwen3_5 $PY311 float16
echo "blocksigma done $(date)"

[ -f results/depth2/qwen3_4b_k64/depth.npz ] || \
  $PY -m chaos_llm.run_depth2 --model $Q4 --run results/models/qwen3_4b/a1 \
    --tokens 0,8,16,32,50,80,100,143,200,250 --k 64 --dtype float16 \
    --out results/depth2/qwen3_4b_k64 || echo "FAILED k64 q4"

echo "QUEUE48 DONE $(date)"
touch results/QUEUE48_DONE
