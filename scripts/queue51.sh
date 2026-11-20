#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
export CHAOS_DTYPE=float16
echo "QUEUE51 START $(date)"

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q4=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PH=~/LLM/models--microsoft--Phi-4-mini-instruct
Q35=~/LLM/models--Qwen--Qwen3.5-2B

run_t () {  # model run tag bin dtype
  local M=$1 RUN=$2 TAG=$3 BIN=$4 DT=$5
  out=results/transport/$TAG
  [ -f "$out/transport.npz" ] && return
  $BIN -m chaos_llm.run_transport --model "$M" --run "$RUN" --positions all \
    --dirs 8 --r 0.1 --layers 1,7,14,21,last --dtype "$DT" --out "$out" \
    || echo "FAILED transport $TAG"
  sleep 10
}

run_t $DS results/big4/a1_all_layers/prompts deepseek $PY39 float32
run_t $Q4 results/models/qwen3_4b/a1 qwen3_4b $PY float16
run_t $PH results/models/phi4mini/a1 phi4mini $PY float16
run_t $Q35 results/models/qwen3_5/a1 qwen3_5 $PY311 float16
# second prompt for DeepSeek and Qwen3
run_t $DS results/big4/a1_all_layers/p2 deepseek_p2 $PY39 float32
run_t $Q4 results/models/qwen3_4b/a1_p2 qwen3_4b_p2 $PY float16

echo "QUEUE51 DONE $(date)"
touch results/QUEUE51_DONE
