#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python

while pgrep -f "queue33.sh" > /dev/null; do sleep 20; done
echo "QUEUE34 START $(date)"

# ---------------- 1. dimensionality datasets ----------------
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
echo "DIM DONE $(date)"

# ---------------- 2. layer probe, three prompts per model ----------------
run_probe () {  # model run tag layers dirs r py dtype extra
  local M=$1 RUN=$2 TAG=$3 LAYERS=$4 DIRS=$5 R=$6 PYBIN=$7 DT=$8
  for pos in 16 50; do
    out=results/probe/$TAG/pos$pos
    [ -f "$out/layer_probe.npz" ] && continue
    $PYBIN -m chaos_llm.run_layer_probe --model "$M" --run "$RUN" --out "$out" \
      --layers "$LAYERS" --dirs "$DIRS" --r "$R" --pos $pos --dtype "$DT" \
      || echo "FAILED probe $TAG pos$pos"
  done
}

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q4=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PH=~/LLM/models--microsoft--Phi-4-mini-instruct
Q35=~/LLM/models--Qwen--Qwen3.5-2B

run_probe $DS results/big4/a1_all_layers/prompts deepseek_ds1 0,1,2,3,4,5,7,10,14,18,21,24,27 10 0.1 $PY39 float32
run_probe $Q4 results/models/qwen3_4b/a1 qwen3_4b 0,1,2,3,4,5,7,10,14,18,21,25,29,32,35 10 0.1 $PY float16
run_probe $PH results/models/phi4mini/a1 phi4mini 0,1,2,3,4,5,7,10,14,18,21,25,28,31 10 0.1 $PY float16
run_probe $Q35 results/models/qwen3_5/a1 qwen3_5 0,1,2,3,4,6,8,12,16,20,23 8 0.1 $PY311 float16

# second prompt for DeepSeek and Qwen3 (probe depends on the trajectory)
run_probe $DS results/big4/a1_all_layers/p1 deepseek_ds_p1 0,1,2,3,4,5,7,10,14,18,21,24,27 8 0.1 $PY39 float32
run_probe $Q4 results/models/qwen3_4b/a1_p1 qwen3_4b_p1 0,1,2,3,4,5,7,10,14,18,21,25,29,32,35 8 0.1 $PY float16

echo "QUEUE34 DONE $(date)"
