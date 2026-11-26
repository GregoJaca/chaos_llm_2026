#!/bin/bash
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY0=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311/bin/python

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q25=~/LLM/models--Qwen--Qwen2.5-3B-Instruct
Q35=~/LLM/models--Qwen--Qwen3.5-2B

run () { # model py dtype layers fixed out
  local model=$1 py=$2 dtype=$3 layers=$4 fixed=$5 out=$6
  [ -f "$out/a1.npz" ] && { echo "skip $out"; return; }
  echo "=== dense $out $(date)"
  $py -m chaos_llm.run_a1 --model "$model" --dtype "$dtype" --prompt @prompts.txt \
    --window 100 --max-new-tokens 300 --num-directions 128 --dense \
    --radii "0.03,0.1,0.3" --layers "$layers" --batch-dirs 2 --inbatch-ref \
    --fixed-tokens-from "$fixed" --out "$out" || echo "FAILED $out"
}

DSL=$(seq -s, 0 27)
Q25L=$(seq -s, 0 35)
Q35L=$(seq -s, 0 23)

run "$DS"  "$PY0"   float32 "$DSL"  results/a1_ds_w100_layers  results/update/dense_ds15
run "$Q25" "$PY"    float32 "$Q25L" results/models/qwen25_3b/a1  results/update/dense_qwen25
run "$Q35" "$PY311" float32 "$Q35L" results/models/qwen3_5/a1    results/update/dense_qwen35
echo "DENSE QUEUE DONE $(date)"
