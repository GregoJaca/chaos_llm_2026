#!/bin/bash
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY0=python3
PY=~/LLM/venv_tf/bin/python

run () { # model py dtype tag W layers
  local model=$1 py=$2 dt=$3 tag=$4 W=$5 layers=$6
  [ -f "results/update/attn2_${tag}_w$W/attn_frac.npz" ] && { echo "skip $tag $W"; return; }
  echo "=== attn $tag W=$W $(date)"
  $py -m chaos_llm.run_attention_frac --model "$model" --dtype "$dt" --window "$W" \
    --max-new-tokens 200 --layers "$layers" --out "results/update/attn2_${tag}_w$W" || echo "FAILED $tag $W"
}

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q25=~/LLM/models--Qwen--Qwen2.5-3B-Instruct
PHI=~/LLM/models--microsoft--Phi-4-mini-instruct
RQ=~/LLM/models--randqwen-g1.0

for W in 50 100 200 4096; do run "$DS" "$PY0" float32 ds15 "$W" "0,7,14,21,27"; done
run "$Q25" "$PY" float32 qwen25 100 "0,7,14,21,35"
run "$PHI" "$PY" float16 phi 100 "0,7,14,21,31"
run "$RQ"  "$PY" float32 randqwen 100 "0,7,14,21,27"
echo "ATTN QUEUE DONE $(date)"

