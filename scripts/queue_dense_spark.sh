#!/bin/bash
set -u
cd /home/grego/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
PY=/home/grego/LLM/venv5/bin/python

run () { # model tag fixed
  local model=$1 tag=$2 fixed=$3
  [ -f "results/update/dense_$tag/a1.npz" ] && { echo "skip $tag"; return; }
  echo "=== dense $tag $(date)"
  $PY -m chaos_llm.run_a1 --model "$model" --dtype float32 --prompt @prompts.txt \
    --window 100 --max-new-tokens 300 --num-directions 128 --dense \
    --radii "0.03,0.1,0.3" --layers "0" --batch-dirs 1 --inbatch-ref \
    --fixed-tokens-from "$fixed" --out "results/update/dense_$tag" || echo "FAILED $tag"
}

run /home/grego/LLM/models/Qwen3-14B qwen314 results_big/qwen314/a1_inter_w0
run /home/grego/LLM/models/phi-4 phi4 results_big/phi4/a1_inter_w0
run /home/grego/LLM/models/DeepSeek-R1-Distill-Qwen-14B ds14 results_big/ds14/a1_inter_w100
echo "SPARK DENSE DONE $(date)"
