#!/bin/bash
set -u
cd /home/grego/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/grego/LLM/venv5/bin/python

wait_mem () { # need GB
  local need=$1
  while true; do
    local av=$(free -g | awk '/Mem:/{print $7}')
    [ "$av" -ge "$need" ] && return 0
    echo "waiting for memory: ${av}G < ${need}G $(date)"
    sleep 600
  done
}

run () { # tag model fixed
  local tag=$1 model=$2 fixed=$3
  [ -f "results/update/dense14_$tag/a1.npz" ] && { echo "skip $tag"; return; }
  wait_mem 66
  echo "=== dense14 $tag $(date)"
  $PY -m chaos_llm.run_a1 --model "$model" --dtype float32 --prompt @prompts.txt \
    --window 100 --max-new-tokens 100 --num-directions 64 --dense \
    --radii "0.01,0.03,0.1,0.3" --layers "0" --batch-dirs 1 --inbatch-ref \
    --fixed-tokens-from "$fixed" --out "results/update/dense14_$tag" || echo "FAILED $tag"
}

run qwen314 /home/grego/LLM/models/Qwen3-14B results_big/qwen314/a1_inter_w0
run phi4 /home/grego/LLM/models/phi-4 results_big/phi4/a1_inter_w0
run ds14 /home/grego/LLM/models/DeepSeek-R1-Distill-Qwen-14B results_big/ds14/a1_inter_w100
echo "SPARK WAIT QUEUE DONE $(date)"
