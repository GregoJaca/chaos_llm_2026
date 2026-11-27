#!/bin/bash
# Waits for the v3 Spark campaign, then runs the top-k rotation / alignment
# probe on the 14B models and Phi-4.
set -u
cd /home/grego/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/grego/LLM/venv5/bin/python

while pgrep -f queue_v3_spark.sh > /dev/null; do
  echo "waiting for v3 spark $(date)"; sleep 300
done

run () { # tag model rundir
  local tag=$1 model=$2 rd=$3
  local D=results/update/topk_$tag
  [ -f "$D/topk.npz" ] && { echo "skip $D"; return; }
  echo "=== topk $tag $(date)"
  $PY -m chaos_llm.run_topk --model "$model" --run "$rd" --tokens 50,140 \
    --k 10 --iters 8 --dirs 6 --r 0.1 --dtype float16 --out "$D" || echo "FAILED topk $tag"
}

run qwen314 /home/grego/LLM/models/Qwen3-14B                 results_big/qwen314/a1_inter_w0
run ds14    /home/grego/LLM/models/DeepSeek-R1-Distill-Qwen-14B results_big/ds14/a1_inter_w100
run phi4    /home/grego/LLM/models/phi-4                     results_big/phi4/a1_inter_w0
run qwen35_9b /home/grego/LLM/models/Qwen3.5-9B              results_big/qwen35/a1_inter_w0
echo "TOPK SPARK DONE $(date)"
