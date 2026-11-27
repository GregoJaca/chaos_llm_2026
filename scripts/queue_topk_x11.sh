#!/bin/bash
# Waits for the v3 campaign, then runs the top-k rotation / alignment probe on
# every small model and the controls.
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY0=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311/bin/python

while pgrep -f queue_v3_x11.sh > /dev/null; do
  echo "waiting for v3 x11 $(date)"; sleep 300
done

run () { # tag model py dtype rundir
  local tag=$1 model=$2 py=$3 dt=$4 rd=$5
  local D=results/update/topk_$tag
  [ -f "$D/topk.npz" ] && { echo "skip $D"; return; }
  echo "=== topk $tag $(date)"
  $py -m chaos_llm.run_topk --model "$model" --run "$rd" --tokens 16,80,140 \
    --k 10 --iters 10 --dirs 8 --r 0.1 --dtype "$dt" --out "$D" || echo "FAILED topk $tag"
}

run ds15     ~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B $PY0 float32 results/rstar/deepseek/p1
run qwen25   ~/LLM/models--Qwen--Qwen2.5-3B-Instruct               $PY  float32 results/rstar/qwen25_3b/p1
run qwen34   ~/LLM/models--Qwen--Qwen3-4B-Instruct-2507            $PY  float16 results/rstar/qwen3_4b/p1
run qwen35   ~/LLM/models--Qwen--Qwen3.5-2B                        $PY311 float32 results/rstar/qwen3_5/p1
run phi      ~/LLM/models--microsoft--Phi-4-mini-instruct          $PY  float16 results/rstar/phi4mini/p1
run pythia   ~/LLM/models--EleutherAI--pythia-410m                 $PY  float32 results/ctrl/pythia410/a1i_txt
run randqwen ~/LLM/models--randqwen-g1.0                           $PY0 float32 results/ctrl/randqwen_g1.0/a1i_txt
echo "TOPK X11 DONE $(date)"
