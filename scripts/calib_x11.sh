#!/bin/bash
# Calibration: how slow is a 64-direction soft loop vs 8 directions?
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for K in 8 64; do
  echo "=== K=$K 100 steps $(date)"
  /usr/bin/time -f "K=$K wall %e s" python3 -m chaos_llm.run_softloop \
    --model ~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B \
    --steps 100 --temp 1.0 --dirs $K --r 0.03 --feedback soft \
    --save-layers 7,14,21,27 --dtype float16 --window 4096 \
    --out /tmp/calib_k$K 2>&1 | tail -n 3
done
