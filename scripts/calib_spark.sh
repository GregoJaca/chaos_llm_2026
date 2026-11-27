#!/bin/bash
set -u
cd /home/grego/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for K in 32 64; do
  echo "=== K=$K 100 steps $(date)"
  /usr/bin/time -f "K=$K wall %e s" /home/grego/LLM/venv5/bin/python -m chaos_llm.run_softloop \
    --model /home/grego/LLM/models/Qwen3-14B --steps 100 --temp 1.0 --dirs "$K" --r 0.1 \
    --feedback soft --save-layers 10,20,30,39 --dtype float16 --window 4096 \
    --out /tmp/calib14_k$K 2>&1 | tail -n 3
  free -g | head -2
done
