#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export CHAOS_DTYPE=float16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv311b/bin/python
M=~/LLM/models--Qwen--Qwen3.5-2B
echo "Q35 SUITE start $(date)"

$PY scripts/r_calib.py --model $M --dtype float16 --tokens 100 --dirs 4 2>&1 | tail -8

$PY -m chaos_llm.run_a1 --model $M --prompt @prompts.txt --window 100 \
  --max-new-tokens 200 --num-directions 8 --radii 0.1,0.3 \
  --layers 0,4,8,12,16,20,23 --batch-dirs 1 --out results/models/qwen3_5/a1 \
  || echo "FAILED q35 a1"

if [ -f results/models/qwen3_5/a1/a1.npz ]; then
  $PY -m chaos_llm.run_state2 --model $M --run results/models/qwen3_5/a1 \
    --steps 16,50,100 --dirs 8 --radii 0.1 --layers 12,20,23 --inject-layer 0 \
    --batch-dirs 1 --out results/models/qwen3_5/state || echo "FAILED q35 state"
  $PY -m chaos_llm.collect_trajectories --model $M --prompts prompts_long2.txt \
    --window 4096 --max-new-tokens 500 --layers 4,12,20,23 \
    --out results/models/qwen3_5/traj || echo "FAILED q35 traj"
fi
echo "Q35 SUITE done $(date)"
