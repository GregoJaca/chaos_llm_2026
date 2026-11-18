#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "QUEUE5 START $(date)"

# 1. More autoregressive validation of single-token kicks (r=0.1)
python3 -m chaos_llm.run_state_freerun --run results/a1_ds_w100_layers \
  --steps 8,32,75,100,150,200 --dirs 24 --r 0.1 --seed 1000 --out results/state_freerun_r0.1b || echo "FAILED sf1"

# 2. Same at a larger kick size (r=0.3)
python3 -m chaos_llm.run_state_freerun --run results/a1_ds_w100_layers \
  --steps 8,16,32,50,75,100,150,200 --dirs 24 --r 0.3 --seed 1000 --out results/state_freerun_r0.3 || echo "FAILED sf2"

# 3. Another batch of long trajectories
python3 -m chaos_llm.collect_trajectories --prompts prompts_more3.txt --window 4096 \
  --max-new-tokens 1000 --layers 7,14,21,27 --out results/long_traj4 || echo "FAILED lt4"

echo "QUEUE5 DONE $(date)"
