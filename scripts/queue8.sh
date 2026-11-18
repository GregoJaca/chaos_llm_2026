#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "QUEUE8 START $(date)"

# 1. Map the sensitive region for single-position perturbations
python3 -m chaos_llm.run_state_freerun --run results/a1_ds_w100_layers \
  --steps 60,65,70,80,90 --dirs 24 --r 0.3 --seed 1000 --out results/state_freerun_map1 || echo "FAILED map1"

# 2. Long response run: 800 generated tokens, fixed tokens, window 100
python3 -m chaos_llm.run_a1 --model $M --prompt @prompts_long2.txt --window 100 \
  --max-new-tokens 800 --fixed-tokens-from results/long_traj_2k/ocean_currents.npz \
  --num-directions 6 --radii 0.03,0.1,0.3 --layers 7,14,21,27 --batch-dirs 1 \
  --out results/a1_long800 || echo "FAILED a1long"

# 3. Same tokens, two windows (repeated flushes over a long horizon)
for w in 100 1000; do
python3 -m chaos_llm.run_a1 --model $M --prompt @prompts_long2.txt --window $w \
  --max-new-tokens 800 --fixed-tokens-from results/long_traj_2k/ocean_currents.npz \
  --num-directions 6 --radii 0.1 --layers 14,21,27 --batch-dirs 1 \
  --out results/window_long/w$w || echo "FAILED wlong $w"
done

echo "QUEUE8 DONE $(date)"
