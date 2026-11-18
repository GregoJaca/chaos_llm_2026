#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "QUEUE3 START $(date)"

# 1. Current-state kicks: autoregressive validation of the latent-kick claim
python3 -m chaos_llm.run_state_freerun --run results/a1_ds_w100_layers \
  --steps 16,50 --dirs 24 --r 0.1 --seed 1000 --out results/state_freerun_r0.1 || echo "FAILED state_freerun"

# 2. Residual injection at layer 14 (fixed hook)
python3 -m chaos_llm.run_state2 --run results/a1_ds_w100_layers \
  --steps 16,50,100,150 --dirs 16 --radii 0.01,0.03 \
  --layers 14,21,27 --inject-layer 14 --batch-dirs 2 --out results/state_deep_L14 || echo "FAILED state_L14"

# 3. Free-run at larger radii on the second prompt (for flip statistics)
python3 -m chaos_llm.run_thresholds --run results/a1_cyber_w100 --radii 0.5,1.0 \
  --chunk 6 --out results/a1_cyber_w100/free_run_large || echo "FAILED cyber freerun"

# 4. Teacher-forced vs autoregressive on the second prompt
python3 -m chaos_llm.run_tf_predict --run results/a1_cyber_w100 \
  --free-run results/a1_cyber_w100/free_run_large --radii 0.5,1.0 \
  --chunk 6 --out results/tf_predict_cyber || echo "FAILED tf_cyber"

# 5. Another batch of long trajectories (6 more prompts)
python3 -m chaos_llm.collect_trajectories --prompts prompts_more2.txt --window 4096 \
  --max-new-tokens 1000 --layers 7,14,21,27 --out results/long_traj3 || echo "FAILED long_traj3"

echo "QUEUE3 DONE $(date)"
