#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "QUEUE START $(date)"

# 1. Window natural experiment on a FIXED token sequence (same inputs, different window)
for w in 100 150 200 300 1000; do
  echo "=== window $w $(date)"
  python3 -m chaos_llm.run_a1 --model $M --prompt @prompts.txt --window $w \
    --max-new-tokens 400 --fixed-tokens-from results/long_traj/interstellar_travel.npz \
    --num-directions 8 --radii 0.01,0.1,0.3 --layers 7,14,21,27 --batch-dirs 2 \
    --out results/window_fixed/w$w
done

# 1b. Same test on a second prompt (cyber), shorter
for w in 100 200 1000; do
  echo "=== window cyber $w $(date)"
  python3 -m chaos_llm.run_a1 --model $M --prompt @prompts_cyber.txt --window $w \
    --max-new-tokens 300 --fixed-tokens-from results/long_traj/cybersecurity.npz \
    --num-directions 8 --radii 0.01,0.1,0.3 --layers 7,14,21,27 --batch-dirs 2 \
    --out results/window_cyber/w$w
done

# 2. Free-run at intermediate radii for the main w100 run (for TF-vs-AR)
python3 -m chaos_llm.run_thresholds --run results/a1_ds_w100_layers \
  --radii 0.05,0.1,0.2,0.3 --chunk 6 --out results/a1_ds_w100_layers/free_run_mid

# 3. Teacher-forced token prediction vs autoregressive tokens
python3 -m chaos_llm.run_tf_predict --run results/a1_ds_w100_layers \
  --free-run results/a1_ds_w100_layers/free_run_mid --radii 0.05,0.1,0.2,0.3 \
  --chunk 6 --out results/tf_predict_w100

# 4. Deep current-state perturbation: input embedding injection
python3 -m chaos_llm.run_state2 --run results/a1_ds_w100_layers \
  --steps 8,16,32,50,75,100,150,200 --dirs 24 --radii 0.003,0.01,0.03,0.1 \
  --layers 7,14,21,27 --inject-layer 0 --batch-dirs 6 --out results/state_deep_emb

# 5. Deep current-state perturbation: residual injection at layer 14
python3 -m chaos_llm.run_state2 --run results/a1_ds_w100_layers \
  --steps 16,50,100,150 --dirs 16 --radii 0.01,0.03 \
  --layers 14,21,27 --inject-layer 14 --batch-dirs 4 --out results/state_deep_L14

# 6. More long trajectories (5 new prompts)
python3 -m chaos_llm.collect_trajectories --prompts prompts_more.txt --window 4096 \
  --max-new-tokens 1000 --layers 7,14,21,27 --out results/long_traj2

echo "QUEUE DONE $(date)"
