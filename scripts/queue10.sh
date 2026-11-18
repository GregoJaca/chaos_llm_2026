#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "QUEUE10 START $(date)"

# 1. Layer-resolved amplification (fixed)
python3 -m chaos_llm.run_layer_probe --run results/a1_ds_w100_layers \
  --layers 0,2,4,7,10,14,18,21,24,27 --dirs 6 --r 0.05 --pos 50 \
  --out results/layer_probe_pos50 || echo "FAILED probe50"
python3 -m chaos_llm.run_layer_probe --run results/a1_ds_w100_layers \
  --layers 0,2,4,7,10,14,18,21,24,27 --dirs 6 --r 0.05 --pos 16 \
  --out results/layer_probe_pos16 || echo "FAILED probe16"
python3 -m chaos_llm.run_layer_probe --run results/a1_ds_w100_layers \
  --layers 0,7,14,21,27 --dirs 6 --r 0.2 --pos 50 \
  --out results/layer_probe_r0.2 || echo "FAILED probe_r02"

# 2. Window sweep on a second prompt (generality of the window regime)
for w in 50 100 200 400 800 4096; do
  echo "=== cyber window sweep $w $(date)"
  python3 -m chaos_llm.run_a1 --model $M --prompt @prompts_cyber.txt --window $w \
    --max-new-tokens 300 --fixed-tokens-from results/long_traj/cybersecurity.npz \
    --num-directions 10 --radii 0.03,0.1,0.3 --layers 7,14,21,27 --batch-dirs 1 \
    --out results/window_sweep_cyber/w$w || echo "FAILED wcyber $w"
done

echo "QUEUE10 DONE $(date)"
