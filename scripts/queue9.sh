#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "QUEUE9 START $(date)"

# 1. Layer-resolved amplification: inject into each layer, read every layer
python3 -m chaos_llm.run_layer_probe --run results/a1_ds_w100_layers \
  --layers 0,2,4,7,10,14,18,21,24,27 --dirs 6 --r 0.05 --pos 50 \
  --out results/layer_probe_pos50 || echo "FAILED probe50"
python3 -m chaos_llm.run_layer_probe --run results/a1_ds_w100_layers \
  --layers 0,2,4,7,10,14,18,21,24,27 --dirs 6 --r 0.05 --pos 16 \
  --out results/layer_probe_pos16 || echo "FAILED probe16"

# 2. Window sweep at fixed tokens (same inputs): does the regime change with W?
for w in 50 100 200 400 800 4096; do
  echo "=== window sweep $w $(date)"
  python3 -m chaos_llm.run_a1 --model $M --prompt @prompts_long2.txt --window $w \
    --max-new-tokens 300 --fixed-tokens-from results/long_traj_2k/ocean_currents.npz \
    --num-directions 10 --radii 0.03,0.1,0.3 --layers 7,14,21,27 --batch-dirs 1 \
    --out results/window_sweep/w$w || echo "FAILED wsweep $w"
done

# 3. More radii for the flip-time statistics (stable-chaos scaling)
python3 -m chaos_llm.run_thresholds --run results/a1_ds_w100_layers \
  --radii 0.15,0.25,0.4 --chunk 6 --out results/a1_ds_w100_layers/free_run_extra || echo "FAILED extra"

echo "QUEUE9 DONE $(date)"
