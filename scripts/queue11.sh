#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "QUEUE11 START $(date)"

# 1. Finite-time rate statistics across prompts and windows (same protocol)
for p in prompts_long.txt prompts_long2.txt; do
  for w in 100 4096; do
    python3 -m chaos_llm.run_a1 --model $M --prompt @$p --window $w \
      --max-new-tokens 300 --num-directions 10 --radii 0.03,0.1 \
      --layers 7,14,21,27 --batch-dirs 1 \
      --out results/rate_stats/${p%.txt}_w$w || echo "FAILED rate $p $w"
  done
done

# 2. Layer probes at more positions (statistics for N2)
python3 -m chaos_llm.run_layer_probe --run results/a1_ds_w100_layers \
  --layers 0,7,14,21,27 --dirs 6 --r 0.05 --pos 32 \
  --out results/layer_probe_pos32 || echo "FAILED probe32"
python3 -m chaos_llm.run_layer_probe --run results/a1_ds_w100_layers \
  --layers 0,7,14,21,27 --dirs 6 --r 0.05 --pos 100 \
  --out results/layer_probe_pos100 || echo "FAILED probe100"

# 3. Shallow layers for the window-flush picture (layers 1,2,3)
for w in 100 400; do
  python3 -m chaos_llm.run_a1 --model $M --prompt @prompts.txt --window $w \
    --max-new-tokens 300 --fixed-tokens-from results/long_traj/interstellar_travel.npz \
    --num-directions 6 --radii 0.1 --layers 1,2,3,7 --batch-dirs 1 \
    --out results/window_flush/w$w || echo "FAILED flush $w"
done

echo "QUEUE11 DONE $(date)"
