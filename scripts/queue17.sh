#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "QUEUE17 START $(date)"

# 1. Flush at more windows (growth summary needs more points)
for w in 250 400; do
  python3 -m chaos_llm.run_a1 --model $M --prompt @prompts.txt --window $w \
    --max-new-tokens 500 --fixed-tokens-from results/long_traj/interstellar_travel.npz \
    --num-directions 4 --radii 0.1 --layers 1,2,3,4,6,7 --batch-dirs 1 \
    --out results/flush_sweep/w$w || echo "FAILED flush $w"
done

# 2. Flush with a longer horizon (800 tokens) to catch deeper layers
python3 -m chaos_llm.run_a1 --model $M --prompt @prompts.txt --window 100 \
  --max-new-tokens 800 --fixed-tokens-from results/long_traj_2k/ocean_currents.npz \
  --num-directions 3 --radii 0.1 --layers 1,2,3,4,5,6 --batch-dirs 1 \
  --out results/flush_long || echo "FAILED flush_long"

# 3. More prompts for the window-growth comparison (W=100 and W=50)
for i in 4 5 6; do
  sed -n "${i}p" prompts_more.txt > /tmp/p_g_$i.txt
  python3 -m chaos_llm.run_a1 --model $M --prompt @/tmp/p_g_$i.txt --window 100 \
    --max-new-tokens 300 --num-directions 8 --radii 0.1 --layers 7,14,21,27 --batch-dirs 1 \
    --out results/rate_prompts/p$i || echo "FAILED rate p$i"
  python3 -m chaos_llm.run_a1 --model $M --prompt @/tmp/p_g_$i.txt --window 50 \
    --max-new-tokens 300 --num-directions 8 --radii 0.1 --layers 7,14,21,27 --batch-dirs 1 \
    --out results/rate_w50/p$i || echo "FAILED w50 p$i"
done

echo "QUEUE17 DONE $(date)"
