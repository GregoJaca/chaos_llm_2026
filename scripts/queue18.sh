#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "QUEUE18 START $(date)"

# 1. Continuous window scan across W/P (prompt = 65 tokens)
for w in 55 60 65 70 80 90 110 120; do
  echo "=== W scan $w $(date)"
  python3 -m chaos_llm.run_a1 --model $M --prompt @prompts.txt --window $w \
    --max-new-tokens 300 --fixed-tokens-from results/long_traj/interstellar_travel.npz \
    --num-directions 6 --radii 0.1 --layers 7,14,21,27 --batch-dirs 1 \
    --out results/window_scan/w$w || echo "FAILED scan $w"
done

# 2. Flush law on the cyber prompt
for w in 150 300; do
  python3 -m chaos_llm.run_a1 --model $M --prompt @prompts_cyber.txt --window $w \
    --max-new-tokens 500 --fixed-tokens-from results/long_traj/cybersecurity.npz \
    --num-directions 4 --radii 0.1 --layers 1,2,3,4 --batch-dirs 1 \
    --out results/flush_cyber/w$w || echo "FAILED flushcyber $w"
done

# 3. Retry the missing W=50 prompts for the growth comparison
for i in 4 5 6; do
  sed -n "${i}p" prompts_more.txt > /tmp/p_g_$i.txt
  python3 -m chaos_llm.run_a1 --model $M --prompt @/tmp/p_g_$i.txt --window 50 \
    --max-new-tokens 300 --num-directions 6 --radii 0.1 --layers 7,14,21,27 --batch-dirs 1 \
    --out results/rate_w50/p$i || echo "FAILED w50 p$i"
done

# 4. State-window control at W=100
python3 -m chaos_llm.run_state2 --run results/a1_ds_w100_layers \
  --steps 16,50,100 --dirs 12 --radii 0.03,0.1 \
  --layers 7,14,21,27 --inject-layer 0 --batch-dirs 1 --window 100 \
  --out results/state_window/w100 || echo "FAILED statew 100"

echo "QUEUE18 DONE $(date)"
