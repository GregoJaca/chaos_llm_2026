#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "QUEUE12 START $(date)"

# 1. Finite-time rate across six different prompts (W=100, 300 tokens)
for i in 1 2 3 4 5 6; do
  sed -n "${i}p" prompts_more.txt > /tmp/p_rate_$i.txt
  echo "=== rate prompt $i $(date)"
  python3 -m chaos_llm.run_a1 --model $M --prompt @/tmp/p_rate_$i.txt --window 100 \
    --max-new-tokens 300 --num-directions 10 --radii 0.1 \
    --layers 7,14,21,27 --batch-dirs 2 \
    --out results/rate_prompts/p$i || echo "FAILED rate p$i"
done

# 2. Stable-chaos scaling on a second prompt
python3 -m chaos_llm.run_thresholds --run results/a1_cyber_w100 \
  --radii 0.15,0.25,0.4 --chunk 6 --out results/a1_cyber_w100/free_run_extra || echo "FAILED cyber extra"

# 3. Longer horizon at W=100 (500 tokens) for the rate
python3 -m chaos_llm.run_a1 --model $M --prompt @prompts_long2.txt --window 100 \
  --max-new-tokens 500 --num-directions 8 --radii 0.1 \
  --layers 7,14,21,27 --batch-dirs 1 \
  --out results/rate_long500 || echo "FAILED rate_long500"

echo "QUEUE12 DONE $(date)"
