#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
while pgrep -f "queue12.sh" > /dev/null; do sleep 20; done
echo "QUEUE13 START $(date)"

# 1. Finite-time rate across six prompts, memory-safe batch size
for i in 1 2 3 4 5 6; do
  sed -n "${i}p" prompts_more.txt > /tmp/p_rate_$i.txt
  echo "=== rate prompt $i $(date)"
  python3 -m chaos_llm.run_a1 --model $M --prompt @/tmp/p_rate_$i.txt --window 100 \
    --max-new-tokens 300 --num-directions 8 --radii 0.1 \
    --layers 7,14,21,27 --batch-dirs 1 \
    --out results/rate_prompts/p$i || echo "FAILED rate p$i"
done

# 2. Same prompts at W=50 (is the W<P regime universal?)
for i in 1 2 3; do
  sed -n "${i}p" prompts_more.txt > /tmp/p_w50_$i.txt
  python3 -m chaos_llm.run_a1 --model $M --prompt @/tmp/p_w50_$i.txt --window 50 \
    --max-new-tokens 300 --num-directions 8 --radii 0.1 \
    --layers 7,14,21,27 --batch-dirs 1 \
    --out results/rate_w50/p$i || echo "FAILED w50 p$i"
done

# 3. 500-token horizon at W=100 (retry with batch 1)
python3 -m chaos_llm.run_a1 --model $M --prompt @prompts_long2.txt --window 100 \
  --max-new-tokens 500 --num-directions 6 --radii 0.1 \
  --layers 7,14,21,27 --batch-dirs 1 \
  --out results/rate_long500 || echo "FAILED rate_long500"

echo "QUEUE13 DONE $(date)"
