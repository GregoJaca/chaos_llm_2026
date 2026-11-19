#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "QUEUE24 START $(date)"

# 1. Flush law on two more prompts (10 windows, 6 layers)
for i in 1 2; do
  sed -n "${i}p" prompts_more2.txt > /tmp/FL_$i.txt
  for w in 60 80 100 150 200 300 500; do
    out=results/big4/flush_more/p${i}_w${w}
    [ -f "$out/a1.npz" ] && continue
    python3 -m chaos_llm.run_a1 --model $M --prompt @/tmp/FL_$i.txt --window $w \
      --max-new-tokens 600 --num-directions 6 --radii 0.1 --layers 1,2,3,4,5,6 \
      --batch-dirs 1 --out "$out" || echo "FAILED flush $i $w"
  done
done
echo "flush_more done $(date)"

# 2. 4000-token teacher-forced with more directions (8) for two prompts
for i in 1 2; do
  src=results/big4/traj4k/L2_$i.npz
  [ -f "$src" ] || continue
  out=results/big4/a1_4k_d8/L2_$i
  [ -f "$out/a1.npz" ] && continue
  python3 -m chaos_llm.run_a1_long --prompt @prompts_long2.txt --window 100 \
    --max-new-tokens 4000 --fixed-tokens-from "$src" --num-directions 8 \
    --radii 0.1 --layers 7,14,21,27 --out "$out" || echo "FAILED a1_4k_d8 $i"
done

echo "QUEUE24 DONE $(date)"
