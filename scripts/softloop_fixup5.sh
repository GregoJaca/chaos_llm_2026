#!/bin/bash
# Fixup 5: multi-prompt dense survival statistics (the frozen prompt set)
# and long-horizon runs to resolve the low-T censored question.
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B

while pgrep -f "bash scripts/softloop_fixup4.sh" > /dev/null; do sleep 120; done
echo "FIXUP5 START $(date)"

PROMPTS="prompts.txt prompts_childhood.txt prompts_cyber.txt prompts_more.txt prompts_more2.txt prompts_more3.txt prompts_d2.txt prompts_flush_extra.txt prompts_lengths.txt prompts_long2.txt"
for P in $PROMPTS; do
  [ -f $P ] || continue
  for T in 0.1 0.2 1.0; do
    D=results/softloop_dense10/ds15/p${P%.txt}/T$T
    if [ ! -f $D/softloop.npz ]; then
      echo "=== dense10 p=$P T=$T $(date)"
      python3 -m chaos_llm.run_softloop --model $DS --steps 600 --temp $T \
        --prompt @$P --dirs 64 --r 0.03 --save-layers 27 \
        --out $D --window 4096 || echo "FAILED $P T=$T"
    fi
  done
done

# long horizon: does the T=0.05-0.15 loop always separate eventually?
for T in 0.05 0.1 0.15; do
  D=results/softloop_long/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== long T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $DS --steps 3000 --temp $T \
      --dirs 32 --r 0.03 --save-layers 27 --out $D --window 4096
  fi
done

echo "FIXUP5 DONE $(date)"
