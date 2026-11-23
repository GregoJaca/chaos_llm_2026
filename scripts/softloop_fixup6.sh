#!/bin/bash
# Fixup 6: robustness — window grid at K=64, dense band tail, seed replication.
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "FIXUP6 START $(date)"

# window grid at higher direction count (the w100 comparison at paper-grade K)
for T in 0.1 0.2 0.5 1.0; do
  D=results/softloop_w100dense/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== w100 K=64 T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $DS --steps 600 --temp $T \
      --dirs 64 --r 0.03 --save-layers 27 --out $D --window 100
  fi
  if [ ! -f $D/replay/softloop.npz ] && [ -f $D/softloop.npz ]; then
    python3 -m chaos_llm.run_softloop --model $DS --steps 600 --temp $T \
      --mode replay --replay-file $D/softloop.npz \
      --dirs 64 --r 0.03 --save-layers 27 --out $D/replay --window 100
  fi
done

# dense band tail on the main prompt
for T in 0.5 0.7 1.4; do
  D=results/softloop_dense/ds15/pprompts/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== dense128 T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $DS --steps 600 --temp $T \
      --dirs 128 --r 0.03 --save-layers 27 --out $D --window 4096
  fi
done

# seed replication at the two key temperatures
for T in 0.2 1.0; do
  D=results/softloop_seed70/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== seed70 T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $DS --steps 600 --temp $T \
      --dirs 64 --r 0.03 --seed 70 --save-layers 27 --out $D --window 4096
  fi
done

echo "FIXUP6 DONE $(date)"
