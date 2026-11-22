#!/bin/bash
# Chain 4: multi-prompt ds15 grid (orbit heterogeneity), r=0.1 comparison,
# and the Qwen3.5-2B hybrid attempt.
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
TEMPS="0.1,0.2,0.3,0.5,0.7,1.0"

while pgrep -f "bash scripts/softloop_queue3.sh" > /dev/null; do sleep 60; done
echo "SOFTLOOP4 START $(date)"

# 1. orbit heterogeneity: two more prompts, mid T grid
for P in prompts_childhood.txt prompts_cyber.txt; do
  for T in $(echo $TEMPS | tr ',' ' '); do
    D=results/softloop/ds15_alt/p${P%.txt}/T$T
    if [ ! -f $D/softloop.npz ]; then
      echo "=== alt prompt $P T=$T $(date)"
      python3 -m chaos_llm.run_softloop --model $M --steps 600 --temp $T \
        --prompt @$P --dirs 8 --r 0.03 --save-layers 0,7,14,21,27 --out $D --window 4096
    fi
  done
done

# 2. radius comparison at T=1.0 with the queue batch shape (9 rows)
for R in 0.1 0.3; do
  D=results/softloop/ds15_r01/T1.0_r$R
  if [ ! -f $D/softloop.npz ]; then
    echo "=== r-check T=1.0 r=$R $(date)"
    python3 -m chaos_llm.run_softloop --model $M --steps 600 --temp 1.0 \
      --dirs 8 --r $R --save-layers 0,7,14,21,27 --out $D --window 4096
  fi
done

echo "SOFTLOOP4 DONE $(date)"
