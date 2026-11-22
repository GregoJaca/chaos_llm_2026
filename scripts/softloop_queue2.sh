#!/bin/bash
# Soft-loop follow-ups: long attractor runs (for D2/RP), r-scaling check.
# Waits for the main softloop queue to finish first.
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B

while pgrep -f "bash scripts/softloop_queue.sh" > /dev/null; do sleep 60; done
echo "SOFTLOOP2 START $(date)"

# 1. long attractor runs with saved last-layer states (ref row)
for T in 0.5 1.0 2.0 1e-9; do
  D=results/softloop_states/ds15/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== states T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $M --steps 3000 --temp $T \
      --dirs 2 --r 0.03 --save-layers 27 --save-states --out $D --window 4096
  fi
done

# 2. radius scaling of the separation rate (smooth chaos: r-independent rate,
#    log(r) shift in saturation time; threshold: r-dependent)
for T in 0.35 1.0; do
  for R in 0.003 0.01 0.1; do
    D=results/softloop_rscale/T${T}/r${R}
    if [ ! -f $D/softloop.npz ]; then
      echo "=== r-scale T=$T r=$R $(date)"
      python3 -m chaos_llm.run_softloop --model $M --steps 600 --temp $T \
        --dirs 8 --r $R --save-layers 0,14,27 --out $D --window 4096
    fi
  done
done

echo "SOFTLOOP2 DONE $(date)"
