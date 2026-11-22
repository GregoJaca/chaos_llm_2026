#!/bin/bash
# Chain 9: r=0.03 anchor at T=0.35 (for the r-scaling set), w100 rerun with the
# corrected sigma estimator (gate columns for the window figure).
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B

while pgrep -f "bash scripts/softloop_queue8.sh" > /dev/null; do sleep 60; done
echo "SOFTLOOP9 START $(date)"

D=results/softloop_rscale/T0.35/r0.03
if [ ! -f $D/softloop.npz ]; then
  echo "=== r-scale T=0.35 r=0.03 $(date)"
  python3 -m chaos_llm.run_softloop --model $M --steps 600 --temp 0.35 \
    --dirs 8 --r 0.03 --save-layers 0,14,27 --out $D --window 4096
fi

for T in 1e-9 0.05 0.1 0.2 0.5 1.0; do
  D=results/softloop/ds15_w100v2/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== w100v2 loop T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $M --steps 600 --temp $T \
      --dirs 8 --r 0.03 --save-layers 0,7,14,21,27 --out $D --window 100
  fi
done

echo "SOFTLOOP9 DONE $(date)"
