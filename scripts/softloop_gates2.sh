#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "GATES2 START $(date)"
for T in 1e-9 0.05 0.2 0.5 1.0; do
  D=results/softloop_top2/T$T
  if [ ! -f $D/gates/gates.npz ] && [ -f $D/softloop.npz ]; then
    echo "=== gates T=$T $(date)"
    python3 -m chaos_llm.run_gates --model $M --loop $D/softloop.npz --out $D/gates \
      || echo "GATES FAILED T=$T"
  fi
done
echo "GATES2 DONE $(date)"
