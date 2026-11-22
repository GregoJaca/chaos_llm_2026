#!/bin/bash
# Chain 5: top-2 recording for key loop orbits + gate extraction + the
# two-token gate prediction check (predicted sigma vs recorded sig_read).
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B

while pgrep -f "bash scripts/softloop_queue4.sh" > /dev/null; do sleep 60; done
echo "SOFTLOOP5 START $(date)"

for T in 1e-9 0.05 0.2 0.5 1.0; do
  D=results/softloop_top2/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== loop+top2 T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $M --steps 600 --temp $T \
      --dirs 8 --r 0.03 --save-layers 27 --out $D --window 4096
  fi
  if [ -f $D/softloop.npz ] && [ ! -f $D/gates/gates.npz ]; then
    echo "=== gates T=$T $(date)"
    python3 -m chaos_llm.run_gates --model $M --loop $D/softloop.npz --out $D/gates
  fi
done

echo "SOFTLOOP5 DONE $(date)"
