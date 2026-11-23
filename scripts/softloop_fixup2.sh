#!/bin/bash
# Fixup 2: Phi-4-mini needs fewer directions (KV-cache OOM at K=16).
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PHI=~/LLM/models--microsoft--Phi-4-mini-instruct

while pgrep -f "bash scripts/softloop_fixup.sh" > /dev/null; do sleep 120; done
echo "FIXUP2 START $(date)"

for T in 0.05 0.1 0.2 1.0; do
  D=results/softloop_val/phi4mini/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== loopval phi4mini T=$T (K=8) $(date)"
    python3 -m chaos_llm.run_softloop --model $PHI --steps 300 --temp $T \
      --dirs 8 --r 0.3 --save-layers 31 --out $D --window 4096 \
      || echo "FAILED phi loopval T=$T"
  fi
done

TEMPS="1e-9,0.05,0.1,0.15,0.2,0.3,0.5,0.7,1.0,1.4,2.0"
for T in $(echo $TEMPS | tr ',' ' '); do
  D=results/softloop/phi4mini/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== grid phi4mini T=$T (K=8) $(date)"
    python3 -m chaos_llm.run_softloop --model $PHI --steps 600 --temp $T \
      --dirs 8 --r 0.3 --save-layers 31 --out $D --window 4096 \
      || echo "FAILED phi grid T=$T"
  fi
  if [ ! -f $D/replay/softloop.npz ] && [ -f $D/softloop.npz ]; then
    echo "=== replay phi4mini T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $PHI --steps 600 --temp $T \
      --mode replay --replay-file $D/softloop.npz \
      --dirs 8 --r 0.3 --save-layers 31 --out $D/replay --window 4096
  fi
done

echo "FIXUP2 DONE $(date)"
