#!/bin/bash
# Mamba grid, run immediately (resume-safe; queue10 will skip what exists).
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MM=~/LLM/models--state-spaces--mamba-130m-hf
TEMPS="1e-9,0.05,0.1,0.15,0.2,0.3,0.5,0.7,1.0,1.4,2.0"
echo "MAMBA NOW START $(date)"
for T in $(echo $TEMPS | tr ',' ' '); do
  D=results/softloop/mamba130/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== mamba loop T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $MM --steps 600 --temp $T \
      --dirs 8 --r 0.1 --save-layers 0,6,12,18,23 --out $D --window 0 \
      || echo "MAMBA FAILED T=$T"
  fi
  if [ ! -f $D/replay/softloop.npz ] && [ -f $D/softloop.npz ]; then
    echo "=== mamba replay T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $MM --steps 600 --temp $T \
      --mode replay --replay-file $D/softloop.npz \
      --dirs 8 --r 0.1 --save-layers 0,6,12,18,23 --out $D/replay --window 0 \
      || echo "MAMBA REPLAY FAILED T=$T"
  fi
done
echo "MAMBA NOW DONE $(date)"
