#!/bin/bash
# Chain 7: mamba grid (fixed cache API) + optional Qwen3.5-2B hybrid attempt.
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MM=~/LLM/models--state-spaces--mamba-130m-hf
Q35=~/LLM/models--Qwen--Qwen3.5-2B
TEMPS="1e-9,0.05,0.1,0.15,0.2,0.3,0.5,0.7,1.0,1.4,2.0"

while pgrep -f "bash scripts/softloop_queue6.sh" > /dev/null; do sleep 60; done
echo "SOFTLOOP7 START $(date)"

for T in $(echo $TEMPS | tr ',' ' '); do
  D=results/softloop/mamba130/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== mamba loop T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $MM --steps 600 --temp $T \
      --dirs 8 --r 0.03 --save-layers 0,6,12,18,23 --out $D --window 0 \
      || echo "MAMBA FAILED T=$T"
  fi
  if [ ! -f $D/replay/softloop.npz ] && [ -f $D/softloop.npz ]; then
    echo "=== mamba replay T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $MM --steps 600 --temp $T \
      --mode replay --replay-file $D/softloop.npz \
      --dirs 8 --r 0.03 --save-layers 0,6,12,18,23 --out $D/replay --window 0 \
      || echo "MAMBA REPLAY FAILED T=$T"
  fi
done

# Qwen3.5-2B hybrid attempt (venv311, transformers 5.x) -- exploratory
for T in 1e-9 0.5 1.0; do
  D=results/softloop/qwen35_2b/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== qwen35 loop T=$T $(date)"
    ~/LLM/venv311/bin/python -m chaos_llm.run_softloop --model $Q35 --steps 300 \
      --temp $T --dirs 8 --r 0.1 --save-layers 0,12,23 --out $D --window 0 \
      || echo "QWEN35 FAILED T=$T"
  fi
done

echo "SOFTLOOP7 DONE $(date)"
