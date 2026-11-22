#!/bin/bash
# Chain 8: Qwen2.5-3B grid rerun with the fixed fp16 feedback dtype.
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
QM=~/LLM/models--Qwen--Qwen2.5-3B-Instruct
TEMPS="1e-9,0.05,0.1,0.15,0.2,0.3,0.5,0.7,1.0,1.4,2.0"

while pgrep -f "bash scripts/softloop_queue7.sh" > /dev/null; do sleep 60; done
echo "SOFTLOOP8 START $(date)"

for T in $(echo $TEMPS | tr ',' ' '); do
  D=results/softloop/qwen25_3b/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== qwen25 loop T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $QM --steps 600 --temp $T \
      --dirs 8 --r 0.1 --save-layers 0,9,18,27,35 --out $D --window 4096 \
      || echo "QWEN25 FAILED T=$T"
  fi
  if [ ! -f $D/replay/softloop.npz ] && [ -f $D/softloop.npz ]; then
    echo "=== qwen25 replay T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $QM --steps 600 --temp $T \
      --mode replay --replay-file $D/softloop.npz \
      --dirs 8 --r 0.1 --save-layers 0,9,18,27,35 --out $D/replay --window 4096 \
      || echo "QWEN25 REPLAY FAILED T=$T"
  fi
done

echo "SOFTLOOP8 DONE $(date)"
