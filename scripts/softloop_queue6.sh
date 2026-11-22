#!/bin/bash
# Chain 6: re-run the ds15 temperature grid with the corrected sigma estimator
# (the original ds15 group recorded a broken sig_read; all other fields were
# unaffected). Output to ds15v2.
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
TEMPS="1e-9,0.05,0.1,0.15,0.2,0.3,0.5,0.7,1.0,1.4,2.0"

while pgrep -f "bash scripts/softloop_queue5.sh" > /dev/null; do sleep 60; done
echo "SOFTLOOP6 START $(date)"

for T in $(echo $TEMPS | tr ',' ' '); do
  D=results/softloop/ds15v2/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== loop T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $M --steps 600 --temp $T \
      --dirs 8 --r 0.03 --save-layers 0,7,14,21,27 --out $D --window 4096
  fi
  if [ ! -f $D/replay/softloop.npz ] && [ -f $D/softloop.npz ]; then
    echo "=== replay T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $M --steps 600 --temp $T \
      --mode replay --replay-file $D/softloop.npz \
      --dirs 8 --r 0.03 --save-layers 0,7,14,21,27 --out $D/replay
  fi
done

echo "SOFTLOOP6 DONE $(date)"
