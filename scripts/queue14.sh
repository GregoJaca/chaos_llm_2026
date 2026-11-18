#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "queue1[23].sh" > /dev/null; do sleep 20; done
echo "QUEUE14 START $(date)"
python3 -m chaos_llm.run_after_flip --run results/a1_ds_w100_layers \
  --free-run results/a1_ds_w100_layers/free_run_large --radii 0.5,1.0 \
  --dirs 0,1,2,5,10,11,12,13,17,23 --out results/after_flip || echo "FAILED after_flip"
echo "QUEUE14 DONE $(date)"
