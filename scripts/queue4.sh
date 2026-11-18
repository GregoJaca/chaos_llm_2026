#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "queue3.sh" > /dev/null; do sleep 20; done
echo "QUEUE4 START $(date)"
python3 -m chaos_llm.run_state_freerun --run results/a1_ds_w100_layers \
  --steps 16,50 --dirs 24 --r 0.1 --seed 1000 --out results/state_freerun_r0.1 || echo "FAILED state_freerun"
# prompt-perturbation sign check on the second prompt for TF-vs-AR generality
python3 -m chaos_llm.run_thresholds --run results/a1_childhood_w100 --radii 0.5,1.0 \
  --chunk 6 --out results/a1_childhood_w100/free_run_large || echo "FAILED childhood freerun"
python3 -m chaos_llm.run_tf_predict --run results/a1_childhood_w100 \
  --free-run results/a1_childhood_w100/free_run_large --radii 0.5,1.0 \
  --chunk 6 --out results/tf_predict_childhood || echo "FAILED tf_childhood"
echo "QUEUE4 DONE $(date)"
