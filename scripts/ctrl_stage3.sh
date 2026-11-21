#!/bin/bash
# Control study, stage 3: in-batch Jacobian measurements for Mamba.
# Waits for stage 2 to release the GPU, then runs the Jacobian suite.
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv_tf/bin/python
M130=~/LLM/models--state-spaces--mamba-130m-hf
M370=~/LLM/models--state-spaces--mamba-370m-hf

echo "=== stage3 start $(date)"
while pgrep -f ctrl_stage2.sh > /dev/null 2>&1; do sleep 60; done
echo "stage2 finished, GPU free $(date)"

if [ ! -f results/ctrl/mamba130/jac/jac.npz ]; then
  echo "--- jac mamba130 $(date)"
  $PY -m chaos_llm.run_mamba_jacobian --model "$M130" \
    --tokens-from results/ctrl/mamba130/a1i_txt/a1.npz \
    --t0-list 80,150,220 --horizon 200 \
    --out results/ctrl/mamba130/jac || echo "FAILED jac130"
fi

if [ ! -f results/ctrl/mamba370/jac/jac.npz ]; then
  echo "--- jac mamba370 $(date)"
  $PY -m chaos_llm.run_mamba_jacobian --model "$M370" \
    --tokens-from results/ctrl/mamba370/a1i_txt/a1.npz \
    --t0-list 80,150,220 --horizon 200 \
    --out results/ctrl/mamba370/jac || echo "FAILED jac370"
fi

echo "=== stage3 done $(date)"
