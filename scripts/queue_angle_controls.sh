#!/bin/bash
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=python3

run () { # model tag layers fixed
  local model=$1 tag=$2 layers=$3 fixed=$4
  [ -f "results/update/angle_$tag/angle.npz" ] && { echo "skip $tag"; return; }
  echo "=== angle $tag $(date)"
  $PY -m chaos_llm.run_angle --model "$model" --dtype float32 --window 4096 \
    --fixed-tokens-from "$fixed" --num-directions 32 --radii 0.03,0.1 \
    --layers "$layers" --out "results/update/angle_$tag" || echo "FAILED $tag"
}

run ~/LLM/models--state-spaces--mamba-130m-hf mamba130 "$(seq -s, 0 24)" results/ctrl/mamba130/a1i_txt
run ~/LLM/models--EleutherAI--pythia-410m pythia410 "$(seq -s, 0 23)" results/ctrl/pythia410/a1i_txt
run ~/LLM/models--randqwen-g1.0 randqwen "$(seq -s, 0 27)" results/ctrl/randqwen_g1.0/a1i_txt
echo "ANGLE CONTROLS DONE $(date)"
