#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
echo "QUEUE33 START $(date)"

# Qwen3-4B window sweep with explicit 4D sliding mask
for w in 60 100 200 4096; do
  out=results/models/qwen3_4b/window4d/w$w
  [ -f "$out/a1.npz" ] && continue
  $PY -m chaos_llm.run_a1 --model ~/LLM/models--Qwen--Qwen3-4B-Instruct-2507 \
    --prompt @prompts.txt --window $w --window4d --max-new-tokens 300 \
    --num-directions 6 --radii 0.1 --layers 1,2,3,4,5,6,7,14,20,27,35 \
    --batch-dirs 1 --out "$out" || echo "FAILED q4 w$w"
done

# Phi-4-mini window sweep
for w in 60 100 200 4096; do
  out=results/models/phi4mini/window4d/w$w
  [ -f "$out/a1.npz" ] && continue
  $PY -m chaos_llm.run_a1 --model ~/LLM/models--microsoft--Phi-4-mini-instruct \
    --prompt @prompts.txt --window $w --window4d --max-new-tokens 300 \
    --num-directions 6 --radii 0.1 --layers 1,2,3,4,5,6,7,14,20,27,31 \
    --batch-dirs 1 --out "$out" || echo "FAILED phi w$w"
done

# Qwen3.5 window sweep
export CHAOS_DTYPE=float16
for w in 60 100 200 4096; do
  out=results/models/qwen3_5/window4d/w$w
  [ -f "$out/a1.npz" ] && continue
  $PY311 -m chaos_llm.run_a1 --model ~/LLM/models--Qwen--Qwen3.5-2B \
    --prompt @prompts.txt --window $w --window4d --max-new-tokens 200 \
    --num-directions 6 --radii 0.1 --layers 1,2,3,4,8,12,16,20,23 \
    --batch-dirs 1 --out "$out" || echo "FAILED q35 w$w"
done

echo "QUEUE33 DONE $(date)"
