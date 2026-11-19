#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
ALL=$(seq -s, 0 27)
echo "QUEUE31 START $(date)"

# DeepSeek: six prompts, all 28 layers
for i in 1 2 3 4 5 6; do
  sed -n "${i}p" prompts_more2.txt > /tmp/S_$i.txt
  out=results/big4/a1_all_layers/p$i
  [ -f "$out/a1.npz" ] && continue
  $PY39 -m chaos_llm.run_a1 --prompt @/tmp/S_$i.txt --window 100 --max-new-tokens 300 \
    --num-directions 10 --radii 0.1 --layers $ALL --batch-dirs 1 --out "$out" || echo "FAILED ds p$i"
done

# Qwen3-4B: p4-p6, all layers
ALLQ=$(seq -s, 0 35)
for i in 4 5 6; do
  sed -n "${i}p" prompts_more2.txt > /tmp/S_$i.txt
  out=results/models/qwen3_4b/a1_p$i
  [ -f "$out/a1.npz" ] && continue
  $PY -m chaos_llm.run_a1 --model ~/LLM/models--Qwen--Qwen3-4B-Instruct-2507 \
    --prompt @/tmp/S_$i.txt --window 100 --max-new-tokens 300 --num-directions 10 \
    --radii 0.1 --layers $ALLQ --batch-dirs 1 --out "$out" || echo "FAILED q4 p$i"
done

# Phi: p4-p6, all layers
ALLP=$(seq -s, 0 31)
for i in 4 5 6; do
  sed -n "${i}p" prompts_more2.txt > /tmp/S_$i.txt
  out=results/models/phi4mini/a1_p$i
  [ -f "$out/a1.npz" ] && continue
  $PY -m chaos_llm.run_a1 --model ~/LLM/models--microsoft--Phi-4-mini-instruct \
    --prompt @/tmp/S_$i.txt --window 100 --max-new-tokens 300 --num-directions 10 \
    --radii 0.3 --layers $ALLP --batch-dirs 1 --out "$out" || echo "FAILED phi p$i"
done

# Qwen3.5: p4-p6
export CHAOS_DTYPE=float16
for i in 4 5 6; do
  sed -n "${i}p" prompts_more2.txt > /tmp/S_$i.txt
  out=results/models/qwen3_5/a1_p$i
  [ -f "$out/a1.npz" ] && continue
  $PY311 -m chaos_llm.run_a1 --model ~/LLM/models--Qwen--Qwen3.5-2B \
    --prompt @/tmp/S_$i.txt --window 100 --max-new-tokens 200 --num-directions 8 \
    --radii 0.1 --layers 0,4,8,12,16,20,23 --batch-dirs 1 --out "$out" || echo "FAILED q35 p$i"
done

echo "QUEUE31 DONE $(date)"
