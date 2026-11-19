#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
export CHAOS_DTYPE=float16

while pgrep -f "queue34.sh" > /dev/null; do sleep 20; done
echo "QUEUE35 START $(date)"

# Native sliding window vs the explicit 4D mask: identical?
for mode in native mask4d; do
  out=results/big4/wcheck/$mode
  [ -f "$out/a1.npz" ] && continue
  extra=""
  [ "$mode" = "mask4d" ] && extra="--window4d"
  $PY39 -m chaos_llm.run_a1 --prompt @prompts_long2.txt --window 100 $extra \
    --max-new-tokens 300 --num-directions 4 --radii 0.1 --layers 1,2,3,7,14,21,27 \
    --batch-dirs 1 --out "$out" || echo "FAILED wcheck $mode"
done

# Large-radius teacher-forced runs: margin coordinate u = dm/m, flips at u = -1
for i in 1 2; do
  sed -n "${i}p" prompts_more2.txt > /tmp/BR_$i.txt
  out=results/big4/bigr/p$i
  if [ ! -f "$out/a1.npz" ]; then
    $PY39 -m chaos_llm.run_a1 --prompt @/tmp/BR_$i.txt --window 100 \
      --max-new-tokens 300 --num-directions 20 --radii 0.3,0.5,0.8,1.2,2.0 \
      --layers 7,14,21,27 --batch-dirs 1 --out "$out" || echo "FAILED bigr $i"
  fi
done

for i in 1 2; do
  sed -n "${i}p" prompts_more2.txt > /tmp/BR_$i.txt
  out=results/models/qwen3_4b/bigr/p$i
  if [ ! -f "$out/a1.npz" ]; then
    $PY -m chaos_llm.run_a1 --model ~/LLM/models--Qwen--Qwen3-4B-Instruct-2507 \
      --prompt @/tmp/BR_$i.txt --window 100 --max-new-tokens 300 --num-directions 20 \
      --radii 0.3,0.5,0.8,1.2,2.0 --layers 7,14,21,32,35 --batch-dirs 1 \
      --out "$out" || echo "FAILED q4 bigr $i"
  fi
  out=results/models/phi4mini/bigr/p$i
  if [ ! -f "$out/a1.npz" ]; then
    $PY -m chaos_llm.run_a1 --model ~/LLM/models--microsoft--Phi-4-mini-instruct \
      --prompt @/tmp/BR_$i.txt --window 100 --max-new-tokens 300 --num-directions 20 \
      --radii 0.3,0.5,0.8,1.2,2.0 --layers 7,14,21,28,31 --batch-dirs 1 \
      --out "$out" || echo "FAILED phi bigr $i"
  fi
done

echo "QUEUE35 DONE $(date)"
