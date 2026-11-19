#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export CHAOS_DTYPE=float16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv311b/bin/python
M=~/LLM/models--Qwen--Qwen3.5-2B
echo "QUEUE28 START $(date)"

# Qwen3.5: three more prompts, A1 all saved layers
for i in 1 2 3; do
  sed -n "${i}p" prompts_more2.txt > /tmp/Q35_$i.txt
  out=results/models/qwen3_5/a1_p$i
  [ -f "$out/a1.npz" ] && continue
  $PY -m chaos_llm.run_a1 --model $M --prompt @/tmp/Q35_$i.txt --window 100 \
    --max-new-tokens 200 --num-directions 8 --radii 0.1 \
    --layers 0,4,8,12,16,20,23 --batch-dirs 1 --out "$out" || echo "FAILED q35 p$i"
done

# Qwen3.5: stable-chaos free runs for the first prompt (5 radii)
for r in 0.1 0.3 0.6 1.0 1.5; do
  out=results/models/qwen3_5/chaos/p1_r${r}
  [ -f "$out/free_run.npz" ] && continue
  $PY -m chaos_llm.run_thresholds --model $M --run results/models/qwen3_5/a1 \
    --radii $r --chunk 4 --out "$out" || echo "FAILED q35 chaos $r"
done

echo "QUEUE28 DONE $(date)"
