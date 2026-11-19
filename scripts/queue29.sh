#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv_tf/bin/python
echo "QUEUE29 START $(date)"

for i in 4 5 6; do
  sed -n "${i}p" prompts_more2.txt > /tmp/C_$i.txt
  [ -s /tmp/C_$i.txt ] || continue

  # Qwen3-4B
  g=results/models/qwen3_4b/chaos_geom/p$i
  if [ ! -f "$g/a1.npz" ]; then
    $PY -m chaos_llm.run_a1 --model ~/LLM/models--Qwen--Qwen3-4B-Instruct-2507 \
      --prompt @/tmp/C_$i.txt --window 100 --max-new-tokens 300 --num-directions 20 \
      --radii 0.01,0.1 --layers 7,14,35 --batch-dirs 1 --out "$g" || echo "FAILED q4 geom $i"
  fi
  for r in 0.05 0.1 0.2 0.4 0.8 1.5; do
    out=results/models/qwen3_4b/chaos/p${i}_r${r}
    [ -f "$out/free_run.npz" ] && continue
    $PY -m chaos_llm.run_thresholds --model ~/LLM/models--Qwen--Qwen3-4B-Instruct-2507 \
      --run "$g" --radii $r --chunk 6 --out "$out" || echo "FAILED q4 chaos $i $r"
  done

  # Phi-4-mini
  g=results/models/phi4mini/chaos_geom/p$i
  if [ ! -f "$g/a1.npz" ]; then
    $PY -m chaos_llm.run_a1 --model ~/LLM/models--microsoft--Phi-4-mini-instruct \
      --prompt @/tmp/C_$i.txt --window 100 --max-new-tokens 300 --num-directions 20 \
      --radii 0.01,0.3 --layers 7,14,31 --batch-dirs 1 --out "$g" || echo "FAILED phi geom $i"
  fi
  for r in 0.1 0.3 0.6 1.0 1.5; do
    out=results/models/phi4mini/chaos/p${i}_r${r}
    [ -f "$out/free_run.npz" ] && continue
    $PY -m chaos_llm.run_thresholds --model ~/LLM/models--microsoft--Phi-4-mini-instruct \
      --run "$g" --radii $r --chunk 6 --out "$out" || echo "FAILED phi chaos $i $r"
  done
done

echo "QUEUE29 DONE $(date)"
