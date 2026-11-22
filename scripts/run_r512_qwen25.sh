#!/usr/bin/env bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=/home/grego/LLM/models--Qwen--Qwen2.5-3B-Instruct

mkdir -p results/rstar/qwen25_3b

for pi in 2 3 4 5; do
  idx=$((pi - 1))
  OUT=results/rstar/qwen25_3b/p${pi}
  if [ ! -f $OUT/a1.npz ]; then
    echo "=== Running Qwen2.5-3B r512 prompt $pi (line $idx of prompts_more.txt) $(date) ==="
    sed -n "${idx}p" prompts_more.txt > /tmp/p${pi}.txt
    ~/LLM/venv_tf/bin/python src/chaos_llm/run_a1.py \
      --model "$MODEL" \
      --prompt @/tmp/p${pi}.txt \
      --window 100 \
      --max-new-tokens 300 \
      --num-directions 512 \
      --batch-dirs 8 \
      --inbatch-ref \
      --radii 0.1 \
      --layers "35" \
      --out "$OUT" || echo "FAILED Qwen2.5-3B p$pi"
  fi
done
echo "=== Qwen2.5-3B r512 completed $(date) ==="
