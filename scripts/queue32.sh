#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export CHAOS_DTYPE=float16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv311b/bin/python
M=~/LLM/models--Qwen--Qwen3.5-2B
echo "QUEUE32 START $(date)"
for p in 4 5 6; do
  for r in 0.1 0.3 0.6 1.0 1.5; do
    out=results/models/qwen3_5/chaos/p${p}_r${r}
    [ -f "$out/free_run.npz" ] && continue
    $PY -m chaos_llm.run_thresholds --model $M --run results/models/qwen3_5/a1_p$p \
      --radii $r --chunk 4 --out "$out" || echo "FAILED q35 p$p r$r"
  done
done
echo "QUEUE32 DONE $(date)"
