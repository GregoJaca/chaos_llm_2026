#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY311=~/LLM/venv311b/bin/python
export CHAOS_DTYPE=float32
echo "QUEUE58 START $(date)"

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q35=~/LLM/models--Qwen--Qwen3.5-2B

# superposition at r=0.03 on more DeepSeek prompts (fp32)
for p in 2 3; do
  out=results/superposition/deepseek_p$p
  [ -f "$out/superposition.npz" ] && continue
  $PY39 -m chaos_llm.run_superposition --model $DS --run results/sens/deepseek_p$p/geom \
    --trials 3 --r 0.03 --layers 1,7,14,last --dtype float32 --out "$out" \
    || echo "FAILED sup ds p$p"
  sleep 5
done

# Qwen3.5-2B in fp32 (fits the T4) so the superposition test is measurable there
out=results/superposition/qwen3_5_fp32
[ -f "$out/superposition.npz" ] || \
  $PY311 -m chaos_llm.run_superposition --model $Q35 --run results/sens/qwen3_5_p1/geom \
    --trials 3 --r 0.03 --layers 1,7,14,last --dtype float32 --out "$out" \
    || echo "FAILED sup q35 fp32"

echo "QUEUE58 DONE $(date)"
touch results/QUEUE58_DONE
