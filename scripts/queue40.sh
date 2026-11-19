#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
export CHAOS_DTYPE=float16

while pgrep -f "queue39.sh" > /dev/null; do sleep 30; done
echo "QUEUE40 START $(date)"

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q4=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PH=~/LLM/models--microsoft--Phi-4-mini-instruct
Q35=~/LLM/models--Qwen--Qwen3.5-2B

# --- depth tangent spectrum (lambda_1 vs layer), all models
[ -f results/depth2/deepseek/depth.npz ] || \
  $PY39 -m chaos_llm.run_depth2 --model $DS --run results/big4/a1_all_layers/prompts \
    --tokens 0,4,8,12,16,20,30,40,50,70,100,123,143,180,200,250 --k 8 \
    --dtype float32 --out results/depth2/deepseek || echo "FAILED depth deepseek"

[ -f results/depth2/qwen3_4b/depth.npz ] || \
  $PY -m chaos_llm.run_depth2 --model $Q4 --run results/models/qwen3_4b/a1 \
    --tokens 0,8,16,32,50,80,100,143,200,250 --k 6 \
    --dtype float16 --out results/depth2/qwen3_4b || echo "FAILED depth qwen3_4b"

[ -f results/depth2/phi4mini/depth.npz ] || \
  $PY -m chaos_llm.run_depth2 --model $PH --run results/models/phi4mini/a1 \
    --tokens 0,8,16,32,50,80,100,143,200,250 --k 6 \
    --dtype float16 --out results/depth2/phi4mini || echo "FAILED depth phi"

[ -f results/depth2/qwen3_5/depth.npz ] || \
  $PY311 -m chaos_llm.run_depth2 --model $Q35 --run results/models/qwen3_5/a1 \
    --tokens 0,8,16,32,50,80,100,143 --k 6 \
    --dtype float16 --out results/depth2/qwen3_5 || echo "FAILED depth qwen3_5"

[ -f results/depth2/qwen25_3b/depth.npz ] && echo "qwen25 depth already done" || \
  $PY -m chaos_llm.run_depth2 --model ~/LLM/models--Qwen--Qwen2.5-3B-Instruct \
    --run results/models/qwen25_3b/a1 \
    --tokens 0,8,16,32,50,80,100,143,200,250 --k 6 \
    --dtype float16 --out results/depth2/qwen25_3b || echo "FAILED depth qwen25"

# --- Qwen3.5 with all 24 layers saved
ALL35=$(seq -s, 0 23)
for i in 1 2 3; do
  sed -n "${i}p" prompts_more2.txt > /tmp/Q35A_$i.txt
  out=results/models/qwen3_5/all_layers/p$i
  [ -f "$out/a1.npz" ] && continue
  $PY311 -m chaos_llm.run_a1 --model $Q35 --prompt @/tmp/Q35A_$i.txt --window 100 \
    --max-new-tokens 200 --num-directions 10 --radii 0.1 --layers $ALL35 \
    --batch-dirs 1 --out "$out" || echo "FAILED q35 all layers $i"
done

echo "QUEUE40 DONE $(date)"
