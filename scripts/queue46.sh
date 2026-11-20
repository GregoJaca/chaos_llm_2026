#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
export CHAOS_DTYPE=float16

while pgrep -f "queue44.sh" > /dev/null; do sleep 60; done
echo "QUEUE46 START $(date)"

# A. 64 tangent directions (instead of 8), 20 tokens
[ -f results/depth2/deepseek_k64/depth.npz ] || \
  $PY39 -m chaos_llm.run_depth2 \
    --model ~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B \
    --run results/big4/a1_all_layers/prompts \
    --tokens 0,4,8,12,16,24,32,40,50,64,80,100,123,143,180,200,230,250 --k 64 \
    --dtype float32 --out results/depth2/deepseek_k64 || echo "FAILED k64 ds"

[ -f results/depth2/qwen3_4b_k64/depth.npz ] || \
  $PY -m chaos_llm.run_depth2 --model ~/LLM/models--Qwen--Qwen3-4B-Instruct-2507 \
    --run results/models/qwen3_4b/a1 \
    --tokens 0,8,16,32,50,80,100,143,200,250 --k 64 --dtype float16 \
    --out results/depth2/qwen3_4b_k64 || echo "FAILED k64 q4"
echo "A done $(date)"

# B. full-dimension top singular value of every block, 3 positions
[ -f results/block_sigma/deepseek/block_sigma.npz ] || \
  $PY39 -m chaos_llm.run_block_sigma \
    --model ~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B \
    --run results/big4/a1_all_layers/prompts --tokens 16,50,100 --iters 25 \
    --dtype float32 --out results/block_sigma/deepseek || echo "FAILED blocksigma ds"

[ -f results/block_sigma/qwen3_4b/block_sigma.npz ] || \
  $PY -m chaos_llm.run_block_sigma --model ~/LLM/models--Qwen--Qwen3-4B-Instruct-2507 \
    --run results/models/qwen3_4b/a1 --tokens 16,50,100 --iters 25 \
    --dtype float16 --out results/block_sigma/qwen3_4b || echo "FAILED blocksigma q4"

echo "QUEUE46 DONE $(date)"
touch results/QUEUE46_DONE
