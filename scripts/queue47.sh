#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
export CHAOS_DTYPE=float16
echo "QUEUE47 START $(date)"

Q4=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PH=~/LLM/models--microsoft--Phi-4-mini-instruct
Q35=~/LLM/models--Qwen--Qwen3.5-2B
Q25=~/LLM/models--Qwen--Qwen2.5-3B-Instruct

sleep 10

# retry the OOM'd run, then the other models
[ -f results/block_sigma/qwen3_4b/block_sigma.npz ] || \
  $PY -m chaos_llm.run_block_sigma --model $Q4 --run results/models/qwen3_4b/a1 \
    --tokens 16,50,100 --iters 25 --dtype float16 --out results/block_sigma/qwen3_4b \
    || echo "FAILED bs q4"
[ -f results/block_sigma/phi4mini/block_sigma.npz ] || \
  $PY -m chaos_llm.run_block_sigma --model $PH --run results/models/phi4mini/a1 \
    --tokens 16,50,100 --iters 25 --dtype float16 --out results/block_sigma/phi4mini \
    || echo "FAILED bs phi"
[ -f results/block_sigma/qwen3_5/block_sigma.npz ] || \
  $PY311 -m chaos_llm.run_block_sigma --model $Q35 --run results/models/qwen3_5/a1 \
    --tokens 16,50,100 --iters 25 --dtype float16 --out results/block_sigma/qwen3_5 \
    || echo "FAILED bs q35"
echo "block sigma done $(date)"

# k=64 for the remaining models
[ -f results/depth2/phi4mini_k64/depth.npz ] || \
  $PY -m chaos_llm.run_depth2 --model $PH --run results/models/phi4mini/a1 \
    --tokens 0,8,16,32,50,80,100,143,200,250 --k 64 --dtype float16 \
    --out results/depth2/phi4mini_k64 || echo "FAILED k64 phi"
[ -f results/depth2/qwen3_5_k64/depth.npz ] || \
  $PY311 -m chaos_llm.run_depth2 --model $Q35 --run results/models/qwen3_5/a1 \
    --tokens 0,8,16,32,50,80,100,143 --k 64 --dtype float16 \
    --out results/depth2/qwen3_5_k64 || echo "FAILED k64 q35"
echo "k64 done $(date)"

# more flips for the hazard at other models (512 dirs)
for m in q4 phi; do
  case $m in
    q4) M=$Q4; TAG=qwen3_4b ;;
    phi) M=$PH; TAG=phi4mini ;;
  esac
  for r in 0.1 0.3 0.6 1.0; do
    out=results/hazard/${TAG}_r$r
    [ -f "$out/free_run.npz" ] && continue
    $PY -m chaos_llm.run_thresholds --model $M --run results/rstar/$TAG/p1 \
      --radii $r --chunk 16 --out "$out" || echo "FAILED hazard $TAG $r"
  done
done

echo "QUEUE47 DONE $(date)"
touch results/QUEUE47_DONE
