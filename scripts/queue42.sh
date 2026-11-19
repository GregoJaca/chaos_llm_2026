#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
export CHAOS_DTYPE=float16

while pgrep -f "queue41.sh" > /dev/null; do sleep 30; done
echo "QUEUE42 START $(date)"

Q4=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PH=~/LLM/models--microsoft--Phi-4-mini-instruct
Q35=~/LLM/models--Qwen--Qwen3.5-2B
Q25=~/LLM/models--Qwen--Qwen2.5-3B-Instruct

$PY -m chaos_llm.run_depth2 --model $Q4 --run results/models/qwen3_4b/a1 \
  --tokens 0,8,16,32,50,80,100,143,200,250 --k 6 --dtype float16 \
  --out results/depth2/qwen3_4b || echo "FAILED depth qwen3_4b"
$PY -m chaos_llm.run_depth2 --model $PH --run results/models/phi4mini/a1 \
  --tokens 0,8,16,32,50,80,100,143,200,250 --k 6 --dtype float16 \
  --out results/depth2/phi4mini || echo "FAILED depth phi"
$PY311 -m chaos_llm.run_depth2 --model $Q35 --run results/models/qwen3_5/a1 \
  --tokens 0,8,16,32,50,80,100,143 --k 6 --dtype float16 \
  --out results/depth2/qwen3_5 || echo "FAILED depth qwen3_5"
$PY -m chaos_llm.run_depth2 --model $Q25 --run results/models/qwen25_3b/a1 \
  --tokens 0,8,16,32,50,80,100,143,200,250 --k 6 --dtype float16 \
  --out results/depth2/qwen25_3b || echo "FAILED depth qwen25"
echo "DEPTH2 ALL DONE $(date)"

# long trajectories: does the regime persist beyond 300 tokens?
for m in q4 phi; do
  case $m in
    q4)  M=$Q4;  TAG=qwen3_4b ;;
    phi) M=$PH;  TAG=phi4mini ;;
  esac
  for i in 1 2 3; do
    sed -n "${i}p" prompts_long2.txt > /tmp/LT_$i.txt
    out=results/models/$TAG/traj2k/p$i.npz
    [ -f "$out" ] && continue
    $PY -m chaos_llm.collect_trajectories --model "$M" --prompts /tmp/LT_$i.txt \
      --window 4096 --max-new-tokens 2000 --layers 7,14,21,27,31 \
      --out results/models/$TAG/traj2k_tmp_$i || echo "FAILED traj2k $TAG $i"
    mkdir -p results/models/$TAG/traj2k
    mv results/models/$TAG/traj2k_tmp_$i/*.npz "$out" 2>/dev/null
  done
  echo "traj2k $TAG done $(date)"
done

echo "QUEUE42 DONE $(date)"
