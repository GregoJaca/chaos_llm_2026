#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
export CHAOS_DTYPE=float16

while pgrep -f "queue42.sh" > /dev/null; do sleep 30; done
echo "QUEUE43 START $(date)"

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q4=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507

# 1. finer depth map: 32 tokens
[ -f results/depth2/deepseek32/depth.npz ] || \
  $PY39 -m chaos_llm.run_depth2 --model $DS --run results/big4/a1_all_layers/prompts \
    --tokens $(seq -s, 0 8 250) --k 8 --dtype float32 \
    --out results/depth2/deepseek32 || echo "FAILED depth32"

# 2. window x radius grid on DeepSeek (teacher-forced, all relevant layers)
for w in 80 100 120 150 200 300; do
  for r in 0.03 0.1 0.3; do
    out=results/big4/wgrid/w${w}_r${r}
    [ -f "$out/a1.npz" ] && continue
    $PY39 -m chaos_llm.run_a1 --prompt @prompts.txt --window $w --window4d \
      --max-new-tokens 300 --num-directions 8 --radii $r \
      --layers 1,3,7,14,21,27 --batch-dirs 1 --out "$out" \
      || echo "FAILED wgrid $w $r"
  done
done
echo "wgrid done $(date)"

# 3. propagation map with more directions at a few positions
probe () {
  local M=$1 RUN=$2 TAG=$3 PYBIN=$4 DT=$5 R=$6
  for pos in 8 12 16 24 32 48; do
    out=results/probe3/$TAG/pos$pos
    [ -f "$out/layer_probe.npz" ] && continue
    $PYBIN -m chaos_llm.run_layer_probe --model "$M" --run "$RUN" --out "$out" \
      --layers 0,4,8,12 --dirs 10 --r "$R" --pos $pos --dtype "$DT" > /dev/null 2>&1 \
      || echo "FAILED probe3 $TAG $pos"
  done
}
probe $DS results/big4/a1_all_layers/prompts deepseek $PY39 float32 0.1
probe $Q4 results/models/qwen3_4b/a1 qwen3_4b $PY float16 0.1
echo "probe3 done $(date)"

# 4. longer free runs for Qwen3-4B
for i in 1 2; do
  sed -n "${i}p" prompts_long2.txt > /tmp/LQ_$i.txt
  out=results/models/qwen3_4b/traj2k/p$i.npz
  [ -f "$out" ] && continue
  $PY -m chaos_llm.collect_trajectories --model $Q4 --prompts /tmp/LQ_$i.txt \
    --window 4096 --max-new-tokens 2000 --layers 7,14,21,27,35 \
    --out results/models/qwen3_4b/traj2k_tmp_$i || echo "FAILED traj2k q4 $i"
  mkdir -p results/models/qwen3_4b/traj2k
  mv results/models/qwen3_4b/traj2k_tmp_$i/*.npz "$out" 2>/dev/null
done

echo "QUEUE43 DONE $(date)"
