#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
export CHAOS_DTYPE=float16
echo "QUEUE44 START $(date)"

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q4=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PH=~/LLM/models--microsoft--Phi-4-mini-instruct
Q35=~/LLM/models--Qwen--Qwen3.5-2B
Q25=~/LLM/models--Qwen--Qwen2.5-3B-Instruct

# ---------------- A. depth tangent spectrum (fixed QR, per-block singular values)
[ -f results/depth2/qwen3_4b/depth.npz ] || \
  $PY -m chaos_llm.run_depth2 --model $Q4 --run results/models/qwen3_4b/a1 \
    --tokens 0,8,16,24,32,50,80,100,143,200,250 --k 8 --dtype float16 \
    --out results/depth2/qwen3_4b || echo "FAILED A qwen3_4b"
[ -f results/depth2/phi4mini/depth.npz ] || \
  $PY -m chaos_llm.run_depth2 --model $PH --run results/models/phi4mini/a1 \
    --tokens 0,8,16,24,32,50,80,100,143,200,250 --k 8 --dtype float16 \
    --out results/depth2/phi4mini || echo "FAILED A phi"
[ -f results/depth2/qwen3_5/depth.npz ] || \
  $PY311 -m chaos_llm.run_depth2 --model $Q35 --run results/models/qwen3_5/a1 \
    --tokens 0,8,16,24,32,50,80,100,143 --k 8 --dtype float16 \
    --out results/depth2/qwen3_5 || echo "FAILED A qwen3_5"
[ -f results/depth2/qwen25_3b/depth.npz ] || \
  $PY -m chaos_llm.run_depth2 --model $Q25 --run results/models/qwen25_3b/a1 \
    --tokens 0,8,16,24,32,50,80,100,143,200,250 --k 8 --dtype float16 \
    --out results/depth2/qwen25_3b || echo "FAILED A qwen25"

# many tokens for DeepSeek: 128 independent starts
[ -f results/depth2/deepseek128/depth.npz ] || \
  $PY39 -m chaos_llm.run_depth2 --model $DS --run results/big4/a1_all_layers/prompts \
    --tokens $(seq -s, 0 2 254) --k 8 --dtype float32 \
    --out results/depth2/deepseek128 || echo "FAILED A deepseek128"
echo "A done $(date)"

# ---------------- B. r* universality: 5 prompts x 512 directions
for i in 1 2 3 4 5; do
  sed -n "${i}p" prompts_more2.txt > /tmp/R_$i.txt
  [ -s /tmp/R_$i.txt ] || continue
  for m in ds q4 phi q35; do
    case $m in
      ds)  M=$DS;  BIN=$PY39;  TAG=deepseek ;;
      q4)  M=$Q4;  BIN=$PY;    TAG=qwen3_4b ;;
      phi) M=$PH;  BIN=$PY;    TAG=phi4mini ;;
      q35) M=$Q35; BIN=$PY311; TAG=qwen3_5 ;;
    esac
    out=results/rstar/$TAG/p$i
    [ -f "$out/a1.npz" ] && continue
    $BIN -m chaos_llm.run_a1 --model "$M" --prompt @/tmp/R_$i.txt --window 100 \
      --max-new-tokens 300 --num-directions 512 --radii 0.1 \
      --layers 7,14,21 --batch-dirs 8 --out "$out" || echo "FAILED B $TAG $i"
  done
  echo "rstar prompt $i done $(date)"
done

# ---------------- C. flip hazard with 512 directions
for r in 0.1 0.3 0.6 1.0; do
  out=results/hazard/deepseek_r$r
  [ -f "$out/free_run.npz" ] && continue
  $PY39 -m chaos_llm.run_thresholds --run results/rstar/deepseek/p1 \
    --radii $r --chunk 16 --out "$out" || echo "FAILED C $r"
done
echo "C done $(date)"

# ---------------- D. long horizon: 8 more 4000-token trajectories
for i in 1 2 3 4 5 6 7 8; do
  sed -n "${i}p" prompts_more2.txt > /tmp/T_$i.txt
  out=results/long4k/p$i.npz
  [ -f "$out" ] && continue
  $PY39 -m chaos_llm.collect_trajectories --prompts /tmp/T_$i.txt --window 4096 \
    --max-new-tokens 4000 --layers 7,14,21,27 \
    --out results/long4k_tmp_$i || echo "FAILED D $i"
  mkdir -p results/long4k
  mv results/long4k_tmp_$i/*.npz "$out" 2>/dev/null
  echo "long4k $i done $(date)"
done

# ---------------- E. position scan, more positions
probe () {
  local M=$1 RUN=$2 TAG=$3 PYBIN=$4 DT=$5 R=$6
  for pos in 6 10 14 18 22 28 35 45 55 70; do
    out=results/probe2/$TAG/pos$pos
    [ -f "$out/layer_probe.npz" ] && continue
    $PYBIN -m chaos_llm.run_layer_probe --model "$M" --run "$RUN" --out "$out" \
      --layers 0,5,10 --dirs 8 --r "$R" --pos $pos --dtype "$DT" > /dev/null 2>&1 \
      || echo "FAILED E $TAG $pos"
  done
}
probe $DS results/big4/a1_all_layers/prompts deepseek $PY39 float32 0.1
probe $Q4 results/models/qwen3_4b/a1 qwen3_4b $PY float16 0.1
probe $PH results/models/phi4mini/a1 phi4mini $PY float16 0.3
probe $Q35 results/models/qwen3_5/a1 qwen3_5 $PY311 float16 0.1
probe $Q25 results/models/qwen25_3b/a1 qwen25_3b $PY float16 0.1
echo "E done $(date)"

echo "QUEUE44 DONE $(date)"
touch results/QUEUE44_DONE
