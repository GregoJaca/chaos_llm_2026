#!/bin/bash
# v3 campaign, Spark (GB10).  512 directions per (model, prompt) at T=1,
# plus T=2, multi-prompt, long horizon, attention and rotation probes.
# Measured: 64 dirs, 1200 steps ≈ 12 min for a 14B fp16 model.
set -u
cd /home/grego/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/grego/LLM/venv5/bin/python

declare -A MODEL RL LL
MODEL[qwen314]=/home/grego/LLM/models/Qwen3-14B;                 RL[qwen314]=0.1; LL[qwen314]=10,20,30,39
MODEL[ds14]=/home/grego/LLM/models/DeepSeek-R1-Distill-Qwen-14B; RL[ds14]=0.1;    LL[ds14]=12,24,36,47
MODEL[phi4]=/home/grego/LLM/models/phi-4;                        RL[phi4]=0.1;    LL[phi4]=10,20,30,39
MODEL[qwen35_9b]=/home/grego/LLM/models/Qwen3.5-9B;              RL[qwen35_9b]=0.1; LL[qwen35_9b]=8,16,24,31

ORDER="qwen314 ds14 qwen35_9b phi4"
P1=@prompts.txt
PS=results/prompts_single

sl () { # out tag temp dirs steps seed prompt win mode loopdir
  local out=$1 tag=$2 temp=$3 dirs=$4 steps=$5 seed=$6 prompt=$7 win=$8 mode=$9
  local loopdir=${10:-}
  [ -f "$out/softloop.npz" ] && { echo "skip $out"; return; }
  if [ "$mode" = "replay" ] && [ ! -f "$loopdir/softloop.npz" ]; then echo "no loop for $out"; return; fi
  echo "=== $out T$temp K$dirs n$steps $(date)"
  local args="--model ${MODEL[$tag]} --steps $steps --temp $temp --dirs $dirs --r ${RL[$tag]} \
    --save-layers ${LL[$tag]} --dtype float16 --window $win --seed $seed --prompt $prompt --out $out"
  if [ "$mode" = "replay" ]; then
    $PY -m chaos_llm.run_softloop $args --mode replay --replay-file "$loopdir/softloop.npz" || echo "FAILED $out"
  else
    $PY -m chaos_llm.run_softloop $args --feedback soft || echo "FAILED $out"
  fi
}

# ---------------------------------------------------------------- A. T=1, 8 x 64 = 512 dirs
for tag in $ORDER; do
  for c in 1 2 3 4 5 6 7 8; do
    L=results/longloop_v3/$tag/T1.0/k$c/loop
    R=results/longloop_v3/$tag/T1.0/k$c/replay
    sl "$L" "$tag" 1.0 64 1200 $((7000+c)) "$P1" 4096 loop
    sl "$R" "$tag" 1.0 64 1200 $((7000+c)) "$P1" 4096 replay "$L"
  done
done

# ---------------------------------------------------------------- B. multi-prompt, 256 dirs
for tag in qwen314 ds14; do
  for p in 2 3; do
    for c in 1 2 3 4; do
      L=results/longloop_v3_prompt/$tag/p$p/k$c/loop
      R=results/longloop_v3_prompt/$tag/p$p/k$c/replay
      sl "$L" "$tag" 1.0 64 1200 $((8000+c)) "$PS/p$p.txt" 4096 loop
      sl "$R" "$tag" 1.0 64 1200 $((8000+c)) "$PS/p$p.txt" 4096 replay "$L"
    done
  done
done

# ---------------------------------------------------------------- C. T=2 and T=0.5, 256 dirs
for tag in $ORDER; do
  for c in 1 2 3 4; do
    L=results/longloop_v3/$tag/T2.0/k$c/loop
    R=results/longloop_v3/$tag/T2.0/k$c/replay
    sl "$L" "$tag" 2.0 64 1200 $((9000+c)) "$P1" 4096 loop
    sl "$R" "$tag" 2.0 64 1200 $((9000+c)) "$P1" 4096 replay "$L"
  done
  sl results/longloop_v3/$tag/T0.5/loop "$tag" 0.5 128 1200 9500 "$P1" 4096 loop
done

# ---------------------------------------------------------------- D. long horizon
for tag in qwen314 ds14 phi4; do
  L=results/longloop_v3/$tag/long3000/loop
  R=results/longloop_v3/$tag/long3000/replay
  sl "$L" "$tag" 1.0 64 3000 9600 "$P1" 4096 loop
  sl "$R" "$tag" 1.0 64 3000 9600 "$P1" 4096 replay "$L"
done

# ---------------------------------------------------------------- E. attention, all layers
for tag in $ORDER; do
  case $tag in
    qwen314) AL="0,1,2,10,20,30,39";; ds14) AL="0,1,2,12,24,36,47";;
    phi4)    AL="0,1,2,10,20,30,39";; qwen35_9b) AL="0,1,2,8,16,24,31";;
  esac
  for W in 100 4096; do
    D=results/update/attn_${tag}_w$W
    [ -f "$D/attn_frac.npz" ] && { echo "skip $D"; continue; }
    echo "=== attn $tag W=$W $(date)"
    $PY -m chaos_llm.run_attention_frac --model "${MODEL[$tag]}" --dtype float16 \
      --window "$W" --max-new-tokens 300 --layers "$AL" --out "$D" || echo "FAILED attn $tag $W"
  done
done
echo "V3 SPARK DONE $(date)"
