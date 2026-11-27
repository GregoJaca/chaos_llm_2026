#!/bin/bash
# v3 campaign, x11 (T4 15 GB).  Goal: enough statistics for the closed-loop vs
# replay comparison (512 directions per model/T/prompt instead of 8), plus the
# attention, window and rotation probes the PI asked for.
#
# Every job is resume-safe (skips when its softloop.npz / a1.npz exists) and a
# failure prints FAILED and moves on.  Nothing is killed, nothing retries.
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY0=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311/bin/python

declare -A MODEL PYV DT RL LL
MODEL[ds15]=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B; PYV[ds15]=$PY0;   DT[ds15]=float16; RL[ds15]=0.03; LL[ds15]=7,14,21,27
MODEL[qwen25]=~/LLM/models--Qwen--Qwen2.5-3B-Instruct;               PYV[qwen25]=$PY;    DT[qwen25]=float16; RL[qwen25]=0.03; LL[qwen25]=9,18,27,35
MODEL[qwen34]=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507;            PYV[qwen34]=$PY;    DT[qwen34]=float16; RL[qwen34]=0.03; LL[qwen34]=9,18,27,35
MODEL[qwen35]=~/LLM/models--Qwen--Qwen3.5-2B;                        PYV[qwen35]=$PY311; DT[qwen35]=float16; RL[qwen35]=0.03; LL[qwen35]=6,12,18,23
MODEL[phi]=~/LLM/models--microsoft--Phi-4-mini-instruct;             PYV[phi]=$PY;       DT[phi]=float16; RL[phi]=0.1;  LL[phi]=8,16,24,31
MODEL[mamba]=~/LLM/models--state-spaces--mamba-130m-hf;              PYV[mamba]=$PY0;    DT[mamba]=float16; RL[mamba]=0.03; LL[mamba]=6,12,18,23
MODEL[pythia]=~/LLM/models--EleutherAI--pythia-410m;                 PYV[pythia]=$PY;    DT[pythia]=float16; RL[pythia]=0.1; LL[pythia]=6,12,18,23
MODEL[randqwen]=~/LLM/models--randqwen-g1.0;                         PYV[randqwen]=$PY0; DT[randqwen]=float16; RL[randqwen]=0.03; LL[randqwen]=7,14,21,27

ORDER="ds15 qwen25 qwen34 qwen35 phi mamba pythia randqwen"

# $1=out $2=tag $3=T $4=dirs $5=steps $6=seed $7=prompt $8=window $9=mode(loop|replay|null)
sl () {
  local out=$1 tag=$2 temp=$3 dirs=$4 steps=$5 seed=$6 prompt=$7 win=$8 mode=$9
  local loopdir=${10:-}
  [ -f "$out/softloop.npz" ] && { echo "skip $out"; return; }
  if [ "$mode" = "replay" ] && [ ! -f "$loopdir/softloop.npz" ]; then echo "no loop for $out"; return; fi
  echo "=== $out T$temp K$dirs n$steps $(date)"
  local args="--model ${MODEL[$tag]} --steps $steps --temp $temp --dirs $dirs --r ${RL[$tag]} \
    --save-layers ${LL[$tag]} --dtype ${DT[$tag]} --window $win --seed $seed --prompt $prompt --out $out"
  if [ "$mode" = "replay" ]; then
    ${PYV[$tag]} -m chaos_llm.run_softloop $args --mode replay --replay-file "$loopdir/softloop.npz" \
      || echo "FAILED $out"
  else
    ${PYV[$tag]} -m chaos_llm.run_softloop $args --feedback soft || echo "FAILED $out"
  fi
}

P1=@prompts.txt
PS=results/prompts_single

# ---------------------------------------------------------------- A. T=1, 512 dirs
for tag in $ORDER; do
  for c in 1 2; do
    L=results/longloop_v3/$tag/T1.0/k$c/loop
    R=results/longloop_v3/$tag/T1.0/k$c/replay
    sl "$L" "$tag" 1.0 256 1200 $((1000+c)) "$P1" 4096 loop
    sl "$R" "$tag" 1.0 256 1200 $((1000+c)) "$P1" 4096 replay "$L"
  done
done

# ---------------------------------------------------------------- B. T=2 loops + replays (256 dirs)
for tag in $ORDER; do
  L=results/longloop_v3/$tag/T2.0/loop
  R=results/longloop_v3/$tag/T2.0/replay
  sl "$L" "$tag" 2.0 256 1200 2001 "$P1" 4096 loop
  sl "$R" "$tag" 2.0 256 1200 2001 "$P1" 4096 replay "$L"
done

# ---------------------------------------------------------------- C. T=0.5 loops (256 dirs)
for tag in $ORDER; do
  sl results/longloop_v3/$tag/T0.5/loop "$tag" 0.5 256 1200 2002 "$P1" 4096 loop
done

# ---------------------------------------------------------------- D. long horizon, 3000 steps
for tag in ds15 qwen34 phi; do
  L=results/longloop_v3/$tag/long3000/loop
  R=results/longloop_v3/$tag/long3000/replay
  sl "$L" "$tag" 1.0 128 3000 3001 "$P1" 4096 loop
  sl "$R" "$tag" 1.0 128 3000 3001 "$P1" 4096 replay "$L"
done

# ---------------------------------------------------------------- E. window dependence of tau
for W in 64 128 256 512; do
  sl results/longloop_v3/ds15/W$W/loop ds15 1.0 64 3000 4001 "$P1" "$W" loop
done
for W in 256 512; do
  sl results/longloop_v3/ds15/W$W/replay ds15 1.0 64 3000 4001 "$P1" "$W" replay results/longloop_v3/ds15/W$W/loop
done

# ---------------------------------------------------------------- F. multi-prompt, 256 dirs
for tag in ds15 qwen34 phi qwen25; do
  for p in 2 3 4; do
    L=results/longloop_v3_prompt/$tag/p$p/T1.0/loop
    R=results/longloop_v3_prompt/$tag/p$p/T1.0/replay
    sl "$L" "$tag" 1.0 256 1200 5000 "$PS/p$p.txt" 4096 loop
    sl "$R" "$tag" 1.0 256 1200 5000 "$PS/p$p.txt" 4096 replay "$L"
  done
done

# ---------------------------------------------------------------- G. attention: all layers, several W
for tag in $ORDER; do
  case $tag in
    ds15)   AL="0,1,2,3,7,14,21,27";; qwen25) AL="0,1,2,9,18,27,35";;
    qwen34) AL="0,1,2,9,18,27,35";;  qwen35) AL="0,1,2,6,12,18,23";;
    phi)    AL="0,1,2,8,16,24,31";;  mamba)  AL="0,1,2,6,12,18,23";;
    pythia) AL="0,1,2,6,12,18,23";;  randqwen) AL="0,1,2,7,14,21,27";;
  esac
  for W in 50 100 200 4096; do
    D=results/update/attn_${tag}_w$W
    [ -f "$D/attn_frac.npz" ] && { echo "skip $D"; continue; }
    echo "=== attn $tag W=$W $(date)"
    ${PYV[$tag]} -m chaos_llm.run_attention_frac --model "${MODEL[$tag]}" --dtype "${DT[$tag]}" \
      --window "$W" --max-new-tokens 300 --layers "$AL" --out "$D" || echo "FAILED attn $tag $W"
  done
done

# ---------------------------------------------------------------- H. A1 window control (no mask cut)
for tag in ds15 phi qwen35 qwen34; do
  D=results/long_a1/$tag
  [ -f "$D/a1.npz" ] && { echo "skip $D"; continue; }
  echo "=== a1 $tag W=4096 $(date)"
  ${PYV[$tag]} -m chaos_llm.run_a1 --model "${MODEL[$tag]}" --dtype "${DT[$tag]}" --prompt "$P1" \
    --window 4096 --max-new-tokens 400 --num-directions 128 --radii "${RL[$tag]}" \
    --layers "${LL[$tag]}" --batch-dirs 2 --out "$D" || echo "FAILED a1 $tag"
done

# ---------------------------------------------------------------- I. top-k rotation probe
# (separate script: scripts/queue_topk_x11.sh, added after the probe is written)
echo "V3 X11 DONE $(date)"
