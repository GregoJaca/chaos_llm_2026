#!/bin/bash
# v3b campaign, x11 (T4 15 GB).  FIXED direction counts: the KV cache grows with
# rows x steps, so 256 directions x 1200 steps does not fit (it OOMs around t=1000).
# Measured safe: 32 directions x 800 steps for every model here (KV 0.8-4 GB).
# Many directions are obtained by running several chunks with different seeds; all
# chunks of one (model, T) share the same unperturbed reference orbit, so their
# directions pool.
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
MODEL[qwen35]=~/LLM/models--Qwen--Qwen3.5-2B;                        PYV[qwen35]=$PY311; DT[qwen35]=float32; RL[qwen35]=0.03; LL[qwen35]=6,12,18,23
MODEL[phi]=~/LLM/models--microsoft--Phi-4-mini-instruct;             PYV[phi]=$PY;       DT[phi]=float16; RL[phi]=0.1;  LL[phi]=8,16,24,31
MODEL[mamba]=~/LLM/models--state-spaces--mamba-130m-hf;              PYV[mamba]=$PY0;    DT[mamba]=float16; RL[mamba]=0.03; LL[mamba]=6,12,18,23
MODEL[pythia]=~/LLM/models--EleutherAI--pythia-410m;                 PYV[pythia]=$PY;    DT[pythia]=float16; RL[pythia]=0.1; LL[pythia]=6,12,18,23
MODEL[randqwen]=~/LLM/models--randqwen-g1.0;                         PYV[randqwen]=$PY0; DT[randqwen]=float16; RL[randqwen]=0.03; LL[randqwen]=7,14,21,27
ORDER="ds15 qwen25 qwen34 qwen35 phi mamba pythia randqwen"

sl () { # out tag temp dirs steps seed prompt mode loopdir
  local out=$1 tag=$2 temp=$3 dirs=$4 steps=$5 seed=$6 prompt=$7 mode=$8
  local loopdir=${9:-}
  [ -f "$out/softloop.npz" ] && { echo "skip $out"; return; }
  if [ "$mode" = "replay" ] && [ ! -f "$loopdir/softloop.npz" ]; then echo "no loop $out"; return; fi
  echo "=== $out T$temp K$dirs n$steps $(date)"
  local args="--model ${MODEL[$tag]} --steps $steps --temp $temp --dirs $dirs --r ${RL[$tag]} \
    --save-layers ${LL[$tag]} --dtype ${DT[$tag]} --window 4096 --seed $seed --prompt $prompt --out $out"
  if [ "$mode" = "replay" ]; then
    ${PYV[$tag]} -m chaos_llm.run_softloop $args --mode replay --replay-file "$loopdir/softloop.npz" || echo "FAILED $out"
  else
    ${PYV[$tag]} -m chaos_llm.run_softloop $args --feedback soft || echo "FAILED $out"
  fi
}

# ---------------- A. T=1, 4 chunks x 32 = 128 directions (the headline)
for tag in $ORDER; do
  for c in 1 2 3 4; do
    L=results/longloop_v3b/$tag/T1.0/k$c/loop
    R=results/longloop_v3b/$tag/T1.0/k$c/replay
    sl "$L" "$tag" 1.0 32 800 $((1100+c)) @prompts.txt loop
    sl "$R" "$tag" 1.0 32 800 $((1100+c)) @prompts.txt replay "$L"
  done
done

# ---------------- B. T=2, 2 chunks = 64 directions
for tag in $ORDER; do
  for c in 1 2; do
    L=results/longloop_v3b/$tag/T2.0/k$c/loop
    R=results/longloop_v3b/$tag/T2.0/k$c/replay
    sl "$L" "$tag" 2.0 32 800 $((2100+c)) @prompts.txt loop
    sl "$R" "$tag" 2.0 32 800 $((2100+c)) @prompts.txt replay "$L"
  done
done

# ---------------- C. T=0.5, 2 chunks, loop only
for tag in $ORDER; do
  for c in 1 2; do
    sl results/longloop_v3b/$tag/T0.5/k$c/loop "$tag" 0.5 32 800 $((3100+c)) @prompts.txt loop
  done
done

# ---------------- D. multi-prompt (2 prompts x 2 chunks) for 3 models
for tag in ds15 qwen34 phi; do
  for p in 2 3; do
    for c in 1 2; do
      L=results/longloop_v3b_prompt/$tag/p$p/k$c/loop
      R=results/longloop_v3b_prompt/$tag/p$p/k$c/replay
      sl "$L" "$tag" 1.0 32 800 $((4100+c)) results/prompts_single/p$p.txt loop
      sl "$R" "$tag" 1.0 32 800 $((4100+c)) results/prompts_single/p$p.txt replay "$L"
    done
  done
done

# ---------------- E. long horizon with 32 directions
for tag in ds15 qwen34 phi; do
  L=results/longloop_v3b/$tag/long3000/loop
  R=results/longloop_v3b/$tag/long3000/replay
  sl "$L" "$tag" 1.0 32 3000 5100 @prompts.txt loop
  sl "$R" "$tag" 1.0 32 3000 5100 @prompts.txt replay "$L"
done

# ---------------- F. layer-0 runs (does the perturbation re-enter at the input?)
for tag in ds15 qwen34 randqwen; do
  L=results/longloop_v3b/$tag/layer0/loop
  R=results/longloop_v3b/$tag/layer0/replay
  echo "=== layer0 $tag $(date)"
  ${PYV[$tag]} -m chaos_llm.run_softloop --model "${MODEL[$tag]}" --steps 800 --temp 1.0 \
    --dirs 32 --r "${RL[$tag]}" --save-layers 0,${LL[$tag]} --dtype "${DT[$tag]}" \
    --window 4096 --out "$L" --feedback soft || echo "FAILED $L"
  ${PYV[$tag]} -m chaos_llm.run_softloop --model "${MODEL[$tag]}" --steps 800 --temp 1.0 \
    --dirs 32 --r "${RL[$tag]}" --save-layers 0,${LL[$tag]} --dtype "${DT[$tag]}" \
    --window 4096 --out "$R" --mode replay --replay-file "$L/softloop.npz" || echo "FAILED $R"
done

# ---------------- G. attention: all layers, four windows
for tag in $ORDER; do
  case $tag in
    ds15) AL="0,1,2,3,7,14,21,27";; qwen25) AL="0,1,2,9,18,27,35";;
    qwen34) AL="0,1,2,9,18,27,35";; qwen35) AL="0,1,2,6,12,18,23";;
    phi) AL="0,1,2,8,16,24,31";;   mamba) AL="0,1,2,6,12,18,23";;
    pythia) AL="0,1,2,6,12,18,23";; randqwen) AL="0,1,2,7,14,21,27";;
  esac
  for W in 50 100 200 4096; do
    D=results/update/attnv3_${tag}_w$W
    [ -f "$D/attn_frac.npz" ] && { echo "skip $D"; continue; }
    echo "=== attn $tag W=$W $(date)"
    ${PYV[$tag]} -m chaos_llm.run_attention_frac --model "${MODEL[$tag]}" --dtype "${DT[$tag]}" \
      --window "$W" --max-new-tokens 300 --layers "$AL" --out "$D" || echo "FAILED attn $tag $W"
  done
done
echo "V3B X11 DONE $(date)"
