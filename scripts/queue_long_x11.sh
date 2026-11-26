#!/bin/bash
# Looping campaign on x11 (T4, 15 GB). Rounds, so coverage is balanced:
#   A: T=1.0 loop + replay for every model   (top priority: closed vs replay log-log)
#   B: T=2.0 loop + replay
#   C: CRN + independent sampled feedback at T=1
#   D: long teacher-forced runs (1000 tokens)
#   E: T=0.5 loop
# Resume-safe (skips finished runs). No retries, no memory-gating games.
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY0=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311/bin/python

STEPS=${STEPS:-3000}

declare -A MODEL PYV DT RL LL
MODEL[ds15]=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B; PYV[ds15]=$PY0; DT[ds15]=float32; RL[ds15]=0.03; LL[ds15]=7,14,21,27
MODEL[qwen25]=~/LLM/models--Qwen--Qwen2.5-3B-Instruct; PYV[qwen25]=$PY; DT[qwen25]=float32; RL[qwen25]=0.03; LL[qwen25]=9,18,27,35
MODEL[qwen34]=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507; PYV[qwen34]=$PY; DT[qwen34]=float16; RL[qwen34]=0.03; LL[qwen34]=9,18,27,35
MODEL[phi]=~/LLM/models--microsoft--Phi-4-mini-instruct; PYV[phi]=$PY; DT[phi]=float16; RL[phi]=0.1; LL[phi]=8,16,24,31
MODEL[mamba]=~/LLM/models--state-spaces--mamba-130m-hf; PYV[mamba]=$PY0; DT[mamba]=float32; RL[mamba]=0.03; LL[mamba]=6,12,18,23
MODEL[pythia]=~/LLM/models--EleutherAI--pythia-410m; PYV[pythia]=$PY; DT[pythia]=float32; RL[pythia]=0.1; LL[pythia]=6,12,18,23
MODEL[randqwen]=~/LLM/models--randqwen-g1.0; PYV[randqwen]=$PY0; DT[randqwen]=float32; RL[randqwen]=0.03; LL[randqwen]=7,14,21,27
MODEL[qwen35]=~/LLM/models--Qwen--Qwen3.5-2B; PYV[qwen35]=$PY311; DT[qwen35]=float32; RL[qwen35]=0.03; LL[qwen35]=6,12,18,23

ORDER="ds15 qwen25 qwen34 qwen35 phi mamba pythia randqwen"

loop () { # tag temp
  local tag=$1 temp=$2
  local D=results/longloop/$tag/T$temp/loop
  [ -f "$D/softloop.npz" ] && { echo "skip $tag T$temp loop"; return; }
  echo "=== loop $tag T$temp $(date)"
  ${PYV[$tag]} -m chaos_llm.run_softloop --model "${MODEL[$tag]}" --steps "$STEPS" \
    --temp "$temp" --dirs 8 --r "${RL[$tag]}" --feedback soft \
    --save-layers "${LL[$tag]}" --dtype "${DT[$tag]}" --window 4096 --out "$D" \
    || echo "FAILED $tag T$temp loop"
}
replay () { # tag temp
  local tag=$1 temp=$2
  local L=results/longloop/$tag/T$temp/loop
  local D=results/longloop/$tag/T$temp/replay
  [ -f "$D/softloop.npz" ] && { echo "skip $tag T$temp replay"; return; }
  [ -f "$L/softloop.npz" ] || { echo "no loop for $tag T$temp"; return; }
  echo "=== replay $tag T$temp $(date)"
  ${PYV[$tag]} -m chaos_llm.run_softloop --model "${MODEL[$tag]}" --steps "$STEPS" \
    --temp "$temp" --dirs 8 --r "${RL[$tag]}" --mode replay --replay-file "$L/softloop.npz" \
    --save-layers "${LL[$tag]}" --dtype "${DT[$tag]}" --window 4096 --out "$D" \
    || echo "FAILED $tag T$temp replay"
}
sample () { # tag feedback
  local tag=$1 fb=$2
  local D=results/longloop/$tag/T1.0/${fb#sample_}
  [ -f "$D/softloop.npz" ] && { echo "skip $tag $fb"; return; }
  echo "=== $fb $tag $(date)"
  ${PYV[$tag]} -m chaos_llm.run_softloop --model "${MODEL[$tag]}" --steps "$STEPS" \
    --temp 1.0 --dirs 8 --r "${RL[$tag]}" --feedback "$fb" \
    --save-layers "${LL[$tag]}" --dtype "${DT[$tag]}" --window 4096 --out "$D" \
    || echo "FAILED $tag $fb"
}
tf () { # tag tokens
  local tag=$1 ntok=${2:-1000}
  local D=results/longtf/$tag
  [ -f "$D/a1.npz" ] && { echo "skip tf $tag"; return; }
  echo "=== tf $tag $(date)"
  ${PYV[$tag]} -m chaos_llm.run_a1 --model "${MODEL[$tag]}" --dtype "${DT[$tag]}" --prompt @prompts.txt \
    --window 4096 --max-new-tokens "$ntok" --num-directions 8 --radii "${RL[$tag]}" \
    --layers "${LL[$tag]}" --batch-dirs 2 --out "$D" || echo "FAILED tf $tag"
}

# ---- round A: the top-priority comparison
for tag in $ORDER; do loop "$tag" 1.0; done
for tag in $ORDER; do replay "$tag" 1.0; done
# ---- round B: temperature dependence
for tag in $ORDER; do loop "$tag" 2.0; done
for tag in $ORDER; do replay "$tag" 2.0; done
# ---- round C: sampled-feedback controls
for tag in $ORDER; do sample "$tag" sample_crn; done
for tag in $ORDER; do sample "$tag" sample_ind; done
# ---- round D: long teacher-forced horizon
for tag in $ORDER; do tf "$tag" 1000; done
# ---- round E: the other temperature
for tag in $ORDER; do loop "$tag" 0.5; done
for tag in $ORDER; do replay "$tag" 0.5; done
echo "LONG X11 DONE $(date)"
