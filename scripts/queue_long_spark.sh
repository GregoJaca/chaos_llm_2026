#!/bin/bash
# Looping campaign on Spark (GB10, unified memory). Rounds, balanced coverage.
# 14B models run at ~3.4 s/step, so 1600 steps keeps every run at ~2 h.
set -u
cd /home/grego/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/grego/LLM/venv5/bin/python

STEPS=${STEPS:-1600}

declare -A MODEL RL LL
MODEL[qwen314]=/home/grego/LLM/models/Qwen3-14B; RL[qwen314]=0.1; LL[qwen314]=10,20,30,39
MODEL[ds14]=/home/grego/LLM/models/DeepSeek-R1-Distill-Qwen-14B; RL[ds14]=0.1; LL[ds14]=12,24,36,47
MODEL[phi4]=/home/grego/LLM/models/phi-4; RL[phi4]=0.1; LL[phi4]=10,20,30,39
MODEL[qwen35_9b]=/home/grego/LLM/models/Qwen3.5-9B; RL[qwen35_9b]=0.1; LL[qwen35_9b]=8,16,24,31

ORDER="qwen314 ds14 qwen35_9b phi4"

loop () {
  local tag=$1 temp=$2
  local D=results/longloop/$tag/T$temp/loop
  [ -f "$D/softloop.npz" ] && { echo "skip $tag T$temp loop"; return; }
  echo "=== loop $tag T$temp $(date)"
  $PY -m chaos_llm.run_softloop --model "${MODEL[$tag]}" --steps "$STEPS" --temp "$temp" \
    --dirs 8 --r "${RL[$tag]}" --feedback soft --save-layers "${LL[$tag]}" \
    --dtype float16 --window 4096 --out "$D" || echo "FAILED $tag T$temp loop"
}
replay () {
  local tag=$1 temp=$2
  local L=results/longloop/$tag/T$temp/loop
  local D=results/longloop/$tag/T$temp/replay
  [ -f "$D/softloop.npz" ] && { echo "skip $tag T$temp replay"; return; }
  [ -f "$L/softloop.npz" ] || { echo "no loop $tag T$temp"; return; }
  echo "=== replay $tag T$temp $(date)"
  $PY -m chaos_llm.run_softloop --model "${MODEL[$tag]}" --steps "$STEPS" --temp "$temp" \
    --dirs 8 --r "${RL[$tag]}" --mode replay --replay-file "$L/softloop.npz" \
    --save-layers "${LL[$tag]}" --dtype float16 --window 4096 --out "$D" \
    || echo "FAILED $tag T$temp replay"
}
tf () {
  local tag=$1 ntok=${2:-800}
  local D=results/longtf/$tag
  [ -f "$D/a1.npz" ] && { echo "skip tf $tag"; return; }
  echo "=== tf $tag $(date)"
  $PY -m chaos_llm.run_a1 --model "${MODEL[$tag]}" --dtype float16 --prompt @prompts.txt \
    --window 4096 --max-new-tokens "$ntok" --num-directions 8 --radii "${RL[$tag]}" \
    --layers "${LL[$tag]}" --batch-dirs 1 --out "$D" || echo "FAILED tf $tag"
}

# A: top priority, closed vs replay at T=1
for tag in $ORDER; do loop "$tag" 1.0; done
for tag in $ORDER; do replay "$tag" 1.0; done
# B: temperature
for tag in $ORDER; do loop "$tag" 2.0; done
# C: long teacher-forced horizon
for tag in $ORDER; do tf "$tag" 800; done
# D: replay at T=2 (uses the orbits from B)
for tag in $ORDER; do replay "$tag" 2.0; done
# E: third temperature
for tag in $ORDER; do loop "$tag" 0.5; done
echo "LONG SPARK DONE $(date)"
