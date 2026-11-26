#!/bin/bash
# Runs AFTER queue_long_x11.sh has finished. Two things the main campaign cannot
# cover: (a) 32-direction loops, so the closed-vs-replay figures get real
# percentiles instead of an 8-direction IQR; (b) multi-prompt replication of the
# loop, the single most important missing measurement (everything so far is one
# prompt).
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY0=python3
PY=~/LLM/venv_tf/bin/python

STEPS=3000
BIGSTEPS=1600

while pgrep -f queue_long_x11.sh > /dev/null; do
  echo "waiting for the main x11 queue $(date)"
  sleep 300
done

loop () { # out model tag py dtype r layers steps dirs prompt
  local out=$1 model=$2 py=$3 dt=$4 r=$5 layers=$6 steps=$7 dirs=$8 prompt=$9
  [ -f "$out/softloop.npz" ] && { echo "skip $out"; return; }
  echo "=== $out $(date)"
  $py -m chaos_llm.run_softloop --model "$model" --steps "$steps" --temp 1.0 \
    --dirs "$dirs" --r "$r" --feedback soft --save-layers "$layers" \
    --dtype "$dt" --window 4096 --prompt "$prompt" --out "$out" || echo "FAILED $out"
}
replay () { # out model py dtype r layers steps dirs loopdir
  local out=$1 model=$2 py=$3 dt=$4 r=$5 layers=$6 steps=$7 dirs=$8 src=$9
  [ -f "$out/softloop.npz" ] && { echo "skip $out"; return; }
  [ -f "$src/softloop.npz" ] || { echo "no loop for $out"; return; }
  echo "=== $out $(date)"
  $py -m chaos_llm.run_softloop --model "$model" --steps "$steps" --temp 1.0 \
    --dirs "$dirs" --r "$r" --mode replay --replay-file "$src/softloop.npz" \
    --save-layers "$layers" --dtype "$dt" --window 4096 --out "$out" || echo "FAILED $out"
}

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q34=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507

# (a) 32-direction loops: percentiles for the headline figure
loop results/longloop/ds15w32/T1.0/loop "$DS" $PY0 float16 0.03 7,14,21,27 $BIGSTEPS 32 @prompts.txt
replay results/longloop/ds15w32/T1.0/replay "$DS" $PY0 float16 0.03 7,14,21,27 $BIGSTEPS 32 results/longloop/ds15w32/T1.0/loop
loop results/longloop/qwen34w32/T1.0/loop "$Q34" $PY float16 0.03 9,18,27,35 $BIGSTEPS 32 @prompts.txt
replay results/longloop/qwen34w32/T1.0/replay "$Q34" $PY float16 0.03 9,18,27,35 $BIGSTEPS 32 results/longloop/qwen34w32/T1.0/loop

# (b) prompt replication (prompts 2 and 3 of prompts_more.txt)
for p in 2 3; do
  loop   results/longloop_prompt/ds15/p$p/T1.0/loop   "$DS" $PY0 float32 0.03 7,14,21,27 $STEPS 8 @results/prompts_single/p$p.txt
  replay results/longloop_prompt/ds15/p$p/T1.0/replay "$DS" $PY0 float32 0.03 7,14,21,27 $STEPS 8 results/longloop_prompt/ds15/p$p/T1.0/loop
  loop   results/longloop_prompt/qwen34/p$p/T1.0/loop   "$Q34" $PY float16 0.03 9,18,27,35 $STEPS 8 @results/prompts_single/p$p.txt
  replay results/longloop_prompt/qwen34/p$p/T1.0/replay "$Q34" $PY float16 0.03 9,18,27,35 $STEPS 8 results/longloop_prompt/qwen34/p$p/T1.0/loop
done
echo "AFTER-X11 DONE $(date)"
