#!/bin/bash
# Runs AFTER queue_long_spark.sh. Adds prompt replication for the two 14B models
# (everything in the report is one prompt) and one 16-direction loop so the
# 14B curves get a percentile band.
set -u
cd /home/grego/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/grego/LLM/venv5/bin/python
STEPS=1600

while pgrep -f queue_long_spark.sh > /dev/null; do
  echo "waiting for the main spark queue $(date)"
  sleep 300
done

loop () { # out model r layers steps dirs prompt
  local out=$1 model=$2 r=$3 layers=$4 steps=$5 dirs=$6 prompt=$7
  [ -f "$out/softloop.npz" ] && { echo "skip $out"; return; }
  echo "=== $out $(date)"
  $PY -m chaos_llm.run_softloop --model "$model" --steps "$steps" --temp 1.0 \
    --dirs "$dirs" --r "$r" --feedback soft --save-layers "$layers" \
    --dtype float16 --window 4096 --prompt "$prompt" --out "$out" || echo "FAILED $out"
}
replay () { # out model r layers steps dirs src
  local out=$1 model=$2 r=$3 layers=$4 steps=$5 dirs=$6 src=$7
  [ -f "$out/softloop.npz" ] && { echo "skip $out"; return; }
  [ -f "$src/softloop.npz" ] || { echo "no loop for $out"; return; }
  echo "=== $out $(date)"
  $PY -m chaos_llm.run_softloop --model "$model" --steps "$steps" --temp 1.0 \
    --dirs "$dirs" --r "$r" --mode replay --replay-file "$src/softloop.npz" \
    --save-layers "$layers" --dtype float16 --window 4096 --out "$out" || echo "FAILED $out"
}

Q314=/home/grego/LLM/models/Qwen3-14B
DS14=/home/grego/LLM/models/DeepSeek-R1-Distill-Qwen-14B

# 16-direction loop for the headline figure's percentile band
loop   results/longloop/qwen314w16/T1.0/loop "$Q314" 0.1 10,20,30,39 $STEPS 16 @prompts.txt
replay results/longloop/qwen314w16/T1.0/replay "$Q314" 0.1 10,20,30,39 $STEPS 16 results/longloop/qwen314w16/T1.0/loop

# prompt replication at 14B
for p in 2 3; do
  loop   results/longloop_prompt/qwen314/p$p/T1.0/loop "$Q314" 0.1 10,20,30,39 $STEPS 8 @results/prompts_single/p$p.txt
  replay results/longloop_prompt/qwen314/p$p/T1.0/replay "$Q314" 0.1 10,20,30,39 $STEPS 8 results/longloop_prompt/qwen314/p$p/T1.0/loop
  loop   results/longloop_prompt/ds14/p$p/T1.0/loop "$DS14" 0.1 12,24,36,47 $STEPS 8 @results/prompts_single/p$p.txt
  replay results/longloop_prompt/ds14/p$p/T1.0/replay "$DS14" 0.1 12,24,36,47 $STEPS 8 results/longloop_prompt/ds14/p$p/T1.0/loop
done
echo "AFTER-SPARK DONE $(date)"
