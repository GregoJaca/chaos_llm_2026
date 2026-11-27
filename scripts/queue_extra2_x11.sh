#!/bin/bash
# Extra x11 work, run after v3 + the top-k probe have finished.
#   1. Qwen3.5-2B in fp32 (its fp16 path hits CUDNN_STATUS_NOT_INITIALIZED)
#   2. loop runs that also save LAYER 0 (the input embedding) - this is the direct
#      measurement of "the perturbation re-enters through the feedback, not the
#      token lookup": in the closed loop dh(layer 0) > 0, in replay it is 0.
#   3. A1 with 8 layers at W=100 for every small model (the 100-token staircase)
#      plus a W=4096 control.
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY0=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311/bin/python

while pgrep -f queue_v3_x11.sh > /dev/null; do echo "wait v3 $(date)"; sleep 300; done
while pgrep -f queue_topk_x11.sh > /dev/null; do echo "wait topk $(date)"; sleep 300; done

sl () { # out model py dtype dirs steps layers win mode loopdir
  local out=$1 model=$2 py=$3 dt=$4 dirs=$5 steps=$6 layers=$7 win=$8 mode=$9
  local loopdir=${10:-}
  [ -f "$out/softloop.npz" ] && { echo "skip $out"; return; }
  if [ "$mode" = "replay" ] && [ ! -f "$loopdir/softloop.npz" ]; then echo "no loop $out"; return; fi
  echo "=== $out $(date)"
  if [ "$mode" = "replay" ]; then
    $py -m chaos_llm.run_softloop --model "$model" --steps "$steps" --temp 1.0 \
      --dirs "$dirs" --r 0.03 --save-layers "$layers" --dtype "$dt" --window "$win" \
      --out "$out" --mode replay --replay-file "$loopdir/softloop.npz" || echo "FAILED $out"
  else
    $py -m chaos_llm.run_softloop --model "$model" --steps "$steps" --temp 1.0 \
      --dirs "$dirs" --r 0.03 --save-layers "$layers" --dtype "$dt" --window "$win" \
      --feedback soft --out "$out" || echo "FAILED $out"
  fi
}

Q35=~/LLM/models--Qwen--Qwen3.5-2B
DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B

# --- 1. Qwen3.5-2B in fp32, 64 directions
for T in 1.0 2.0; do
  L=results/longloop_v3/qwen35/T$T/k1/loop
  R=results/longloop_v3/qwen35/T$T/k1/replay
  [ "$T" = "1.0" ] && { sl "$L" "$Q35" $PY311 float32 64 1200 6,12,18,23 4096 loop
                        sl "$R" "$Q35" $PY311 float32 64 1200 6,12,18,23 4096 replay "$L"; } || \
    sl "$L" "$Q35" $PY311 float32 64 1200 6,12,18,23 4096 loop
done

# --- 2. layer-0 runs (input embedding)
sl results/longloop_v3/ds15/layer0/loop "$DS" $PY0 float16 32 1200 0,7,14,21,27 4096 loop
sl results/longloop_v3/ds15/layer0/replay "$DS" $PY0 float16 32 1200 0,7,14,21,27 4096 replay results/longloop_v3/ds15/layer0/loop
sl results/longloop_v3/randqwen/layer0/loop ~/LLM/models--randqwen-g1.0 $PY0 float16 32 1200 0,7,14,21,27 4096 loop

# --- 3. A1 with all layers at W=100 (staircase) and a W=4096 control
declare -A M2 P2 DT2 R2 L2
M2[deepseek]=$DS;                 P2[deepseek]=$PY0; DT2[deepseek]=float32; R2[deepseek]=0.03; L2[deepseek]=0,1,2,3,6,9,12,18,24,27
M2[qwen25_3b]=~/LLM/models--Qwen--Qwen2.5-3B-Instruct; P2[qwen25_3b]=$PY; DT2[qwen25_3b]=float32; R2[qwen25_3b]=0.03; L2[qwen25_3b]=0,1,2,3,8,16,24,35
M2[qwen3_4b]=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507; P2[qwen3_4b]=$PY; DT2[qwen3_4b]=float16; R2[qwen3_4b]=0.03; L2[qwen3_4b]=0,1,2,3,8,16,24,35
M2[qwen3_5]=$Q35;                 P2[qwen3_5]=$PY311; DT2[qwen3_5]=float32; R2[qwen3_5]=0.03; L2[qwen3_5]=0,1,2,3,6,12,18,23
M2[phi4mini]=~/LLM/models--microsoft--Phi-4-mini-instruct; P2[phi4mini]=$PY; DT2[phi4mini]=float16; R2[phi4mini]=0.1; L2[phi4mini]=0,1,2,3,8,16,24,31

for tag in deepseek qwen25_3b qwen3_4b qwen3_5 phi4mini; do
  for W in 100 4096; do
    D=results/long_a1/${tag}_w$W
    [ -f "$D/a1.npz" ] && { echo "skip $D"; continue; }
    echo "=== a1 $tag W=$W $(date)"
    ${P2[$tag]} -m chaos_llm.run_a1 --model "${M2[$tag]}" --dtype "${DT2[$tag]}" --prompt @prompts.txt \
      --window $W --max-new-tokens 300 --num-directions 128 --radii "${R2[$tag]}" \
      --layers "${L2[$tag]}" --batch-dirs 2 --out "$D" || echo "FAILED a1 $tag $W"
  done
done
echo "EXTRA2 X11 DONE $(date)"
