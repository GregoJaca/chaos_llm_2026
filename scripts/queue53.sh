#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
export CHAOS_DTYPE=float16
echo "QUEUE53 START $(date)"

while pgrep -f "queue52.sh" > /dev/null; do sleep 60; done
sleep 15

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q4=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PH=~/LLM/models--microsoft--Phi-4-mini-instruct
Q35=~/LLM/models--Qwen--Qwen3.5-2B
Q25=~/LLM/models--Qwen--Qwen2.5-3B-Instruct

# 1. superposition test: same direction at every prompt position
for m in deepseek qwen3_4b phi4mini qwen3_5 qwen25_3b; do
  case $m in
    deepseek) M=$DS; BIN=$PY39; DT=float32 ;;
    qwen3_4b) M=$Q4; BIN=$PY; DT=float16 ;;
    phi4mini) M=$PH; BIN=$PY; DT=float16 ;;
    qwen3_5) M=$Q35; BIN=$PY311; DT=float16 ;;
    qwen25_3b) M=$Q25; BIN=$PY; DT=float16 ;;
  esac
  out=results/superposition/${m}_r003
  [ -f "$out/superposition.npz" ] && continue
  $BIN -m chaos_llm.run_superposition --model "$M" --run results/sens/${m}_p1/geom \
    --trials 3 --r 0.03 --layers 1,7,14,last --dtype "$DT" --out "$out" \
    || echo "FAILED superposition $m"
  sleep 10
done
echo "superposition done $(date)"

# 2. window sweep at W=300,500 with a long fixed continuation (1000 tokens)
for m in deepseek qwen3_4b phi4mini; do
  case $m in
    deepseek) M=$DS; BIN=$PY39; DT=float32 ;;
    qwen3_4b) M=$Q4; BIN=$PY; DT=float16 ;;
    phi4mini) M=$PH; BIN=$PY; DT=float16 ;;
  esac
  for w in 300 500; do
    out=results/window_long/${m}_w$w
    [ -f "$out/a1.npz" ] && continue
    $BIN -m chaos_llm.run_a1_long --model "$M" --prompt @prompts.txt --window $w \
      --max-new-tokens 900 --fixed-tokens-from results/long4k/p1.npz \
      --num-directions 4 --radii 0.1 --layers 1,2,3,7,14,21,last \
      --window4d --dtype "$DT" --out "$out" || echo "FAILED window $m $w"
  done
done
echo "window long done $(date)"

# 3. position scan on three more prompts (hot spots)
probe () {
  local M=$1 RUN=$2 TAG=$3 BIN=$4 DT=$5 R=$6
  for pos in 4 8 12 16 20 25 30 40 50 60 80 100; do
    out=results/probe4/$TAG/pos$pos
    [ -f "$out/layer_probe.npz" ] && continue
    $BIN -m chaos_llm.run_layer_probe --model "$M" --run "$RUN" --out "$out" \
      --layers 0,5 --dirs 8 --r "$R" --pos $pos --dtype "$DT" > /dev/null 2>&1 \
      || echo "FAILED probe4 $TAG $pos"
  done
}
for i in 2 3 4; do
  probe $DS results/sens/deepseek_p$i/geom deepseek_p$i $PY39 float32 0.1
  probe $Q4 results/sens/qwen3_4b_p$i/geom qwen3_4b_p$i $PY float16 0.1
  probe $PH results/sens/phi4mini_p$i/geom phi4mini_p$i $PY float16 0.3
done

echo "QUEUE53 DONE $(date)"
touch results/QUEUE53_DONE
