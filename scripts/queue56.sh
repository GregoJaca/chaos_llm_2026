#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
export CHAOS_DTYPE=float16
echo "QUEUE56 START $(date)"

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q4=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PH=~/LLM/models--microsoft--Phi-4-mini-instruct
Q35=~/LLM/models--Qwen--Qwen3.5-2B
Q25=~/LLM/models--Qwen--Qwen2.5-3B-Instruct

# 1. the one missing qwen3_4b p3 radius
for r in 0.1 0.3 0.6 1.0; do
  out=results/sens/qwen3_4b_p3/r$r
  [ -f "$out/free_run.npz" ] && continue
  $PY -m chaos_llm.run_thresholds --model $Q4 --run results/sens/qwen3_4b_p3/geom \
    --radii $r --chunk 16 --out "$out" || echo "FAILED q4 p3 $r"
  sleep 5
done

# 2. superposition at r=0.03 for the four remaining models
for m in qwen3_4b qwen25_3b phi4mini qwen3_5; do
  case $m in
    qwen3_4b) M=$Q4; BIN=$PY; DT=float16 ;;
    qwen25_3b) M=$Q25; BIN=$PY; DT=float16 ;;
    phi4mini) M=$PH; BIN=$PY; DT=float16 ;;
    qwen3_5) M=$Q35; BIN=$PY311; DT=float16 ;;
  esac
  out=results/superposition/${m}_r003
  [ -f "$out/superposition.npz" ] && continue
  $BIN -m chaos_llm.run_superposition --model "$M" --run results/sens/${m}_p1/geom \
    --trials 3 --r 0.03 --layers 1,7,14,last --dtype "$DT" --out "$out" \
    || echo "FAILED sup $m"
  sleep 8
done

# 3. window W=300/500 retries
for m in deepseek qwen3_4b phi4mini; do
  case $m in
    deepseek) M=$DS; BIN=$PY39; DT=float32; LAST=27 ;;
    qwen3_4b) M=$Q4; BIN=$PY; DT=float16; LAST=35 ;;
    phi4mini) M=$PH; BIN=$PY; DT=float16; LAST=31 ;;
  esac
  for w in 300 500; do
    out=results/window_long/${m}_w$w
    [ -f "$out/a1.npz" ] && continue
    $BIN -m chaos_llm.run_a1_long --model "$M" --prompt @prompts.txt --window $w \
      --max-new-tokens 900 --fixed-tokens-from results/long4k/p1.npz \
      --num-directions 4 --radii 0.1 --layers 1,2,3,7,14,21,$LAST \
      --window4d --dtype "$DT" --out "$out" || echo "FAILED window $m $w"
    sleep 5
  done
done

echo "QUEUE56 DONE $(date)"
touch results/QUEUE56_DONE
