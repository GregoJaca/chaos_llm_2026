#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
export CHAOS_DTYPE=float16
echo "QUEUE57 START $(date)"

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q4=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PH=~/LLM/models--microsoft--Phi-4-mini-instruct

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

echo "QUEUE57 DONE $(date)"
touch results/QUEUE57_DONE
