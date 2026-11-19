#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
export CHAOS_DTYPE=float16
echo "QUEUE41 START $(date)"

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q4=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PH=~/LLM/models--microsoft--Phi-4-mini-instruct
Q35=~/LLM/models--Qwen--Qwen3.5-2B
Q25=~/LLM/models--Qwen--Qwen2.5-3B-Instruct

probe () { # model run tag py dtype layers r
  local M=$1 RUN=$2 TAG=$3 PYBIN=$4 DT=$5 LAYERS=$6 R=$7
  for pos in 4 8 12 16 20 25 30 40 50 60 80 100; do
    out=results/probe2/$TAG/pos$pos
    [ -f "$out/layer_probe.npz" ] && continue
    $PYBIN -m chaos_llm.run_layer_probe --model "$M" --run "$RUN" --out "$out" \
      --layers "$LAYERS" --dirs 6 --r "$R" --pos $pos --dtype "$DT" > /dev/null 2>&1 \
      || echo "FAILED probe $TAG $pos"
  done
  echo "probe $TAG done $(date)"
}

probe $DS results/big4/a1_all_layers/prompts deepseek $PY39 float32 0,5,10 0.1
probe $Q4 results/models/qwen3_4b/a1 qwen3_4b $PY float16 0,5,10 0.1
probe $PH results/models/phi4mini/a1 phi4mini $PY float16 0,5,10 0.3
probe $Q35 results/models/qwen3_5/a1 qwen3_5 $PY311 float16 0,5,10 0.1
probe $Q25 results/models/qwen25_3b/a1 qwen25_3b $PY float16 0,5,10 0.1

# --- multi-radius runs for the FSLE, every model
for i in 1 2; do
  sed -n "${i}p" prompts_more2.txt > /tmp/FS_$i.txt
  for m in ds q4 phi q35 q25; do
    case $m in
      ds)  M=$DS;  BIN=$PY39;  DT=float32; L=27; TAG=deepseek ;;
      q4)  M=$Q4;  BIN=$PY;    DT=float16; L=35; TAG=qwen3_4b ;;
      phi) M=$PH;  BIN=$PY;    DT=float16; L=31; TAG=phi4mini ;;
      q35) M=$Q35; BIN=$PY311; DT=float16; L=23; TAG=qwen3_5 ;;
      q25) M=$Q25; BIN=$PY;    DT=float16; L=35; TAG=qwen25_3b ;;
    esac
    out=results/fsle/$TAG/p$i
    [ -f "$out/a1.npz" ] && continue
    $BIN -m chaos_llm.run_a1 --model "$M" --prompt @/tmp/FS_$i.txt --window 100 \
      --max-new-tokens 300 --num-directions 8 --radii 0.001,0.01,0.1,0.3,1.0 \
      --layers $((L-1)),$L --batch-dirs 1 --out "$out" || echo "FAILED fsle $TAG $i"
  done
done
echo "fsle runs done $(date)"

# --- window x nonlinearity at r=0.3 and 1.0
for m in q4 phi; do
  case $m in
    q4)  M=$Q4;  TAG=qwen3_4b; LAYERS=1,2,3,4,7,14,21,27,31,35 ;;
    phi) M=$PH;  TAG=phi4mini; LAYERS=1,2,3,4,7,14,21,27,29,31 ;;
  esac
  for w in 60 100 200 4096; do
    out=results/window_r03/$TAG/w$w
    [ -f "$out/a1.npz" ] && continue
    $PY -m chaos_llm.run_a1 --model "$M" --prompt @prompts.txt --window $w --window4d \
      --max-new-tokens 300 --num-directions 6 --radii 0.3,1.0 \
      --layers $LAYERS --batch-dirs 1 --out "$out" || echo "FAILED window r03 $TAG $w"
  done
done

echo "QUEUE41 DONE $(date)"
