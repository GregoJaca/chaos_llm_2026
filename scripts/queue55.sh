#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
export CHAOS_DTYPE=float16
echo "QUEUE55 START $(date)"

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q4=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PH=~/LLM/models--microsoft--Phi-4-mini-instruct

# 1. free-run token streams vs window (DeepSeek native window)
[ -f results/window_tokens/tokens.npz ] || \
  $PY39 -m chaos_llm.run_window_tokens --model $DS --prompt-file prompts.txt \
    --windows 60,100,150,200,300,500,1000,4096 --tokens 300 --dtype float32 \
    --out results/window_tokens || echo "FAILED window tokens"
sleep 5

# 2. repair the sensitivity grid (Qwen3-4B p4-p8, Phi p2-p4), strictly sequential
mkdir -p /tmp/PSET
head -1 prompts.txt > /tmp/PSET/p1.txt
head -1 prompts_cyber.txt > /tmp/PSET/p2.txt
for i in 1 2 3 4 5 6; do sed -n "${i}p" prompts_more2.txt > /tmp/PSET/p$((i+2)).txt; done

repair () {
  local M=$1 BIN=$2 DT=$3 LAST=$4 TAG=$5 P=$6
  local geom=results/sens/${TAG}/geom
  if [ ! -f "$geom/a1.npz" ]; then
    $BIN -m chaos_llm.run_a1 --model "$M" --prompt @$P --window 100 \
      --max-new-tokens 300 --num-directions 192 --radii 0.01,0.1 \
      --layers 7,14,$LAST --batch-dirs 8 --out "$geom" || echo "FAILED geom $TAG"
    sleep 5
  fi
  if [ ! -f "results/transport/${TAG}/transport.npz" ]; then
    $BIN -m chaos_llm.run_transport --model "$M" --run "$geom" \
      --positions all --dirs 8 --r 0.1 --layers 1,7,14,21,last --dtype "$DT" \
      --out results/transport/${TAG} || echo "FAILED transport $TAG"
    sleep 5
  fi
  for r in 0.1 0.3 0.6 1.0; do
    [ -f "results/sens/${TAG}/r$r/free_run.npz" ] && continue
    $BIN -m chaos_llm.run_thresholds --model "$M" --run "$geom" --radii $r \
      --chunk 16 --out results/sens/${TAG}/r$r || echo "FAILED sens $TAG $r"
    sleep 5
  done
  echo "$TAG done $(date)"
}

for i in 4 5 6 7 8; do repair $Q4 $PY float16 35 qwen3_4b_p$i /tmp/PSET/p$i.txt; done
for i in 2 3 4; do repair $PH $PY float16 31 phi4mini_p$i /tmp/PSET/p$i.txt; done

# 3. window W=300/500 with a long continuation (4D mask, 900 tokens)
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

echo "QUEUE55 DONE $(date)"
touch results/QUEUE55_DONE
