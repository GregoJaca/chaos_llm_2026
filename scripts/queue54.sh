#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
export CHAOS_DTYPE=float16
echo "QUEUE54 START $(date)"

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q4=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PH=~/LLM/models--microsoft--Phi-4-mini-instruct
Q35=~/LLM/models--Qwen--Qwen3.5-2B
Q25=~/LLM/models--Qwen--Qwen2.5-3B-Instruct

# wait for free memory (another agent may be using the GPU)
wait_gpu () {
  for i in $(seq 1 240); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free" -gt 9000 ]; then return 0; fi
    sleep 30
  done
  return 1
}

model_for () {
  case $1 in
    deepseek) echo "$DS|$PY39|float32|27|0.1" ;;
    qwen3_4b) echo "$Q4|$PY|float16|35|0.1" ;;
    phi4mini) echo "$PH|$PY|float16|31|0.3" ;;
    qwen3_5)  echo "$Q35|$PY311|float16|23|0.1" ;;
    qwen25_3b) echo "$Q25|$PY|float16|35|0.1" ;;
  esac
}

# ---------------- 1. repair the sensitivity grid
for m in deepseek qwen3_4b phi4mini qwen3_5 qwen25_3b; do
  IFS='|' read -r M BIN DT LAST RR <<< "$(model_for $m)"
  for i in 1 2 3 4 5 6 7 8; do
    P=/tmp/PSET/p$i.txt
    [ -s "$P" ] || continue
    geom=results/sens/${m}_p$i/geom
    if [ ! -f "$geom/a1.npz" ]; then
      wait_gpu || echo "no gpu mem"
      $BIN -m chaos_llm.run_a1 --model "$M" --prompt @$P --window 100 \
        --max-new-tokens 300 --num-directions 192 --radii 0.01,0.1 \
        --layers 7,14,$LAST --batch-dirs 8 --out "$geom" || echo "FAILED geom $m $i"
      sleep 5
    fi
    out=results/transport/${m}_p$i
    if [ ! -f "$out/transport.npz" ]; then
      wait_gpu
      $BIN -m chaos_llm.run_transport --model "$M" --run "$geom" \
        --positions all --dirs 8 --r 0.1 --layers 1,7,14,21,last --dtype "$DT" \
        --out "$out" || echo "FAILED transport $m $i"
      sleep 5
    fi
    for r in 0.1 0.3 0.6 1.0; do
      out=results/sens/${m}_p$i/r$r
      [ -f "$out/free_run.npz" ] && continue
      wait_gpu
      $BIN -m chaos_llm.run_thresholds --model "$M" --run "$geom" --radii $r \
        --chunk 16 --out "$out" || echo "FAILED sens $m $i $r"
      sleep 5
    done
    echo "$m p$i done $(date)"
  done
done
echo "GRID DONE $(date)"

# ---------------- 2. superposition at r=0.03, all models
for m in deepseek qwen3_4b phi4mini qwen3_5 qwen25_3b; do
  IFS='|' read -r M BIN DT LAST RR <<< "$(model_for $m)"
  out=results/superposition/${m}_r003
  [ -f "$out/superposition.npz" ] && continue
  wait_gpu
  $BIN -m chaos_llm.run_superposition --model "$M" --run results/sens/${m}_p1/geom \
    --trials 3 --r 0.03 --layers 1,7,14,last --dtype "$DT" --out "$out" \
    || echo "FAILED sup $m"
  sleep 5
done

# ---------------- 3. window at W=300,500 with a long continuation
for m in deepseek qwen3_4b phi4mini; do
  IFS='|' read -r M BIN DT LAST RR <<< "$(model_for $m)"
  for w in 300 500; do
    out=results/window_long/${m}_w$w
    [ -f "$out/a1.npz" ] && continue
    wait_gpu
    $BIN -m chaos_llm.run_a1_long --model "$M" --prompt @prompts.txt --window $w \
      --max-new-tokens 900 --fixed-tokens-from results/long4k/p1.npz \
      --num-directions 4 --radii 0.1 --layers 1,2,3,7,14,21,$LAST \
      --window4d --dtype "$DT" --out "$out" || echo "FAILED window $m $w"
    sleep 5
  done
done

echo "QUEUE54 DONE $(date)"
touch results/QUEUE54_DONE
