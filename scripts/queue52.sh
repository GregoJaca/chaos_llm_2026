#!/bin/bash
# Programme A (transport kernel) + B (sensitivity law), 8 prompts x 5 models.
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
PY311=~/LLM/venv311b/bin/python
export CHAOS_DTYPE=float16
echo "QUEUE52 START $(date)"

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q4=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PH=~/LLM/models--microsoft--Phi-4-mini-instruct
Q35=~/LLM/models--Qwen--Qwen3.5-2B
Q25=~/LLM/models--Qwen--Qwen2.5-3B-Instruct

mkdir -p /tmp/PSET
head -1 prompts.txt > /tmp/PSET/p1.txt
head -1 prompts_cyber.txt > /tmp/PSET/p2.txt
for i in 1 2 3 4 5 6; do sed -n "${i}p" prompts_more2.txt > /tmp/PSET/p$((i+2)).txt; done

model_for () {
  case $1 in
    deepseek) echo "$DS|$PY39|float32|27" ;;
    qwen3_4b) echo "$Q4|$PY|float16|35" ;;
    phi4mini) echo "$PH|$PY|float16|31" ;;
    qwen3_5)  echo "$Q35|$PY311|float16|23" ;;
    qwen25_3b) echo "$Q25|$PY|float16|35" ;;
  esac
}

for m in deepseek qwen3_4b phi4mini qwen3_5 qwen25_3b; do
  IFS='|' read -r M BIN DT LAST <<< "$(model_for $m)"
  for i in 1 2 3 4 5 6 7 8; do
    P=/tmp/PSET/p$i.txt
    [ -s "$P" ] || continue
    geom=results/sens/${m}_p$i/geom

    # 1. geometry run: 192 directions, teacher-forced (also provides prompt/gen ids)
    if [ ! -f "$geom/a1.npz" ]; then
      $BIN -m chaos_llm.run_a1 --model "$M" --prompt @$P --window 100 \
        --max-new-tokens 300 --num-directions 192 --radii 0.01,0.1 \
        --layers 7,14,$LAST --batch-dirs 8 --out "$geom" || echo "FAILED geom $m p$i"
    fi

    # 2. transport kernel, all prompt positions
    out=results/transport/${m}_p$i
    if [ ! -f "$out/transport.npz" ]; then
      $BIN -m chaos_llm.run_transport --model "$M" --run "$geom" \
        --positions all --dirs 8 --r 0.1 --layers 1,7,14,21,last --dtype "$DT" \
        --out "$out" || echo "FAILED transport $m p$i"
    fi

    # 3. free runs at four radii (sensitivity law)
    for r in 0.1 0.3 0.6 1.0; do
      out=results/sens/${m}_p$i/r$r
      [ -f "$out/free_run.npz" ] && continue
      $BIN -m chaos_llm.run_thresholds --model "$M" --run "$geom" --radii $r \
        --chunk 16 --out "$out" || echo "FAILED sens $m p$i r$r"
    done
    echo "$m prompt $i done $(date)"
  done
done

echo "QUEUE52 DONE $(date)"
touch results/QUEUE52_DONE
