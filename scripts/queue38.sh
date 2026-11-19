#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python
echo "QUEUE38 START $(date)"

# ---------------- A. window spectroscopy (P3) ----------------
# Sweep W by 1 around the transition, then coarser.  One prompt, all informative layers.
for w in $(seq 60 139) $(seq 140 5 400); do
  out=results/big4/wspec/w$w
  [ -f "$out/a1.npz" ] && continue
  $PY39 -m chaos_llm.run_a1 --prompt @prompts.txt --window $w --window4d \
    --max-new-tokens 200 --num-directions 4 --radii 0.1 \
    --layers 1,2,3,7,14,21,27 --batch-dirs 1 --out "$out" > /dev/null 2>&1 \
    || echo "FAILED wspec $w"
done
echo "WSPEC DONE $(date) ($(ls results/big4/wspec | wc -l) windows)"

# ---------------- B. large-r autoregressive flips (P1) ----------------
for m in deepseek qwen phi; do
  case $m in
    deepseek) M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B; BIN=$PY39; RUN=results/big4/bigr ;;
    qwen)     M=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507; BIN=$PY; RUN=results/models/qwen3_4b/bigr ;;
    phi)      M=~/LLM/models--microsoft--Phi-4-mini-instruct; BIN=$PY; RUN=results/models/phi4mini/bigr ;;
  esac
  for p in 1 2; do
    for r in 0.5 0.8 1.2 2.0; do
      out=${RUN}/ar_p${p}_r${r}
      [ -f "$out/free_run.npz" ] && continue
      $BIN -m chaos_llm.run_thresholds --model "$M" --run ${RUN}/p${p} \
        --radii $r --chunk 6 --out "$out" > /dev/null 2>&1 || echo "FAILED ar $m p$p r$r"
    done
    echo "AR $m prompt $p done $(date)"
  done
done

echo "QUEUE38 DONE $(date)"
