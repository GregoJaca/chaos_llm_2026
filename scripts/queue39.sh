#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY39=python3
PY=~/LLM/venv_tf/bin/python

while pgrep -f "wspec_run.py" > /dev/null; do sleep 20; done
echo "QUEUE39 START $(date)"

# large-r autoregressive flips (P1), reusing the bigr teacher-forced runs
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
echo "AR DONE $(date)"

# Qwen2.5-3B: the architecture control (dense + real 32k window)
if [ -d ~/LLM/models--Qwen--Qwen2.5-3B-Instruct ]; then
  bash scripts/model_suite2.sh ~/LLM/models--Qwen--Qwen2.5-3B-Instruct qwen25_3b 0.1 36
fi

echo "QUEUE39 DONE $(date)"
