#!/bin/bash
# Control study, stage 4: random-initialized Qwen2 (DeepSeek-1.5B architecture),
# ordered-vs-chaotic phase sweep over the residual-update init gain, plus the
# Pythia-410m step0 (random init at standard scale) checkpoint.
# Waits for stage 3 to release the GPU. Resume-safe.
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv_tf/bin/python
PYSTEP0=~/LLM/models--EleutherAI--pythia-410m-step0

echo "=== stage4 start $(date)"
while pgrep -f ctrl_stage3.sh > /dev/null 2>&1; do sleep 60; done
echo "stage3 finished, GPU free $(date)"

RADW="3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1"

for G in 1.0 0.5 2.0 0.35 4.0 0.71 1.41 2.83; do
  M=~/LLM/models--randqwen-g$G
  OUT=results/ctrl/randqwen_g$G
  mkdir -p "$OUT"
  if [ ! -f "$M/config.json" ]; then
    echo "--- build gain=$G $(date)"
    $PY -m chaos_llm.run_random_tf --gain "$G" --out "$M" || { echo "FAILED build $G"; continue; }
  fi
  if [ ! -f "$OUT/a1i_txt/a1.npz" ]; then
    echo "--- a1i gain=$G $(date)"
    $PY -m chaos_llm.run_a1 --model "$M" --prompt @prompts.txt --window 4096 \
      --max-new-tokens 300 --num-directions 20 --batch-dirs 5 --inbatch-ref \
      --radii "$RADW" --layers "0,7,14,21,27" \
      --out "$OUT/a1i_txt" || echo "FAILED a1i $G"
  fi
  if [ -f "$OUT/a1i_txt/a1.npz" ]; then
    RADS=$($PY scripts/pick_radii.py "$OUT/a1i_txt/a1.npz")
    echo "--- gain=$G flip radii: $RADS"
    for R in $RADS; do
      RT=$(echo "$R" | tr -d '.')
      if [ ! -f "$OUT/fr_r${RT}/free_run.npz" ]; then
        echo "--- fr gain=$G r=$R $(date)"
        $PY -m chaos_llm.run_thresholds --model "$M" --run "$OUT/a1i_txt" \
          --radii "$R" --chunk 4 --out "$OUT/fr_r${RT}" || echo "FAILED fr $G $R"
      fi
    done
  fi
  if [ ! -f "$OUT/state/state2.npz" ]; then
    echo "--- depth probe gain=$G $(date)"
    $PY -m chaos_llm.run_state2 --model "$M" --run "$OUT/a1i_txt" \
      --steps 20,80 --dirs 8 --radii 0.05 --layers "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28" \
      --inject-layer 0 --batch-dirs 4 --out "$OUT/state" || echo "FAILED state2 $G"
  fi
done

# Pythia-410m step0: real architecture, random init at standard scale
if [ -f "$PYSTEP0/config.json" ]; then
  OUT=results/ctrl/pythia410_step0
  if [ ! -f "$OUT/a1i_txt/a1.npz" ]; then
    echo "--- a1i pythia step0 $(date)"
    $PY -m chaos_llm.run_a1 --model "$PYSTEP0" --prompt @prompts.txt --window 4096 \
      --max-new-tokens 300 --num-directions 20 --batch-dirs 10 --inbatch-ref \
      --radii "$RADW" --layers "0,6,12,18,23" \
      --out "$OUT/a1i_txt" || echo "FAILED a1i py step0"
  fi
  if [ -f "$OUT/a1i_txt/a1.npz" ]; then
    RADS=$($PY scripts/pick_radii.py "$OUT/a1i_txt/a1.npz")
    echo "--- pythia step0 flip radii: $RADS"
    for R in $RADS; do
      RT=$(echo "$R" | tr -d '.')
      if [ ! -f "$OUT/fr_r${RT}/free_run.npz" ]; then
        echo "--- fr pythia step0 r=$R $(date)"
        $PY -m chaos_llm.run_thresholds --model "$PYSTEP0" --run "$OUT/a1i_txt" \
          --radii "$R" --chunk 6 --out "$OUT/fr_r${RT}" || echo "FAILED fr py step0 $R"
      fi
    done
  fi
fi

echo "=== stage4 done $(date)"
