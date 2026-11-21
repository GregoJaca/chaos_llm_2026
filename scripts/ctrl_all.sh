#!/bin/bash
# Control study, single orchestrator (replaces the stage3/stage4 chains).
# Every item is memory-gated: it starts only when the requested amount of GPU
# is free, so this queue interleaves with other agents' jobs instead of
# colliding. All items are resume-safe ([ -f out ] checks).
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv_tf/bin/python
TOTAL=15360

wait_free() { # $1 = MiB needed
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$((TOTAL - used))" -ge "$1" ]; then break; fi
    sleep 90
  done
}

M130=~/LLM/models--state-spaces--mamba-130m-hf
M370=~/LLM/models--state-spaces--mamba-370m-hf
PY1B=~/LLM/models--EleutherAI--pythia-1b
PYSTEP0=~/LLM/models--EleutherAI--pythia-410m-step0
RADW="3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1"

echo "=== ctrl-all start $(date)"

# ---------- Jacobians (small footprint, can share the GPU) ----------
if [ ! -f results/ctrl/mamba130/jac/jac.npz ]; then
  wait_free 3000
  echo "--- jac mamba130 $(date)"
  $PY -m chaos_llm.run_mamba_jacobian --model "$M130" \
    --tokens-from results/ctrl/mamba130/a1i_txt/a1.npz \
    --t0-list 80,150,220 --horizon 200 \
    --out results/ctrl/mamba130/jac || echo "FAILED jac130"
fi
if [ ! -f results/ctrl/mamba370/jac/jac.npz ]; then
  wait_free 4000
  echo "--- jac mamba370 $(date)"
  $PY -m chaos_llm.run_mamba_jacobian --model "$M370" \
    --tokens-from results/ctrl/mamba370/a1i_txt/a1.npz \
    --t0-list 80,150,220 --horizon 200 \
    --out results/ctrl/mamba370/jac || echo "FAILED jac370"
fi

# ---------- leftovers from stage2 ----------
if [ ! -f results/ctrl/pythia1b/a1i_txt/a1.npz ]; then
  wait_free 8500
  echo "--- a1i pythia1b $(date)"
  $PY -m chaos_llm.run_a1 --model "$PY1B" --prompt @prompts.txt --window 4096 \
    --max-new-tokens 300 --num-directions 20 --batch-dirs 10 --inbatch-ref \
    --radii "$RADW" --layers "0,4,8,12,15" \
    --out results/ctrl/pythia1b/a1i_txt || echo "FAILED a1i py1b"
fi

# ---------- random Qwen2 gain sweep (large footprint: gated hard) ----------
for G in 1.0 0.5 2.0 0.35 4.0 0.71 1.41 2.83; do
  M=~/LLM/models--randqwen-g$G
  OUT=results/ctrl/randqwen_g$G
  mkdir -p "$OUT"
  if [ ! -f "$M/config.json" ]; then
    echo "--- build gain=$G $(date) (cpu)"
    $PY -m chaos_llm.run_random_tf --gain "$G" --out "$M" || { echo "FAILED build $G"; continue; }
  fi
  if [ ! -f "$OUT/a1i_txt/a1.npz" ]; then
    wait_free 11500
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
        wait_free 10000
        echo "--- fr gain=$G r=$R $(date)"
        $PY -m chaos_llm.run_thresholds --model "$M" --run "$OUT/a1i_txt" \
          --radii "$R" --chunk 4 --out "$OUT/fr_r${RT}" || echo "FAILED fr $G $R"
      fi
    done
  fi
  if [ ! -f "$OUT/state/state2.npz" ]; then
    wait_free 10000
    echo "--- depth probe gain=$G $(date)"
    $PY -m chaos_llm.run_state2 --model "$M" --run "$OUT/a1i_txt" \
      --steps 20,80 --dirs 8 --radii 0.05 --layers "0,2,4,6,8,10,12,14,16,18,20,22,24,26,28" \
      --inject-layer 0 --batch-dirs 4 --out "$OUT/state" || echo "FAILED state2 $G"
  fi
done

# ---------- Pythia-410m step0 ----------
if [ -f "$PYSTEP0/config.json" ]; then
  OUT=results/ctrl/pythia410_step0
  if [ ! -f "$OUT/a1i_txt/a1.npz" ]; then
    wait_free 4000
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
        wait_free 4000
        echo "--- fr pythia step0 r=$R $(date)"
        $PY -m chaos_llm.run_thresholds --model "$PYSTEP0" --run "$OUT/a1i_txt" \
          --radii "$R" --chunk 6 --out "$OUT/fr_r${RT}" || echo "FAILED fr py step0 $R"
      fi
    done
  fi
fi

echo "=== ctrl-all done $(date)"
