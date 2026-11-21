#!/bin/bash
# Consolidated control queue (v2). All items behind an flock so this queue
# never overlaps itself, and memory-gated so it interleaves with other agents.
# Resume-safe.
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv_tf/bin/python
exec 9>/tmp/ctrl_queue.lock
flock 9

TOTAL=15360
wait_free() { # $1 = MiB needed
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$((TOTAL - used))" -ge "$1" ] && break
    sleep 90
  done
}

M130=~/LLM/models--state-spaces--mamba-130m-hf
M14=~/LLM/models--state-spaces--mamba-1.4b-hf
PY410=~/LLM/models--EleutherAI--pythia-410m
PYSTEP0=~/LLM/models--EleutherAI--pythia-410m-step0
RADW="3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1"

echo "=== ctrl-all2 start $(date)"

# ---------- small items first (fit alongside other agents) ----------
if [ ! -f results/ctrl/pythia410_step0/a1i_txt/a1.npz ]; then
  wait_free 3500
  echo "--- a1i pythia step0 $(date)"
  $PY -m chaos_llm.run_a1 --model "$PYSTEP0" --prompt @prompts.txt --window 4096 \
    --max-new-tokens 300 --num-directions 20 --batch-dirs 10 --inbatch-ref \
    --radii "$RADW" --layers "0,6,12,18,23" \
    --out results/ctrl/pythia410_step0/a1i_txt || echo "FAILED a1i py step0"
fi
if [ -f results/ctrl/pythia410_step0/a1i_txt/a1.npz ]; then
  RADS=$($PY scripts/pick_radii.py results/ctrl/pythia410_step0/a1i_txt/a1.npz)
  echo "--- step0 flip radii: $RADS"
  for R in $RADS; do
    RT=$(echo "$R" | tr -d '.')
    if [ ! -f results/ctrl/pythia410_step0/fr_r${RT}/free_run.npz ]; then
      wait_free 3500
      echo "--- fr step0 r=$R $(date)"
      $PY -m chaos_llm.run_thresholds --model "$PYSTEP0" --run results/ctrl/pythia410_step0/a1i_txt \
        --radii "$R" --chunk 6 --out results/ctrl/pythia410_step0/fr_r${RT} || echo "FAILED fr py step0 $R"
    fi
  done
fi

# pythia-410 wide radius A1 + r*-scaled free runs
if [ ! -f results/ctrl/pythia410/a1i_txt_wide/a1.npz ]; then
  wait_free 3500
  echo "--- a1i pythia wide $(date)"
  $PY -m chaos_llm.run_a1 --model "$PY410" --prompt @prompts.txt --window 4096 \
    --max-new-tokens 300 --num-directions 20 --batch-dirs 10 --inbatch-ref \
    --radii "0.05,0.1,0.19,0.3,0.5,0.7,1,1.4,2,3" --layers "0,6,12,18,23" \
    --out results/ctrl/pythia410/a1i_txt_wide || echo "FAILED a1i py wide"
fi
for R in 0.19 0.3 0.7 1.4 3; do
  RT=$(echo "$R" | tr -d '.')
  if [ ! -f results/ctrl/pythia410/frw_r${RT}/free_run.npz ]; then
    wait_free 3500
    echo "--- fr pythia wide r=$R $(date)"
    $PY -m chaos_llm.run_thresholds --model "$PY410" --run results/ctrl/pythia410/a1i_txt_wide \
      --radii "$R" --chunk 6 --out results/ctrl/pythia410/frw_r${RT} || echo "FAILED fr py wide $R"
  fi
done

# mamba tfp leftovers
for r in 2 3 5; do
  if [ ! -f results/ctrl/mamba130/tfp_r${r}/tf_predict.npz ]; then
    wait_free 3000
    echo "--- tfp mamba r=$r $(date)"
    $PY -m chaos_llm.run_tf_predict --model "$M130" --run results/ctrl/mamba130/a1i_txt \
      --free-run results/ctrl/mamba130/fr_txt_r${r} --radii "$r" --chunk 6 \
      --out results/ctrl/mamba130/tfp_r${r} || echo "FAILED tfp $r"
  fi
done

# mamba-1.4b A1 (scale check)
if [ ! -f results/ctrl/mamba14/a1i_txt/a1.npz ]; then
  wait_free 7500
  echo "--- a1i mamba14 $(date)"
  $PY -m chaos_llm.run_a1 --model "$M14" --prompt @prompts.txt --window 4096 \
    --max-new-tokens 300 --num-directions 20 --batch-dirs 5 --inbatch-ref \
    --radii "1e-3,1e-2,0.1,0.3,1,1.4,3,5" --layers "0,12,24,35,47" \
    --out results/ctrl/mamba14/a1i_txt || echo "FAILED a1i14"
fi

# ---------- random Qwen2 gain sweep (large: gated hard) ----------
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

echo "=== ctrl-all2 done $(date)"
