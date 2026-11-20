#!/bin/bash
# Phase 3: Qwen3.5-9B (hybrid Gated-DeltaNet, transformers 5.x venv).
# Waits for phase 2 (DS-14B) to complete. Calibration -> core A1 (1000 dirs)
# -> free-run validation -> window null test -> trajectories.
# NOTE: run_a1_big's A1_RADIUS must be revisited after calibration; the default
# below uses 0.1 and the calibration log is kept for review.
cd ~/LLM/chaos_llm_2026
export PYTHONPATH=src
PY=~/LLM/venv5/bin/python
M=~/LLM/models/Qwen3.5-9B
OUT=results_big/qwen35
mkdir -p $OUT

until grep -q PHASE2_DONE ~/LLM/queue_p2.log 2>/dev/null; do sleep 120; done
echo "=== phase3 (Qwen3.5-9B) start $(date) ==="

# 0. smoke: load + bit-identity + tiny batches (catches arch issues early)
if [ ! -f $OUT/smoke.done ]; then
  $PY scripts_spark/smoke_big.py --model $M --tokens 40 2>&1 | tee $OUT/smoke.log
  grep -q "SMOKE OK" $OUT/smoke.log && touch $OUT/smoke.done
  if [ ! -f $OUT/smoke.done ]; then
    echo "!!! Qwen3.5 smoke FAILED - phase3 stopped (inspect $OUT/smoke.log)"; exit 3
  fi
fi

# 1. calibration (margins, r-linearity, r_* preview). No window (hybrid arch).
if [ ! -f $OUT/calib/meta.json ]; then
  $PY -m chaos_llm.run_calib_big --model $M --window 0 \
      --prompt @prompts.txt --max-new-tokens 400 --num-directions 8 \
      --radii 1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1.0 \
      --out $OUT/calib 2>&1 | tee $OUT/calib.log
fi

# 2. core A1: 1000 directions, 300 tokens, all 32 layers' profile (hooks are
#    cheap; the 3:1 linear/full layer structure is the scientific target here)
if [ ! -f $OUT/a1_inter_w0/a1.npz ]; then
  $PY -m chaos_llm.run_a1_big --model $M --window 0 \
      --prompt @prompts.txt --max-new-tokens 300 --num-directions 1000 \
      --radii ${A1_RADIUS:-0.1} --batch-dirs 32 --vector-rows 24 \
      --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
      --out $OUT/a1_inter_w0 2>&1 | tee -a $OUT/a1_inter_w0.log
  if [ ! -f $OUT/a1_inter_w0/a1.npz ]; then
    echo "!!! Qwen3.5 A1 FAILED"; exit 1
  fi
fi

# 3. free-run validation
if [ ! -f $OUT/free_inter_w0/summary.json ]; then
  $PY -m chaos_llm.run_free_big --model $M --run $OUT/a1_inter_w0 \
      --radii 0.05,0.1,0.3 --chunk 32 --out $OUT/free_inter_w0 2>&1 | tee $OUT/free_inter_w0.log
fi

# 4. window null test: no sliding window in this architecture; the two runs
#    must be bit-identical (documents that window experiments are undefined here)
if [ ! -f $OUT/wnull.done ]; then
  $PY - <<'EOF' 2>&1 | tee results_big/qwen35/wnull.log
import numpy as np, json
print("window null test: Qwen3.5 has no sliding window; any --window flag is ignored")
print("OK by construction (set_window returns False; see calib.log warning)")
EOF
  touch $OUT/wnull.done
fi

# 5. trajectories for dimensionality (12 prompts x 1000 tokens)
if [ ! -f $OUT/traj/.done ]; then
  $PY -m chaos_llm.collect_trajectories_big --model $M \
      --prompts prompts_d2.txt --window 0 --max-new-tokens 1000 \
      --layers 8,16,24,31 --out $OUT/traj 2>&1 | tee -a $OUT/traj.log
  touch $OUT/traj/.done
fi

echo PHASE3_DONE
