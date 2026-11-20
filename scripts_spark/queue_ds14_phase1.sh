#!/bin/bash
# Phase 1 campaign for DeepSeek-R1-Distill-Qwen-14B on the DGX Spark (fp32).
# smoke -> calibration -> core A1 ensemble -> free-run validation.
# Every step is resume-safe (skips existing outputs).
cd ~/LLM/chaos_llm_2026
export PYTHONPATH=src
export CHAOS_DTYPE=float32
PY=~/LLM/venv/bin/python
M=~/LLM/models/DeepSeek-R1-Distill-Qwen-14B
OUT=results_big/ds14
mkdir -p $OUT

# 0. smoke test (first contact with the 14B in fp32)
if [ ! -f $OUT/smoke.done ]; then
  $PY scripts_spark/smoke_big.py --model $M --tokens 40 2>&1 | tee $OUT/smoke.log
  grep -q "SMOKE OK" $OUT/smoke.log && touch $OUT/smoke.done
fi

# 1. calibration: bit-identity, margin spectrum, r-linearity, r_* preview
if [ ! -f $OUT/calib/meta.json ]; then
  $PY -m chaos_llm.run_calib_big --model $M --prompt @prompts.txt \
      --window 100 --max-new-tokens 400 --num-directions 8 \
      --radii 1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1.0 \
      --out $OUT/calib 2>&1 | tee $OUT/calib.log
fi

# 2. core A1 ensemble: 1000 directions at the calibrated radius, 300 tokens.
#    (Radii placeholder 0.1: REPLACE after reading the calibration output.)
R=${A1_RADIUS:-0.1}
if [ ! -f $OUT/a1_inter_w100/a1.npz ]; then
  $PY -m chaos_llm.run_a1_big --model $M --prompt @prompts.txt \
      --window 100 --max-new-tokens 300 --num-directions 1000 \
      --radii $R --batch-dirs 16 --vector-rows 24 \
      --out $OUT/a1_inter_w100 2>&1 | tee -a $OUT/a1_inter_w100.log
fi

# 3. free-run validation at three radii (all 1000 directions are cheap in batch)
if [ ! -f $OUT/free_inter_w100/summary.json ]; then
  $PY -m chaos_llm.run_free_big --model $M --run $OUT/a1_inter_w100 \
      --radii 0.05,0.1,0.3 --chunk 24 --out $OUT/free_inter_w100 2>&1 | tee $OUT/free_inter_w100.log
fi

echo PHASE1_DONE
