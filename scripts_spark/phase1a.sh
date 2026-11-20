#!/bin/bash
# Auto-launch: wait for the inference stack, then first GPU work on the 14B:
# smoke test (40 tokens) then calibration (margins, r-linearity, r_* preview).
# The big 1000-direction A1 is NOT started here; that radius choice is made
# after a human/agent reads the calibration output.
cd ~/LLM/chaos_llm_2026
export PYTHONPATH=src
export CHAOS_DTYPE=float32
PY=~/LLM/venv/bin/python
M=~/LLM/models/DeepSeek-R1-Distill-Qwen-14B
OUT=results_big/ds14
mkdir -p $OUT

until grep -q STACK_READY ~/LLM/install_stack.log 2>/dev/null; do sleep 20; done
echo "=== stack ready, starting smoke $(date) ==="

$PY scripts_spark/smoke_big.py --model $M --tokens 40 2>&1 | tee $OUT/smoke.log
if grep -q "SMOKE OK" $OUT/smoke.log; then
  touch $OUT/smoke.done
  echo "=== smoke ok, starting calibration $(date) ==="
  $PY -m chaos_llm.run_calib_big --model $M --prompt @prompts.txt \
      --window 100 --max-new-tokens 400 --num-directions 8 \
      --radii 1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1.0 \
      --out $OUT/calib 2>&1 | tee $OUT/calib.log
  echo "=== calibration done $(date) ==="
else
  echo "!!! SMOKE FAILED — inspect $OUT/smoke.log"
fi
