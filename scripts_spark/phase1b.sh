#!/bin/bash
# Phase 1b: after the (full-attention) phase-1 free validation finishes:
#   1. verify the sliding window really applies (fail hard if not),
#   2. relabel the existing w100 runs as full-attention (that is what they are),
#   3. re-run the core at the TRUE window 100: calibration, A1, free validation.
# The original phase-1 data is kept as the no-clipping reference.
cd ~/LLM/chaos_llm_2026
export PYTHONPATH=src
export CHAOS_DTYPE=float32
PY=~/LLM/venv/bin/python
M=~/LLM/models/DeepSeek-R1-Distill-Qwen-14B
OUT=results_big/ds14
R=0.1

until grep -q PHASE1_DONE ~/LLM/queue_p1.log 2>/dev/null; do sleep 60; done
echo "=== phase1b start $(date) ==="

# 1. window actuation test (the whole phase-2 program depends on it)
if [ ! -f $OUT/window_verified ]; then
  $PY scripts_spark/verify_window.py --model $M --window 100 \
      --fixed-tokens-from $OUT/a1_inter_w100/a1.npz 2>&1 | tee $OUT/verify_window.log
  grep -q WINDOW_OK $OUT/verify_window.log && touch $OUT/window_verified
  if [ ! -f $OUT/window_verified ]; then
    echo "!!! WINDOW BROKEN - phase1b/phase2 stopped. Inspect $OUT/verify_window.log"
    exit 3
  fi
fi

# 2. relabel the full-attention runs (a1 and the free validation that just ended)
if [ -d $OUT/a1_inter_w100 ] && [ ! -d $OUT/a1_inter_full ]; then
  mv $OUT/a1_inter_w100 $OUT/a1_inter_full
  $PY - <<'EOF'
import json
p = "results_big/ds14/a1_inter_full/meta.json"
m = json.load(open(p)); m["window"] = 4096
m["window_note"] = "run executed before the transformers>=4.5x layer_types fix; sliding window was NOT applied; equivalent to full attention (W=4096 semantics)"
json.dump(m, open(p, "w"), indent=2)
EOF
fi
if [ -d $OUT/free_inter_w100 ] && [ ! -d $OUT/free_inter_full ]; then
  mv $OUT/free_inter_w100 $OUT/free_inter_full
  $PY - <<'EOF'
import json
p = "results_big/ds14/free_inter_full/summary.json"
try:
    m = json.load(open(p)); m["window"] = 4096
    m["window_note"] = "executed before the layer_types fix; full attention (W=4096 semantics)"
    json.dump(m, open(p, "w"), indent=2)
except FileNotFoundError:
    pass
EOF
fi

# 3. true-window calibration (margins change under the window!)
if [ ! -f $OUT/calib_w100/meta.json ]; then
  $PY -m chaos_llm.run_calib_big --model $M --prompt @prompts.txt \
      --window 100 --max-new-tokens 400 --num-directions 8 \
      --radii 0.01,0.03,0.1,0.3,1.0 \
      --out $OUT/calib_w100 2>&1 | tee $OUT/calib_w100.log
fi

# 4. core A1 at the true window
if [ ! -f $OUT/a1_inter_w100/a1.npz ]; then
  $PY -m chaos_llm.run_a1_big --model $M --prompt @prompts.txt \
      --window 100 --max-new-tokens 300 --num-directions 1000 \
      --radii $R --batch-dirs 16 --vector-rows 24 \
      --out $OUT/a1_inter_w100 2>&1 | tee -a $OUT/a1_inter_w100.log
  if [ ! -f $OUT/a1_inter_w100/a1.npz ]; then
    echo "!!! true-w100 A1 FAILED - stopping"; exit 1
  fi
fi

# 5. free-run validation at the true window
if [ ! -f $OUT/free_inter_w100/summary.json ]; then
  $PY -m chaos_llm.run_free_big --model $M --run $OUT/a1_inter_w100 \
      --radii 0.05,0.1,0.3 --chunk 24 --out $OUT/free_inter_w100 2>&1 | tee $OUT/free_inter_w100.log
fi

echo PHASE1B_DONE
