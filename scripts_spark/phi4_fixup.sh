#!/bin/bash
# After phase 5 (Qwen3-14B) finishes: re-run phi-4's non-normality (fused-MLP fix),
# then phi-4 trajectory distances on the GPU.
cd ~/LLM/chaos_llm_2026
until grep -q PHASE5_DONE ~/LLM/queue_p5.log 2>/dev/null; do sleep 120; done
echo "=== phi4 nonnorm re-run $(date) ==="
export PYTHONPATH=src
export CHAOS_DTYPE=float32
PY=~/LLM/venv/bin/python
M=~/LLM/models/phi-4
OUT=results_big/phi4
if [ ! -f $OUT/nonnorm_w0/nonnorm.npz ]; then
  $PY -m chaos_llm.run_nonnorm_big --model $M --run $OUT/a1_inter_w0 \
      --tokens 219,277,53,100,200 \
      --out $OUT/nonnorm_w0 2>&1 | tee $OUT/nonnorm_w0.log
fi
echo "=== phi4 d2 distances $(date) ==="
$PY scripts_spark/d2_distances.py --run-dir $OUT/traj --states h_last --out $OUT/d2 2>&1 | tee ~/LLM/d2_dist_phi4.log
echo "=== qwen314 d2 distances $(date) ==="
$PY scripts_spark/d2_distances.py --run-dir results_big/qwen314/traj --states h_last --out results_big/qwen314/d2 2>&1 | tee ~/LLM/d2_dist_q314.log
echo PHI4_FIXUP_DONE
