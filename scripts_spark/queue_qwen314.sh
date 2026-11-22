#!/bin/bash
# Phase 5: Qwen3-14B (Qwen3 arch, no sliding window). Same program as phase 4.
cd ~/LLM/chaos_llm_2026
export PYTHONPATH=src
export CHAOS_DTYPE=float32
PY=~/LLM/venv/bin/python
M=~/LLM/models/Qwen3-14B
OUT=results_big/qwen314
mkdir -p $OUT
LAY="--layers 0,1,2,3,5,8,12,16,20,24,28,32,36,39"

until grep -q PHASE4_DONE ~/LLM/queue_p4.log 2>/dev/null; do sleep 120; done
echo "=== phase5 (Qwen3-14B) start $(date) ==="

if [ ! -f $OUT/smoke.done ]; then
  $PY scripts_spark/smoke_big.py --model $M --tokens 40 2>&1 | tee $OUT/smoke.log
  grep -q "SMOKE OK" $OUT/smoke.log && touch $OUT/smoke.done
  if [ ! -f $OUT/smoke.done ]; then
    echo "!!! Qwen3-14B smoke FAILED - phase5 stopped"; exit 3
  fi
fi

if [ ! -f $OUT/calib/meta.json ]; then
  $PY -m chaos_llm.run_calib_big --model $M --window 0 \
      --prompt @prompts.txt --max-new-tokens 400 --num-directions 8 \
      --radii 1e-3,3e-3,1e-2,3e-2,0.1,0.3,1.0 \
      --out $OUT/calib 2>&1 | tee $OUT/calib.log
fi

if [ ! -f $OUT/a1_inter_w0/a1.npz ]; then
  $PY -m chaos_llm.run_a1_big --model $M --window 0 \
      --prompt @prompts.txt --max-new-tokens 300 --num-directions 1000 \
      --radii ${A1_RADIUS:-0.1} --batch-dirs 16 --vector-rows 24 \
      $LAY --out $OUT/a1_inter_w0 2>&1 | tee -a $OUT/a1_inter_w0.log
  if [ ! -f $OUT/a1_inter_w0/a1.npz ]; then echo "!!! Qwen3-14B A1 FAILED"; exit 1; fi
fi

if [ ! -f $OUT/free_inter_w0/summary.json ]; then
  $PY -m chaos_llm.run_free_big --model $M --run $OUT/a1_inter_w0 \
      --radii 0.1,0.3 --chunk 24 --out $OUT/free_inter_w0 2>&1 | tee $OUT/free_inter_w0.log
fi

if [ ! -f $OUT/state_w0/state2.npz ]; then
  $PY -m chaos_llm.run_state2_big --model $M --run $OUT/a1_inter_w0 \
      --steps 8,16,32,50,75,100,150,200,250,299 --dirs 24 \
      --radii 0.03,0.1 --batch-dirs 8 --window 0 \
      --out $OUT/state_w0 2>&1 | tee $OUT/state_w0.log
fi

if [ ! -f $OUT/depth_w0/depth.npz ]; then
  $PY -m chaos_llm.run_depth_big --model $M --run $OUT/a1_inter_w0 \
      --tokens 0,21,50,100,150,200,250,299 --k 8 \
      --out $OUT/depth_w0 2>&1 | tee $OUT/depth_w0.log
fi

if [ ! -f $OUT/nonnorm_w0/nonnorm.npz ]; then
  $PY -m chaos_llm.run_nonnorm_big --model $M --run $OUT/a1_inter_w0 \
      --tokens 21,50,100,189,273 \
      --out $OUT/nonnorm_w0 2>&1 | tee $OUT/nonnorm_w0.log
fi

for P in prompts_childhood.txt prompts_cyber.txt; do
  TAG=$(echo $P | sed 's/prompts_//; s/.txt//')
  if [ ! -f $OUT/a1_${TAG}_w0/a1.npz ]; then
    $PY -m chaos_llm.run_a1_big --model $M --window 0 \
        --prompt @$P --max-new-tokens 300 --num-directions 200 \
        --radii ${A1_RADIUS:-0.1} --batch-dirs 16 --vector-rows 16 \
        $LAY --out $OUT/a1_${TAG}_w0 2>&1 | tee -a $OUT/multiprompt.log
  fi
done

if [ ! -f $OUT/traj/.done ]; then
  $PY -m chaos_llm.collect_trajectories_big --model $M \
      --prompts prompts_d2.txt --window 0 --max-new-tokens 1000 \
      --layers 10,20,39 --out $OUT/traj 2>&1 | tee -a $OUT/traj.log
  touch $OUT/traj/.done
fi

echo PHASE5_DONE
