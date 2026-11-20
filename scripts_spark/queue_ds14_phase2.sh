#!/bin/bash
# Phase 2 campaign for DeepSeek-R1-Distill-Qwen-14B (fp32, DGX Spark).
# Window flush law replication + support scan + state kicks + depth spectrum
# + second window regime + multi-prompt A1s + long trajectories.
# Waits for phase 1 to complete. Resume-safe.
cd ~/LLM/chaos_llm_2026
export PYTHONPATH=src
export CHAOS_DTYPE=float32
PY=~/LLM/venv/bin/python
M=~/LLM/models/DeepSeek-R1-Distill-Qwen-14B
OUT=results_big/ds14
R=0.1
WFLAG="--layers 0,4,8,12,16,20,24,28,32,36,40,44,47"

# ---- wait for phase 1b (true-window core) ----
until grep -q PHASE1B_DONE ~/LLM/phase1b.log 2>/dev/null; do sleep 60; done
echo "=== phase2 start $(date) ==="

# window actuation sanity (phase1b already verified; cheap re-check)
if [ ! -f $OUT/window_verified ]; then
  echo "!!! window_verified marker missing - refusing to run the window program"; exit 3
fi

# 1. no-clipping reference: W=4096, 32 dirs, 500 tokens (also the fixed-token source)
if [ ! -f $OUT/a1_inter_w4096/a1.npz ]; then
  $PY -m chaos_llm.run_a1_big --model $M --prompt @prompts.txt \
      --window 4096 --max-new-tokens 500 --num-directions 32 \
      --radii $R --batch-dirs 8 $WFLAG \
      --out $OUT/a1_inter_w4096 2>&1 | tee -a $OUT/window.log
  if [ ! -f $OUT/a1_inter_w4096/a1.npz ]; then
    echo "!!! w4096 reference FAILED - stopping (inspect $OUT/window.log)"; exit 1
  fi
fi

# 2. window sweep on the SAME tokens (the causal control): departure at t* = W - P
for W in 50 100 150 200 300 500; do
  if [ ! -f $OUT/a1_inter_w${W}_fixed/a1.npz ]; then
    $PY -m chaos_llm.run_a1_big --model $M --prompt @prompts.txt \
        --window $W --max-new-tokens 500 --num-directions 240 \
        --radii $R --batch-dirs 8 $WFLAG \
        --fixed-tokens-from $OUT/a1_inter_w4096 \
        --out $OUT/a1_inter_w${W}_fixed 2>&1 | tee -a $OUT/window.log
  fi
done

# 3. support scan: directions confined to early/middle/late prompt positions.
#    The law predicts t*_1 = p_last + (W - P), so early support must add (P - p_last)
#    relative to late support. W=100, 240 dirs each.
for POS in 0:22 22:44 44:65; do
  TAG=early_$(echo $POS | tr ':' '-')
  if [ ! -f $OUT/a1_inter_w100_pos$TAG/a1.npz ]; then
    $PY -m chaos_llm.run_a1_big --model $M --prompt @prompts.txt \
        --window 100 --max-new-tokens 300 --num-directions 240 \
        --radii $R --batch-dirs 8 $WFLAG --positions $POS \
        --fixed-tokens-from $OUT/a1_inter_w4096 \
        --out $OUT/a1_inter_w100_pos$TAG 2>&1 | tee -a $OUT/window.log
  fi
done

# 3b. prompt-length scan: t*_1 = p_last + (W - P) has a P dependence; test it
#     directly with a short (P~6) and a long (P~250) prompt at several windows.
for PNAME in short_prompt long_prompt; do
  if [ ! -f $OUT/a1_${PNAME}_w4096/a1.npz ]; then
    $PY -m chaos_llm.run_a1_big --model $M --prompt @prompts_flush_extra.txt::$PNAME \
        --window 4096 --max-new-tokens 300 --num-directions 16 \
        --radii $R --batch-dirs 8 $WFLAG \
        --out $OUT/a1_${PNAME}_w4096 2>&1 | tee -a $OUT/window.log
  fi
done
for PNAME in short_prompt long_prompt; do
  for W in 50 100 200 300 500; do
    if [ ! -f $OUT/a1_${PNAME}_w${W}_fixed/a1.npz ]; then
      $PY -m chaos_llm.run_a1_big --model $M --prompt @prompts_flush_extra.txt::$PNAME \
          --window $W --max-new-tokens 300 --num-directions 120 \
          --radii $R --batch-dirs 8 $WFLAG \
          --fixed-tokens-from $OUT/a1_${PNAME}_w4096 \
          --out $OUT/a1_${PNAME}_w${W}_fixed 2>&1 | tee -a $OUT/window.log
    fi
  done
done

# 4. cascade: fp32 full-forwards at long T materialise [B,H,L,L] attention scores
#    (SDPA math path), so the horizon is capped at 1500 tokens, batch 2.
#    t*_l = t*_1 + (l-1)(W-1) covers layers up to ~14 at W=100 within 1500.
if [ ! -f $OUT/traj_w4096/interstellar_3000.npz ]; then
  printf 'interstellar_3000\t' > /tmp/p3k.txt
  head -1 prompts.txt | cut -f2 >> /tmp/p3k.txt
  $PY -m chaos_llm.collect_trajectories_big --model $M --prompts /tmp/p3k.txt \
      --window 4096 --max-new-tokens 1500 --layers 47 \
      --out $OUT/traj_w4096 2>&1 | tee -a $OUT/window.log
fi
if [ -f $OUT/traj_w4096/interstellar_3000.npz ] && [ ! -f $OUT/a1_inter_w100_cascade/a1.npz ]; then
  $PY -m chaos_llm.run_a1_big --model $M --prompt @prompts.txt \
      --window 100 --max-new-tokens 1500 --num-directions 8 \
      --radii $R --batch-dirs 2 --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 \
      --fixed-tokens-from $OUT/traj_w4096/interstellar_3000.npz \
      --out $OUT/a1_inter_w100_cascade 2>&1 | tee -a $OUT/window.log
fi

# 5. state kicks: single-position perturbations across the sequence
if [ ! -f $OUT/state_w100/state2.npz ]; then
  $PY -m chaos_llm.run_state2_big --model $M --run $OUT/a1_inter_w100 \
      --steps 8,16,32,50,75,100,150,200,250,299 --dirs 24 \
      --radii 0.03,0.1 --batch-dirs 8 \
      --out $OUT/state_w100 2>&1 | tee $OUT/state_w100.log
fi

# 6. depth tangent spectrum (8 tokens x 48 layers x k=8)
if [ ! -f $OUT/depth_w100/depth.npz ]; then
  $PY -m chaos_llm.run_depth_big --model $M --run $OUT/a1_inter_w100 \
      --tokens 0,21,50,100,150,200,250,299 --k 8 \
      --out $OUT/depth_w100 2>&1 | tee $OUT/depth_w100.log
fi

# 7. second window regime: W=500 A1 (margins + directions under a wide window)
if [ ! -f $OUT/a1_inter_w500/a1.npz ]; then
  $PY -m chaos_llm.run_a1_big --model $M --prompt @prompts.txt \
      --window 500 --max-new-tokens 300 --num-directions 240 \
      --radii $R --batch-dirs 16 --vector-rows 24 \
      --out $OUT/a1_inter_w500 2>&1 | tee -a $OUT/a1_inter_w500.log
fi

# 8. multi-prompt generality: 200 dirs on two more prompts
for P in prompts_childhood.txt prompts_cyber.txt; do
  TAG=$(echo $P | sed 's/prompts_//; s/.txt//')
  if [ ! -f $OUT/a1_${TAG}_w100/a1.npz ]; then
    $PY -m chaos_llm.run_a1_big --model $M --prompt @$P \
        --window 100 --max-new-tokens 300 --num-directions 200 \
        --radii $R --batch-dirs 16 --vector-rows 16 \
        --out $OUT/a1_${TAG}_w100 2>&1 | tee -a $OUT/multiprompt.log
  fi
done

# 9. trajectories for dimensionality (12 prompts x 1000 tokens, greedy, W=4096)
if [ ! -f $OUT/traj_w4096/.done ]; then
  $PY -m chaos_llm.collect_trajectories_big --model $M \
      --prompts prompts_d2.txt --window 4096 --max-new-tokens 1000 \
      --layers 12,24,47 --out $OUT/traj_w4096 2>&1 | tee -a $OUT/traj.log
  touch $OUT/traj_w4096/.done
fi

echo PHASE2_DONE
