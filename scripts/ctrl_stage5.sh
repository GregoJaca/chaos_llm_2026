#!/bin/bash
# Control study, stage 5: master.md-comparable experiments.
# KV-cache kicks (DeepSeek), Mamba transport kernel + superposition,
# tie-corrected flip fractions (8 prompts x 192 dirs), r* with 512 directions,
# state kicks at s=16/50, and trajectories for the random-init models (RP/D2).
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTF=~/LLM/venv_tf/bin/python
PYSYS=python3
M130=~/LLM/models--state-spaces--mamba-130m-hf
PYSTEP0=~/LLM/models--EleutherAI--pythia-410m-step0
exec 9>/tmp/ctrl_queue.lock
flock 9

wait_free() {
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$((15360 - used))" -ge "$1" ] && break
    sleep 90
  done
}

echo "=== ctrl-stage5 start $(date)"

# 1) DeepSeek KV-cache kicks (fp32, 7.1GB) - run alone
if [ ! -f results/ctrl/ds_kv/kv_kick.npz ]; then
  wait_free 10500
  echo "--- kv kick deepseek $(date)"
  $PYSYS -m chaos_llm.run_kv_kick \
    --kick-steps 70,150 --eps-list 0.01,0.1,1.0 --layer-modes all,0,14,27 \
    --dirs 8 --horizon 150 \
    --out results/ctrl/ds_kv || echo "FAILED kv kick"
fi

# 2) Mamba transport kernel + superposition
if [ ! -f results/ctrl/mamba130/kernel/kernel.npy ]; then
  wait_free 2500
  echo "--- mamba kernel+superposition $(date)"
  $PYTF -m chaos_llm.run_kernel_mamba \
    --model "$M130" --prompt @prompts.txt --dirs 8 --r 0.1 \
    --sup-radii 0.01,0.03,0.1 --sup-trials 3 \
    --out results/ctrl/mamba130/kernel || echo "FAILED kernel"
fi

# 3) Mamba flip fractions: 8 prompts x 192 dirs x 4 radii
cat prompts.txt prompts_childhood.txt prompts_cyber.txt prompts_more.txt > /tmp/flipfrac_prompts.txt 2>/dev/null
head -c 200000 /tmp/flipfrac_prompts.txt > /tmp/ff8.txt
if [ ! -f results/ctrl/mamba130/flipfrac/meta.json ]; then
  wait_free 2500
  echo "--- mamba flip fractions $(date)"
  $PYTF -m chaos_llm.run_flipfrac --model "$M130" \
    --prompts /tmp/ff8.txt --radii 0.1,0.3,0.6,1.0 --dirs 192 --chunk 24 \
    --out results/ctrl/mamba130/flipfrac || echo "FAILED flipfrac"
fi

# 4) r* with 512 directions, 5 prompts
for PAIR in "prompts.txt:txt" "prompts_childhood.txt:childhood" "prompts_cyber.txt:cyber" "prompts_more.txt:hoc" "prompts_more.txt:astro"; do
  P="${PAIR%%:*}"; TAG="${PAIR##*:}"
  IDX=1
  [ "$TAG" = "astro" ] && IDX=5
  [ "$TAG" = "hoc" ] && IDX=1
  if [ ! -f results/ctrl/mamba130/r512_$TAG/a1.npz ]; then
    wait_free 2500
    echo "--- r512 $TAG $(date)"
    if [ "$TAG" = "hoc" ] || [ "$TAG" = "astro" ]; then
      sed -n "${IDX}p" "$P" > /tmp/r512_$TAG.txt
      $PYTF -m chaos_llm.run_a1 --model "$M130" --prompt @/tmp/r512_$TAG.txt --window 4096 \
        --max-new-tokens 300 --num-directions 512 --batch-dirs 32 --inbatch-ref \
        --radii 0.1 --layers "23" --out results/ctrl/mamba130/r512_$TAG || echo "FAILED r512 $TAG"
    else
      $PYTF -m chaos_llm.run_a1 --model "$M130" --prompt @"$P" --window 4096 \
        --max-new-tokens 300 --num-directions 512 --batch-dirs 32 --inbatch-ref \
        --radii 0.1 --layers "23" --out results/ctrl/mamba130/r512_$TAG || echo "FAILED r512 $TAG"
    fi
  fi
done

# 5) Mamba state kicks at s=16,50 (master section-6/8 convention)
if [ ! -f results/ctrl/mamba130/state_s16_50/state2.npz ]; then
  wait_free 2500
  echo "--- mamba state2 s=16,50 $(date)"
  $PYTF -m chaos_llm.run_state2 --model "$M130" --run results/ctrl/mamba130/a1i_txt \
    --steps 16,50 --dirs 8 --radii 0.1 --layers "0,6,12,18,23" \
    --inject-layer 0 --batch-dirs 4 --out results/ctrl/mamba130/state_s16_50 || echo "FAILED state2 mamba"
fi

# 6) trajectories for the random-init models (RP/D2 comparison)
sed -n "1,3p" prompts_long2.txt > /tmp/p3long.txt
for G in 1.0 0.35 4.0; do
  M=~/LLM/models--randqwen-g$G
  OUT=results/ctrl/randqwen_g$G/traj
  if [ ! -f "$OUT/ocean_currents.npz" ]; then
    wait_free 9500
    echo "--- traj randqwen g=$G $(date)"
    $PYTF -m chaos_llm.collect_trajectories --model "$M" --prompts /tmp/p3long.txt \
      --window 4096 --max-new-tokens 1000 --layers "7,14,21,27" \
      --out results/ctrl/randqwen_g$G/traj_tmp && mkdir -p "$OUT" \
      && mv results/ctrl/randqwen_g$G/traj_tmp/*.npz "$OUT/" || echo "FAILED traj g$G"
  fi
done
if [ ! -f results/ctrl/pythia410_step0/traj/ocean_currents.npz ]; then
  wait_free 3500
  echo "--- traj pythia step0 $(date)"
  $PYTF -m chaos_llm.collect_trajectories --model "$PYSTEP0" --prompts /tmp/p3long.txt \
    --window 4096 --max-new-tokens 1000 --layers "6,12,18,23" \
    --out results/ctrl/pythia410_step0/traj_tmp && mkdir -p results/ctrl/pythia410_step0/traj \
    && mv results/ctrl/pythia410_step0/traj_tmp/*.npz results/ctrl/pythia410_step0/traj/ || echo "FAILED traj step0"
fi

echo "=== ctrl-stage5 done $(date)"
