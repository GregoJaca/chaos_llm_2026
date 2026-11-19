#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "QUEUE25 START $(date)"

# 1. All directions: k = hidden size (1536), one radius, teacher-forced
if [ ! -f results/big4/all_dirs/a1.npz ]; then
  python3 -m chaos_llm.run_a1 --model $M --prompt @prompts.txt --window 100 \
    --max-new-tokens 300 --num-directions 1536 --radii 0.1 \
    --layers 0,7,14,21,27 --batch-dirs 2 \
    --out results/big4/all_dirs || echo "FAILED all_dirs"
fi
echo "all_dirs done $(date)"

# 2. Two more stable-chaos prompts (10 total)
for i in 7 8; do
  sed -n "${i}p" prompts_more2.txt > /tmp/SC_$i.txt
  [ -s /tmp/SC_$i.txt ] || continue
  if [ ! -f results/big4/chaos_geom/p${i}/a1.npz ]; then
    python3 -m chaos_llm.run_a1 --model $M --prompt @/tmp/SC_$i.txt --window 100 \
      --max-new-tokens 300 --num-directions 20 --radii 0.01,0.1 --layers 7,14,21,27 \
      --batch-dirs 1 --out results/big4/chaos_geom/p${i} || echo "FAILED geom $i"
  fi
  for r in 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.7 1.0 1.5 2.0; do
    out=results/big4/chaos/p${i}_r${r}
    [ -f "$out/free_run.npz" ] && continue
    python3 -m chaos_llm.run_thresholds --run results/big4/chaos_geom/p${i} \
      --radii $r --chunk 6 --out $out || echo "FAILED chaos $i $r"
  done
done

# 3. Four more 4000-token trajectories (different prompts)
for i in 5 6; do
  sed -n "${i}p" prompts_long2.txt > /tmp/T4b_$i.txt
  out=results/big4/traj4k/T4b_$i.npz
  [ -f "$out" ] && continue
  python3 -m chaos_llm.collect_trajectories --prompts /tmp/T4b_$i.txt --window 4096 \
    --max-new-tokens 4000 --layers 7,14,21,27 --out results/big4/t4b_tmp_$i || echo "FAILED traj T4b_$i"
  mkdir -p results/big4/traj4k
  mv results/big4/t4b_tmp_$i/*.npz "$out" 2>/dev/null
done

echo "QUEUE25 DONE $(date)"
