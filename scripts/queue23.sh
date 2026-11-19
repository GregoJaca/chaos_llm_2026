#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "QUEUE23 START $(date)"

# 1. 4000-token teacher-forced response on the four prompt lengths, two windows
for i in 1 2 3 4; do
  sed -n "${i}p" prompts_lengths.txt > /tmp/LEN_$i.txt
  src=results/big4/len_traj/p$i.npz
  [ -f "$src" ] || continue
  for w in 100 4096; do
    out=results/big4/len_a1k/p${i}_w${w}
    [ -f "$out/a1.npz" ] && continue
    python3 -m chaos_llm.run_a1_long --prompt @/tmp/LEN_$i.txt --window $w \
      --max-new-tokens 4000 --fixed-tokens-from "$src" --num-directions 4 \
      --radii 0.03,0.1,0.3 --layers 7,14,21,27 --out "$out" || echo "FAILED len_a1k $i $w"
  done
done
echo "len_a1k done $(date)"

# 2. Extend two 2000-token trajectories to 4000
for n in climate_models linguistics_syntax; do
  out=results/big4/traj4k/$n.npz
  [ -f "$out" ] && continue
  python3 -m chaos_llm.extend_trajectories --src results/long_traj_2k/$n.npz \
    --out "$out" --extra 2000 --window 4096 || echo "FAILED extend $n"
done

# 3. Three more 4000-token trajectories from prompts_long.txt
for i in 2 3 4; do
  sed -n "${i}p" prompts_long.txt > /tmp/T4_$i.txt
  out=results/big4/traj4k/T4_$i.npz
  [ -f "$out" ] && continue
  python3 -m chaos_llm.collect_trajectories --prompts /tmp/T4_$i.txt --window 4096 \
    --max-new-tokens 4000 --layers 7,14,21,27 --out results/big4/t4_tmp_$i || echo "FAILED traj T4_$i"
  mkdir -p results/big4/traj4k
  mv results/big4/t4_tmp_$i/*.npz "$out" 2>/dev/null
done

# 4. Their 4000-token teacher-forced responses
for f in results/big4/traj4k/T4_*.npz; do
  [ -f "$f" ] || continue
  b=$(basename "$f" .npz)
  out=results/big4/a1_4k/$b
  [ -f "$out/a1.npz" ] && continue
  python3 -m chaos_llm.run_a1_long --prompt @prompts_long.txt --window 100 \
    --max-new-tokens 4000 --fixed-tokens-from "$f" --num-directions 4 \
    --radii 0.03,0.1,0.3 --layers 7,14,21,27 --out "$out" || echo "FAILED a1_4k $b"
done

echo "QUEUE23 DONE $(date)"
