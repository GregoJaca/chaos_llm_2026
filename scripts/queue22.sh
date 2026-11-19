#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
mkdir -p results/big4
echo "QUEUE22 START $(date)"

# ---- 1. 4000-token trajectories for more prompts ----
for i in 1 2 3 4 5 6; do
  sed -n "${i}p" prompts_long2.txt > /tmp/L2_$i.txt
  out=results/big4/traj4k/L2_$i.npz
  [ -f "$out" ] && continue
  python3 -m chaos_llm.collect_trajectories --prompts /tmp/L2_$i.txt --window 4096 \
    --max-new-tokens 4000 --layers 7,14,21,27 --out results/big4/traj4k_tmp_$i || echo "FAILED traj L2_$i"
  mv results/big4/traj4k_tmp_$i/*.npz "$out" 2>/dev/null
done
for i in 1 2 3; do
  sed -n "${i}p" prompts_more2.txt > /tmp/M2_$i.txt
  out=results/big4/traj4k/M2_$i.npz
  [ -f "$out" ] && continue
  python3 -m chaos_llm.collect_trajectories --prompts /tmp/M2_$i.txt --window 4096 \
    --max-new-tokens 4000 --layers 7,14,21,27 --out results/big4/traj4k_tmp_$i || echo "FAILED traj M2_$i"
  mv results/big4/traj4k_tmp_$i/*.npz "$out" 2>/dev/null
done
echo "traj4k done $(date)"

# ---- 2. 4000-token teacher-forced response on every trajectory ----
for f in results/big4/traj4k/*.npz; do
  [ -f "$f" ] || continue
  b=$(basename "$f" .npz)
  out=results/big4/a1_4k/$b
  [ -f "$out/a1.npz" ] && continue
  python3 -m chaos_llm.run_a1_long --prompt @prompts_long2.txt --window 100 \
    --max-new-tokens 4000 --fixed-tokens-from "$f" --num-directions 4 \
    --radii 0.03,0.1,0.3 --layers 7,14,21,27 --out "$out" || echo "FAILED a1_4k $b"
done
echo "a1_4k done $(date)"

# ---- 3. Prompt-length series: 4 lengths x 4000-token trajectories ----
for i in 1 2 3 4; do
  sed -n "${i}p" prompts_lengths.txt > /tmp/LEN_$i.txt
  out=results/big4/len_traj/p$i.npz
  [ -f "$out" ] && continue
  python3 -m chaos_llm.collect_trajectories --prompts /tmp/LEN_$i.txt --window 4096 \
    --max-new-tokens 4000 --layers 7,14,21,27 --out results/big4/len_tmp_$i || echo "FAILED len $i"
  mv results/big4/len_tmp_$i/*.npz "$out" 2>/dev/null
done
echo "len traj done $(date)"

# ---- 4. Flush sweep on the four prompt lengths (600 tokens) ----
for i in 1 2 3 4; do
  sed -n "${i}p" prompts_lengths.txt > /tmp/LEN_$i.txt
  src=results/big4/len_traj/p$i.npz
  [ -f "$src" ] || continue
  for w in 60 100 150 200 300 500; do
    out=results/big4/flush_len/p${i}_w${w}
    [ -f "$out/a1.npz" ] && continue
    python3 -m chaos_llm.run_a1_long --prompt @/tmp/LEN_$i.txt --window $w \
      --max-new-tokens 600 --fixed-tokens-from "$src" --num-directions 4 \
      --radii 0.1 --layers 1,2,3,4,5,6 --out "$out" || echo "FAILED flush len $i $w"
  done
done
echo "flush_len done $(date)"

# ---- 5. More stable-chaos prompts ----
for i in 3 4 5 6; do
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
echo "QUEUE22 DONE $(date)"
