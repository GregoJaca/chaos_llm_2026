#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "QUEUE20 START $(date)"

# 1. Extend three 1000-token trajectories to 3000 tokens
for n in ocean_currents history_of_computing urban_transportation; do
  out=results/big/long3k/${n}.npz
  [ -f "$out" ] && continue
  python3 -m chaos_llm.extend_trajectories --model $M \
    --src results/long_last/${n}.npz --out "$out" --extra 2000 --window 4096 \
    || echo "FAILED extend $n"
done
echo "extend done $(date)"

# 2. Long-horizon A1: does 1000 tokens fit?
python3 -m chaos_llm.run_a1 --model $M --prompt @prompts_long2.txt --window 100 \
  --max-new-tokens 1000 --fixed-tokens-from results/long_traj_2k/ocean_currents.npz \
  --num-directions 4 --radii 0.1 --layers 7,14,21,27 --batch-dirs 1 \
  --out results/big/long1000 || echo "FAILED long1000"

# 3. Missing chaos prompt p6 (geometry + radii)
sed -n "6p" prompts_more2.txt > /tmp/sc_6.txt
if [ ! -s /tmp/sc_6.txt ]; then cp prompts_more2.txt /tmp/sc_6.txt; fi
python3 -m chaos_llm.run_a1 --model $M --prompt @/tmp/sc_6.txt --window 100 \
  --max-new-tokens 300 --num-directions 20 --radii 0.01,0.1 --layers 7,14,21,27 \
  --batch-dirs 1 --out results/big/chaos_geom/p6 || echo "FAILED geom p6"
for r in 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.7 1.0 1.5 2.0; do
  out=results/big/chaos/p6_r${r}
  [ -f "$out/free_run.npz" ] && continue
  python3 -m chaos_llm.run_thresholds --run results/big/chaos_geom/p6 \
    --radii $r --chunk 6 --out $out || echo "FAILED chaos $out"
done

# 4. Fine window scan on three more prompts
i=0
for pf in prompts_long.txt prompts_more.txt prompts_more2.txt; do
  i=$((i+1)); base=scanB_$i
  for w in 55 60 65 70 75 80 85 90 95 105 110 120 130; do
    out=results/big/scan/${base}_w${w}
    [ -f "$out/a1.npz" ] && continue
    python3 -m chaos_llm.run_a1 --model $M --prompt @$pf --window $w \
      --max-new-tokens 300 --num-directions 6 --radii 0.1 --layers 7,14,21,27 \
      --batch-dirs 1 --out $out || echo "FAILED scan $out"
  done
done

echo "QUEUE20 DONE $(date)"
