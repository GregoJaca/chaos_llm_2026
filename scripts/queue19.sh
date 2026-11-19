#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "QUEUE19 START $(date)"

# 1. Stable chaos on four more prompts (12 radii x 20 dirs, 300 tokens)
for i in 5 6; do
  sed -n "${i}p" prompts_more2.txt > /tmp/sc_$i.txt
  if [ ! -f results/big/chaos_geom/p${i}/a1.npz ]; then
    python3 -m chaos_llm.run_a1 --model $M --prompt @/tmp/sc_$i.txt --window 100 \
      --max-new-tokens 300 --num-directions 20 --radii 0.01,0.1 --layers 7,14,21,27 \
      --batch-dirs 1 --out results/big/chaos_geom/p${i} || echo "FAILED geom p$i"
  fi
  for r in 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.7 1.0 1.5 2.0; do
    out=results/big/chaos/p${i}_r${r}
    [ -f "$out/free_run.npz" ] && continue
    python3 -m chaos_llm.run_thresholds --run results/big/chaos_geom/p${i} \
      --radii $r --chunk 6 --out $out || echo "FAILED chaos $out"
  done
done
echo "chaos2 done $(date)"

# 2. Fine window scan around P for three prompts
for pf in prompts.txt prompts_cyber.txt prompts_long2.txt; do
  base=$(basename $pf .txt)
  for w in 55 60 65 70 75 80 85 90 95 105 110 120 130; do
    out=results/big/scan/${base}_w${w}
    [ -f "$out/a1.npz" ] && continue
    python3 -m chaos_llm.run_a1 --model $M --prompt @$pf --window $w \
      --max-new-tokens 300 --num-directions 6 --radii 0.1 --layers 7,14,21,27 \
      --batch-dirs 1 --out $out || echo "FAILED scan $out"
  done
done
echo "scan done $(date)"

# 3. State-window grid: more windows and positions
for w in 60 80 100 120 200 400 1000; do
  out=results/big/state_grid/w${w}
  [ -f "$out/state2.npz" ] && continue
  python3 -m chaos_llm.run_state2 --run results/a1_ds_w100_layers \
    --steps 8,16,32,50,100,150 --dirs 16 --radii 0.03,0.1 \
    --layers 7,14,21,27 --inject-layer 0 --batch-dirs 1 --window $w \
    --out $out || echo "FAILED state $out"
done
echo "state grid done $(date)"

# 4. Layer probes at two radii and five positions
for r in 0.02 0.2; do
  for pos in 16 32 50 100 150; do
    out=results/big/probe_r${r}_p${pos}
    [ -f "$out/layer_probe.npz" ] && continue
    python3 -m chaos_llm.run_layer_probe --run results/a1_ds_w100_layers \
      --layers 0,4,7,10,14,18,21,24,27 --dirs 6 --r $r --pos $pos \
      --out $out || echo "FAILED probe $out"
  done
done
echo "probes done $(date)"

# 5. Long horizons: 500-token A1 for the six regime prompts
for i in 1 2 3 4 5 6; do
  sed -n "${i}p" prompts_more.txt > /tmp/lh_$i.txt
  out=results/big/long500/p${i}
  [ -f "$out/a1.npz" ] && continue
  python3 -m chaos_llm.run_a1 --model $M --prompt @/tmp/lh_$i.txt --window 100 \
    --max-new-tokens 500 --num-directions 8 --radii 0.1 --layers 7,14,21,27 \
    --batch-dirs 1 --out $out || echo "FAILED long $out"
done

echo "QUEUE19 DONE $(date)"
