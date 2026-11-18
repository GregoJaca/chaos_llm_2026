#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
mkdir -p results/big
echo "BIG QUEUE START $(date)"

run_a1 () {  # out, prompt_file, window, tokens, dirs, radii, layers
  local out=$1 pf=$2 w=$3 t=$4 d=$5 rr=$6 ll=$7
  if [ -f "$out/a1.npz" ]; then echo "skip $out"; return; fi
  python3 -m chaos_llm.run_a1 --model $M --prompt @$pf --window $w \
    --max-new-tokens $t --num-directions $d --radii $rr --layers $ll \
    --batch-dirs 1 --out "$out" || echo "FAILED $out"
}

# ---------- 1. FLUSH: windows x prompts x radii ----------
# prompt files: one prompt per line, first line used
i=0
for pf in prompts.txt prompts_cyber.txt prompts_long2.txt prompts_flush_extra.txt; do
  i=$((i+1))
  for w in 60 80 100 120 150 200 250 300 400 500; do
    run_a1 results/big/flush/p${i}_w${w} $pf $w 600 6 0.1 1,2,3,4,5,6
  done
done

# flush at two smaller radii, main prompt only
for w in 100 200 300; do
  run_a1 results/big/flush_r003/w${w} prompts.txt $w 600 6 0.03 1,2,3,4,5,6
  run_a1 results/big/flush_r03/w${w} prompts.txt $w 600 6 0.3 1,2,3,4,5,6
done
echo "flush done $(date)"

# ---------- 2. REGIMES: prompt perturbation ----------
i=0
for i in 1 2 3 4 5 6; do
  sed -n "${i}p" prompts_more.txt > /tmp/rp_$i.txt
  for w in 50 70 90 100 120 150 200 300 500 4096; do
    run_a1 results/big/regime/p${i}_w${w} /tmp/rp_$i.txt $w 400 12 0.03,0.1,0.3 7,14,21,27
  done
done
echo "regime done $(date)"

# ---------- 3. REGIMES: state perturbation ----------
for w in 50 70 90 100 150 300 4096; do
  if [ -f results/big/state_w${w}/state2.npz ]; then echo "skip state $w"; else
  python3 -m chaos_llm.run_state2 --run results/a1_ds_w100_layers \
    --steps 16,50,100,150 --dirs 16 --radii 0.03,0.1 --layers 7,14,21,27 \
    --inject-layer 0 --batch-dirs 1 --window $w \
    --out results/big/state_w${w} || echo "FAILED state $w"
  fi
done
echo "state done $(date)"

# ---------- 4. LAYER PROBE ----------
for pos in 16 32 50 75 100 150 200; do
  if [ -f results/big/probe_p${pos}/layer_probe.npz ]; then echo "skip probe $pos"; else
  python3 -m chaos_llm.run_layer_probe --run results/a1_ds_w100_layers \
    --layers 0,4,7,10,14,18,21,24,27 --dirs 6 --r 0.05 --pos $pos \
    --out results/big/probe_p${pos} || echo "FAILED probe $pos"
  fi
done
echo "probe done $(date)"

# ---------- 5. STABLE CHAOS: free runs across radii ----------
i=0
for i in 1 2 3 4; do
  sed -n "${i}p" prompts_more2.txt > /tmp/sc_$i.txt
  for r in 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.7 1.0 1.5 2.0; do
    out=results/big/chaos/p${i}_r${r}
    if [ -f "$out/free_run.npz" ]; then echo "skip $out"; continue; fi
    # need the teacher-forced geometry first
    if [ ! -f "results/big/chaos_geom/p${i}/a1.npz" ]; then
      run_a1 results/big/chaos_geom/p${i} /tmp/sc_$i.txt 100 300 20 0.01,0.1 7,14,21,27
    fi
    python3 -m chaos_llm.run_thresholds --run results/big/chaos_geom/p${i} \
      --radii $r --chunk 6 --out $out || echo "FAILED chaos $out"
  done
done
echo "chaos done $(date)"

echo "BIG QUEUE DONE $(date)"
