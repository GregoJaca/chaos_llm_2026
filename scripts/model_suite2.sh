#!/bin/bash
MODEL=$1
TAG=$2
R=$3
L=$4
PY=~/LLM/venv_tf/bin/python
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=results/models/$TAG
mkdir -p "$OUT"
echo "MODEL SUITE $TAG start $(date)"

LAST=$((L-1))
ALL=$(seq -s, 0 $LAST)

echo "--- calib"
$PY scripts/r_calib.py --model "$MODEL" --dtype float16 --tokens 120 --dirs 6 > "$OUT/calib.txt" 2>&1 || echo "FAILED calib"

if [ ! -f "$OUT/a1/a1.npz" ]; then
  echo "--- a1"
  $PY -m chaos_llm.run_a1 --model "$MODEL" --prompt @prompts.txt --window 100 \
    --max-new-tokens 300 --num-directions 20 --radii "0.03,$R,0.3" \
    --layers "$ALL" --batch-dirs 1 --out "$OUT/a1" || echo "FAILED a1"
fi

if [ ! -f "$OUT/a1b/a1.npz" ]; then
  echo "--- a1b"
  $PY -m chaos_llm.run_a1 --model "$MODEL" --prompt @prompts_cyber.txt --window 100 \
    --max-new-tokens 300 --num-directions 20 --radii "0.03,$R,0.3" \
    --layers "$ALL" --batch-dirs 1 --out "$OUT/a1b" || echo "FAILED a1b"
fi

if [ ! -f "$OUT/state/state2.npz" ]; then
  echo "--- state"
  $PY -m chaos_llm.run_state2 --model "$MODEL" --run "$OUT/a1" --steps 16,50,100,150 \
    --dirs 12 --radii "0.03,$R" --layers "7,14,$LAST" --inject-layer 0 --batch-dirs 1 \
    --out "$OUT/state" || echo "FAILED state"
fi

for i in 1 2 3; do
  sed -n "${i}p" prompts_long2.txt > /tmp/MS_$i.txt
  if [ ! -s /tmp/MS_$i.txt ]; then continue; fi
  if [ ! -f "$OUT/traj/p$i.npz" ]; then
    echo "--- traj $i"
    $PY -m chaos_llm.collect_trajectories --model "$MODEL" --prompts /tmp/MS_$i.txt \
      --window 4096 --max-new-tokens 1000 --layers "7,14,21,$LAST" --out "$OUT/traj_tmp_$i" || echo "FAILED traj $i"
    mkdir -p "$OUT/traj"
    mv "$OUT"/traj_tmp_$i/*.npz "$OUT/traj/p$i.npz" 2>/dev/null
  fi
done

for i in 1 2 3; do
  sed -n "${i}p" prompts_long2.txt > /tmp/MS_$i.txt
  if [ ! -s /tmp/MS_$i.txt ]; then continue; fi
  if [ ! -f "$OUT/chaos_geom/p$i/a1.npz" ]; then
    echo "--- chaos geom $i"
    $PY -m chaos_llm.run_a1 --model "$MODEL" --prompt @/tmp/MS_$i.txt --window 100 \
      --max-new-tokens 300 --num-directions 20 --radii "0.01,$R" --layers "7,14,$LAST" \
      --batch-dirs 1 --out "$OUT/chaos_geom/p$i" || echo "FAILED geom $i"
  fi
  for r in 0.05 "$R" 0.2 0.4 0.8 1.5; do
    out=$OUT/chaos/p${i}_r${r}
    if [ -f "$out/free_run.npz" ]; then continue; fi
    echo "--- chaos $i r=$r"
    $PY -m chaos_llm.run_thresholds --model "$MODEL" --run "$OUT/chaos_geom/p$i" \
      --radii "$r" --chunk 6 --out "$out" || echo "FAILED chaos $i $r"
  done
done

if [ -f "$OUT/traj/p1.npz" ]; then
  for w in 100 4096; do
    out=$OUT/window/w$w
    if [ -f "$out/a1.npz" ]; then continue; fi
    echo "--- window $w"
    $PY -m chaos_llm.run_a1_long --model "$MODEL" --prompt @prompts_long2.txt --window "$w" \
      --max-new-tokens 300 --fixed-tokens-from "$OUT/traj/p1.npz" --num-directions 4 \
      --radii "$R" --layers "7,14,$LAST" --out "$out" || echo "FAILED window $w"
  done
fi

echo "MODEL SUITE $TAG done $(date)"
