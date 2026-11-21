#!/bin/bash
# Control study, stage 1: Mamba-130m (negative control) through the same pipeline
# as the DeepSeek/LLM work. Resume-safe: every block checks for its output.
# Run:  nohup setsid bash scripts/ctrl_mamba_stage1.sh > results/CTRL_S1.log 2>&1 < /dev/null & disown
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv_tf/bin/python
M130=~/LLM/models--state-spaces--mamba-130m-hf
OUT=results/ctrl/mamba130
mkdir -p "$OUT"

echo "=== stage1 start $(date)"

# --- A1: prompt perturbation, teacher-forced, 10 radii x 20 dirs, 300 tokens
for PAIR in "prompts.txt:txt" "prompts_childhood.txt:childhood" "prompts_cyber.txt:cyber"; do
  P="${PAIR%%:*}"; TAG="${PAIR##*:}"
  if [ ! -f "$OUT/a1_$TAG/a1.npz" ]; then
    echo "--- a1 $TAG $(date)"
    $PY -m chaos_llm.run_a1 --model "$M130" --prompt @"$P" --window 4096 \
      --max-new-tokens 300 --num-directions 20 --batch-dirs 10 \
      --radii "1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1,3" \
      --layers "0,6,12,18,23" --out "$OUT/a1_$TAG" || echo "FAILED a1_$TAG"
  fi
done

# --- free runs: first-flip observations vs threshold predictions
for TAG in txt childhood cyber; do
  for r in 0.03 0.1 0.3 1 3; do
    if [ ! -f "$OUT/fr_${TAG}_r${r}/free_run.npz" ]; then
      echo "--- free run $TAG r=$r $(date)"
      $PY -m chaos_llm.run_thresholds --model "$M130" --run "$OUT/a1_$TAG" \
        --radii "$r" --chunk 6 --out "$OUT/fr_${TAG}_r${r}" || echo "FAILED fr $TAG $r"
    fi
  done
done

# --- teacher-forced vs free-run exactness (2 radii, interstellar only)
for r in 0.3 1; do
  if [ ! -f "$OUT/tfp_r${r}/tf_predict.npz" ]; then
    echo "--- tf_predict r=$r $(date)"
    $PY -m chaos_llm.run_tf_predict --model "$M130" --run "$OUT/a1_txt" \
      --free-run "$OUT/fr_txt_r${r}" --radii "$r" --chunk 6 \
      --out "$OUT/tfp_r${r}" || echo "FAILED tfp $r"
  fi
done

# --- trajectories for D2/RP: 16 prompts x 1000 tokens, same prompts as DeepSeek
if [ ! -f "$OUT/traj/linguistics_syntax.npz" ]; then
  echo "--- trajectories $(date)"
  $PY -m chaos_llm.collect_trajectories --model "$M130" --prompts prompts_long2.txt \
    --window 4096 --max-new-tokens 1000 --layers "6,12,18,23" \
    --out "$OUT/traj_tmp" && mkdir -p "$OUT/traj" && mv "$OUT"/traj_tmp/*.npz "$OUT/traj/" \
    || echo "FAILED traj"
fi

echo "=== stage1 done $(date)"
