#!/bin/bash
# 2026-11-26 update campaign: angles with many directions + small-radius linearity sweeps.
# Runs on x11 (Tesla T4). Resume-safe: skips outputs that already exist.
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY0=python3
PY=~/LLM/venv_tf/bin/python

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
Q25=~/LLM/models--Qwen--Qwen2.5-3B-Instruct
Q34=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PHI=~/LLM/models--microsoft--Phi-4-mini-instruct

run_angle () {  # model py dtype layers fixed out r
  local model=$1 py=$2 dt=$3 layers=$4 fixed=$5 out=$6 r=$7
  [ -f "$out/angle.npz" ] && { echo "skip $out"; return; }
  echo "=== angle $out dt=$dt r=$r $(date)"
  $py -m chaos_llm.run_angle --model "$model" --dtype "$dt" --window 100 \
    --fixed-tokens-from "$fixed" --num-directions 64 --radii "$r" \
    --layers "$layers" --out "$out" || echo "FAILED $out"
}

run_a1 () {  # model py dtype layers fixed out
  local model=$1 py=$2 dt=$3 layers=$4 fixed=$5 out=$6
  [ -f "$out/a1.npz" ] && { echo "skip $out"; return; }
  echo "=== a1 $out dt=$dt $(date)"
  $py -m chaos_llm.run_a1 --model "$model" --dtype "$dt" --prompt @prompts.txt \
    --window 100 --max-new-tokens 300 --num-directions 24 \
    --radii "1e-4,3e-4,1e-3,3e-3,1e-2,0.03,0.1,0.2,0.3" \
    --layers "$layers" --batch-dirs 2 --inbatch-ref \
    --fixed-tokens-from "$fixed" --out "$out" || echo "FAILED $out"
}

DSL=$(seq -s, 0 27)
Q25L=$(seq -s, 0 35)
Q34L=$(seq -s, 0 35)
PHIL=$(seq -s, 0 31)

# --- angles (64 directions, two radii)
run_angle "$DS"  "$PY0" float32 "$DSL"  results/a1_ds_w100_layers            results/update/angle_ds15   "0.03,0.1"
run_angle "$Q25" "$PY"  float16 "$Q25L" results/models/qwen25_3b/a1            results/update/angle_qwen25 "0.03,0.1"
run_angle "$Q34" "$PY"  float16 "$Q34L" results/models/qwen3_4b/a1             results/update/angle_qwen34 "0.03,0.1"
run_angle "$PHI" "$PY"  float16 "$PHIL" results/models/phi4mini/a1             results/update/angle_phi    "0.03,0.1"

# --- small-radius linearity sweeps (9 radii, 24 directions, all layers)
run_a1 "$DS"  "$PY0" float32 "$DSL"  results/a1_ds_w100_layers            results/update/a1r_ds15
run_a1 "$Q25" "$PY"  float16 "$Q25L" results/models/qwen25_3b/a1            results/update/a1r_qwen25
run_a1 "$Q34" "$PY"  float16 "$Q34L" results/models/qwen3_4b/a1             results/update/a1r_qwen34
run_a1 "$PHI" "$PY"  float16 "$PHIL" results/models/phi4mini/a1             results/update/a1r_phi

echo "QUEUE UPDATE DONE $(date)"
