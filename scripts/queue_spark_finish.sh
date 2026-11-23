#!/bin/bash
# ============================================================================
# Spark queue: finish missing Qwen3.5-9B runs (T=0.2, 0.3, 0.5) and run
# DeepSeek-14B soft-loop campaign (fp16, B=33, 600 steps) + replay.
# ============================================================================
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv5/bin/python
Q35=~/LLM/models/Qwen3.5-9B
DS14=~/LLM/models/DeepSeek-R1-Distill-Qwen-14B
LOG=~/LLM/chaos_llm_2026/results_big/spark_queue.log

echo "=== SPARK QUEUE START $(date) ===" | tee -a "$LOG"

# 1. Missing Qwen3.5-9B runs
for T in 0.2 0.3 0.5; do
  D=results_big/softloop/qwen35_9b/T$T
  mkdir -p "$D"
  if [ ! -f "$D/softloop.npz" ]; then
    echo "=== q35 loop T=$T $(date) ===" | tee -a "$LOG"
    $PY -m chaos_llm.run_softloop --model "$Q35" --steps 600 --temp "$T" \
      --dirs 32 --r 0.3 --save-layers 31 --out "$D" --window 0 2>&1 | tee -a "$LOG"
  fi
  if [ ! -f "$D/replay/softloop.npz" ] && [ -f "$D/softloop.npz" ]; then
    echo "=== q35 replay T=$T $(date) ===" | tee -a "$LOG"
    $PY -m chaos_llm.run_softloop --model "$Q35" --steps 600 --temp "$T" \
      --mode replay --replay-file "$D/softloop.npz" \
      --dirs 32 --r 0.3 --save-layers 31 --out "$D/replay" --window 0 2>&1 | tee -a "$LOG"
  fi
done

# 2. DeepSeek-14B softloop grid (fp16, 600 steps, K=32, r=0.03)
for T in 1e-9 0.02 0.05 0.1 0.2 0.5 1.0; do
  D=results_big/softloop/ds14/T$T
  mkdir -p "$D"
  if [ ! -f "$D/softloop.npz" ]; then
    echo "=== ds14 loop T=$T $(date) ===" | tee -a "$LOG"
    $PY -m chaos_llm.run_softloop --model "$DS14" --dtype float16 \
      --steps 600 --temp "$T" --dirs 32 --r 0.03 --save-layers 47 \
      --out "$D" --window 4096 2>&1 | tee -a "$LOG"
  fi
  if [ ! -f "$D/replay/softloop.npz" ] && [ -f "$D/softloop.npz" ]; then
    echo "=== ds14 replay T=$T $(date) ===" | tee -a "$LOG"
    $PY -m chaos_llm.run_softloop --model "$DS14" --dtype float16 \
      --steps 600 --temp "$T" --mode replay --replay-file "$D/softloop.npz" \
      --dirs 32 --r 0.03 --save-layers 47 --out "$D/replay" --window 4096 2>&1 | tee -a "$LOG"
  fi
done

# Package spark results for easy rsync/scp to local
echo "=== Packaging spark softloop results $(date) ===" | tee -a "$LOG"
tar -czf /tmp/spark_softloop_all.tgz results_big/softloop/

echo "=== SPARK QUEUE COMPLETED $(date) ===" | tee -a "$LOG"
