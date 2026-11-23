#!/bin/bash
# ============================================================================
# X11 queue: run sampling modes (CRN & Independent) across additional architectures
# (Phi-4-mini, Qwen3-4B, Pythia-410m, Mamba-130m, Qwen2.5-3B)
# ============================================================================
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
PHI=~/LLM/models--microsoft--Phi-4-mini-instruct
Q25=~/LLM/models--Qwen--Qwen2.5-3B-Instruct
Q34=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PYT=~/LLM/models--EleutherAI--pythia-410m
MM=~/LLM/models--state-spaces--mamba-130m-hf

LOG=results/x11_sampling_queue.log
echo "=== X11 SAMPLING QUEUE START $(date) ===" | tee -a "$LOG"

# 1. Phi-4-mini
for T in 0.5 1.0; do
  D=results/softloop_sample/phi4mini/crn_T$T
  if [ ! -f "$D/softloop.npz" ]; then
    echo "=== phi4 crn T=$T $(date) ===" | tee -a "$LOG"
    python3 -m chaos_llm.run_softloop --model "$PHI" --steps 600 --temp "$T" \
      --dirs 16 --r 0.3 --feedback sample_crn --save-layers 31 --out "$D" 2>&1 | tee -a "$LOG"
  fi
  D=results/softloop_sample/phi4mini/ind_T$T
  if [ ! -f "$D/softloop.npz" ]; then
    echo "=== phi4 ind T=$T $(date) ===" | tee -a "$LOG"
    python3 -m chaos_llm.run_softloop --model "$PHI" --steps 600 --temp "$T" \
      --dirs 16 --r 0.3 --feedback sample_ind --save-layers 31 --out "$D" 2>&1 | tee -a "$LOG"
  fi
done

# 2. Qwen3-4B
for T in 0.5 1.0; do
  D=results/softloop_sample/qwen3_4b/crn_T$T
  if [ ! -f "$D/softloop.npz" ]; then
    echo "=== qwen34 crn T=$T $(date) ===" | tee -a "$LOG"
    python3 -m chaos_llm.run_softloop --model "$Q34" --steps 600 --temp "$T" \
      --dirs 16 --r 0.3 --feedback sample_crn --save-layers 27 --out "$D" 2>&1 | tee -a "$LOG"
  fi
  D=results/softloop_sample/qwen3_4b/ind_T$T
  if [ ! -f "$D/softloop.npz" ]; then
    echo "=== qwen34 ind T=$T $(date) ===" | tee -a "$LOG"
    python3 -m chaos_llm.run_softloop --model "$Q34" --steps 600 --temp "$T" \
      --dirs 16 --r 0.3 --feedback sample_ind --save-layers 27 --out "$D" 2>&1 | tee -a "$LOG"
  fi
done

# 3. Pythia-410m
for T in 0.5 1.0; do
  D=results/softloop_sample/pythia410/crn_T$T
  if [ ! -f "$D/softloop.npz" ]; then
    echo "=== pythia crn T=$T $(date) ===" | tee -a "$LOG"
    python3 -m chaos_llm.run_softloop --model "$PYT" --steps 600 --temp "$T" \
      --dirs 16 --r 0.1 --feedback sample_crn --save-layers 23 --out "$D" 2>&1 | tee -a "$LOG"
  fi
  D=results/softloop_sample/pythia410/ind_T$T
  if [ ! -f "$D/softloop.npz" ]; then
    echo "=== pythia ind T=$T $(date) ===" | tee -a "$LOG"
    python3 -m chaos_llm.run_softloop --model "$PYT" --steps 600 --temp "$T" \
      --dirs 16 --r 0.1 --feedback sample_ind --save-layers 23 --out "$D" 2>&1 | tee -a "$LOG"
  fi
done

# 4. Mamba-130m (T=0.5)
D=results/softloop_sample/mamba130/crn_T0.5
if [ ! -f "$D/softloop.npz" ]; then
  echo "=== mamba crn T=0.5 $(date) ===" | tee -a "$LOG"
  python3 -m chaos_llm.run_softloop --model "$MM" --steps 600 --temp 0.5 \
    --dirs 16 --r 0.1 --feedback sample_crn --save-layers 23 --out "$D" --window 0 2>&1 | tee -a "$LOG"
fi
D=results/softloop_sample/mamba130/ind_T0.5
if [ ! -f "$D/softloop.npz" ]; then
  echo "=== mamba ind T=0.5 $(date) ===" | tee -a "$LOG"
  python3 -m chaos_llm.run_softloop --model "$MM" --steps 600 --temp 0.5 \
    --dirs 16 --r 0.1 --feedback sample_ind --save-layers 23 --out "$D" --window 0 2>&1 | tee -a "$LOG"
fi

# 5. Qwen2.5-3B (T=0.5)
D=results/softloop_sample/qwen25_3b/crn_T0.5
if [ ! -f "$D/softloop.npz" ]; then
  echo "=== qwen25 crn T=0.5 $(date) ===" | tee -a "$LOG"
  python3 -m chaos_llm.run_softloop --model "$Q25" --steps 600 --temp 0.5 \
    --dirs 16 --r 0.3 --feedback sample_crn --save-layers 35 --out "$D" 2>&1 | tee -a "$LOG"
fi
D=results/softloop_sample/qwen25_3b/ind_T0.5
if [ ! -f "$D/softloop.npz" ]; then
  echo "=== qwen25 ind T=0.5 $(date) ===" | tee -a "$LOG"
  python3 -m chaos_llm.run_softloop --model "$Q25" --steps 600 --temp 0.5 \
    --dirs 16 --r 0.3 --feedback sample_ind --save-layers 35 --out "$D" 2>&1 | tee -a "$LOG"
fi

echo "=== X11 SAMPLING QUEUE COMPLETED $(date) ===" | tee -a "$LOG"
