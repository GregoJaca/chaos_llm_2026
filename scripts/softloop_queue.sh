#!/bin/bash
# Soft-generation loop campaign: temperature grid, loop vs replay decomposition.
# Resume-safe: skips outputs that already exist.
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
MM=~/LLM/models--state-spaces--mamba-130m-hf
QM=~/LLM/models--Qwen--Qwen2.5-3B-Instruct
TEMPS="1e-9,0.05,0.1,0.15,0.2,0.3,0.5,0.7,1.0,1.4,2.0"
STEPS=600
LAYERS=0,7,14,21,27
echo "SOFTLOOP QUEUE START $(date)"

run_pair () {  # $1 model  $2 outdir  $3 extra args
  local MODEL=$1 OUT=$2; shift 2
  for T in $(echo $TEMPS | tr ',' ' '); do
    D=$OUT/T$T
    if [ ! -f $D/softloop.npz ]; then
      echo "=== loop T=$T $(date)"
      python3 -m chaos_llm.run_softloop --model $MODEL --steps $STEPS --temp $T \
        --dirs 8 --r 0.03 --save-layers $LAYERS --out $D "$@"
    fi
    if [ ! -f $D/replay/softloop.npz ] && [ -f $D/softloop.npz ]; then
      echo "=== replay T=$T $(date)"
      python3 -m chaos_llm.run_softloop --model $MODEL --steps $STEPS --temp $T \
        --mode replay --replay-file $D/softloop.npz \
        --dirs 8 --r 0.03 --save-layers $LAYERS --out $D/replay "$@"
    fi
  done
}

# 1. DeepSeek-1.5B fp32, full attention
run_pair $M results/softloop/ds15 --window 4096

# 2. DeepSeek-1.5B fp32, true window 100
run_pair $M results/softloop/ds15_w100 --window 100

# 3. Mamba-130m fp32 (no window mechanism; --window ignored)
run_pair $MM results/softloop/mamba130 --window 0

# 4. Qwen2.5-3B fp16
run_pair $QM results/softloop/qwen25_3b --window 4096

# 5. rate-consistency radii on DeepSeek at three temperatures
for T in 1e-9,0.35,1.0; do :; done
for T in 1e-9 0.35 1.0; do
  D=results/softloop/ds15_r01/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== loop r=0.1 T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $M --steps $STEPS --temp $T \
      --dirs 8 --r 0.1 --save-layers $LAYERS --out $D --window 4096
  fi
done

echo "SOFTLOOP QUEUE DONE $(date)"
