#!/bin/bash
# Fixup 3: qwen3_4b loopval (needs venv_tf) + anything the main fixup missed.
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q34=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PYTF=~/LLM/venv_tf/bin/python

while pgrep -f "bash scripts/softloop_fixup2.sh" > /dev/null; do sleep 120; done
echo "FIXUP3 START $(date)"

for T in 0.05 0.1 0.2 1.0; do
  D=results/softloop_val/qwen3_4b/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== loopval qwen3_4b T=$T $(date)"
    $PYTF -m chaos_llm.run_softloop --model $Q34 --steps 300 --temp $T \
      --dirs 16 --r 0.3 --save-layers 35 --out $D --window 4096 \
      || echo "FAILED q34 loopval T=$T"
  fi
done

# mamba loopval (its own tokenizer scale) for the per-model law figure
MM=~/LLM/models--state-spaces--mamba-130m-hf
for T in 0.05 0.1 0.2 1.0; do
  D=results/softloop_val/mamba130/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== loopval mamba130 T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $MM --steps 300 --temp $T \
      --dirs 16 --r 0.1 --save-layers 23 --out $D --window 0 \
      || echo "FAILED mamba loopval T=$T"
  fi
done

# randqwen + pythia loopval (controls)
for M in "$HOME/LLM/models--randqwen-g1.0 randqwen 27 0.1" \
         "$HOME/LLM/models--EleutherAI--pythia-410m pythia410 23 0.1"; do
  set -- $M
  for T in 0.05 0.1 0.2 1.0; do
    D=results/softloop_val/$2/T$T
    if [ ! -f $D/softloop.npz ]; then
      echo "=== loopval $2 T=$T $(date)"
      python3 -m chaos_llm.run_softloop --model $1 --steps 300 --temp $T \
        --dirs 16 --r $4 --save-layers $3 --out $D --window 4096 \
        || echo "FAILED $2 loopval T=$T"
    fi
  done
done

echo "FIXUP3 DONE $(date)"
