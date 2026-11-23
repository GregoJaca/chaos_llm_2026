#!/bin/bash
# Fix-ups after the main campaign: qwen3_4b (needs venv_tf / transformers 4.57)
# and pythia greedy-orbit gates (24 layers, use --save-layers 23).
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Q34=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
PYT=~/LLM/models--EleutherAI--pythia-410m
PYTF=~/LLM/venv_tf/bin/python

while pgrep -f "bash scripts/softloop_campaign.sh" > /dev/null; do sleep 120; done
echo "FIXUP START $(date)"

# qwen3_4b gates (venv_tf)
for p in p1 p2 p3 p4 p5; do
  D=results/softloop_gates/qwen3_4b/$p
  if [ ! -f $D/gates.npz ]; then
    echo "=== gates qwen3_4b/$p $(date)"
    $PYTF -m chaos_llm.run_gates --model $Q34 \
      --a1 results/rstar/qwen3_4b/$p/a1.npz --out $D || echo "FAILED q34/$p"
  fi
done

# qwen3_4b loop grid + replay (venv_tf), and pythia greedy orbit + gates
TEMPS="1e-9,0.05,0.1,0.15,0.2,0.3,0.5,0.7,1.0,1.4,2.0"
for T in $(echo $TEMPS | tr ',' ' '); do
  D=results/softloop/qwen3_4b/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== grid qwen3_4b T=$T $(date)"
    $PYTF -m chaos_llm.run_softloop --model $Q34 --steps 600 --temp $T \
      --dirs 16 --r 0.3 --save-layers 0,35 --out $D --window 4096 \
      || echo "FAILED q34 T=$T"
  fi
  if [ ! -f $D/replay/softloop.npz ] && [ -f $D/softloop.npz ]; then
    echo "=== replay qwen3_4b T=$T $(date)"
    $PYTF -m chaos_llm.run_softloop --model $Q34 --steps 600 --temp $T \
      --mode replay --replay-file $D/softloop.npz \
      --dirs 16 --r 0.3 --save-layers 0,35 --out $D/replay --window 4096
  fi
done

D=results/softloop_greedy/pythia410
if [ ! -f $D/softloop.npz ]; then
  echo "=== greedy orbit pythia410 (fixed layers) $(date)"
  python3 -m chaos_llm.run_softloop --model $PYT --steps 300 --temp 1.0 \
    --feedback argmax --dirs 8 --r 0.1 --save-layers 23 --out $D --window 4096
fi
if [ -f $D/softloop.npz ] && [ ! -f results/softloop_gates/pythia410/loop0/gates.npz ]; then
  echo "=== gates pythia410 $(date)"
  python3 -m chaos_llm.run_gates --model $PYT --loop $D/softloop.npz \
    --out results/softloop_gates/pythia410/loop0
fi

# qwen3_5 (only if venv311 torch works now)
if ~/LLM/venv311/bin/python -c "import torch" 2>/dev/null; then
  echo "venv311 OK $(date)"
  D=results/softloop_greedy/qwen3_5
  if [ ! -f $D/softloop.npz ]; then
    ~/LLM/venv311/bin/python -m chaos_llm.run_softloop \
      --model ~/LLM/models--Qwen--Qwen3.5-2B --steps 300 --temp 1.0 \
      --feedback argmax --dirs 8 --r 0.1 --save-layers 23 --out $D --window 4096 \
      || echo "FAILED q35_2b greedy"
  fi
  if [ -f $D/softloop.npz ] && [ ! -f results/softloop_gates/qwen3_5/loop0/gates.npz ]; then
    ~/LLM/venv311/bin/python -m chaos_llm.run_gates \
      --model ~/LLM/models--Qwen--Qwen3.5-2B --loop $D/softloop.npz \
      --out results/softloop_gates/qwen3_5/loop0
  fi
  for T in 1e-9 0.1 0.2 0.5 1.0 1.4; do
    D2=results/softloop/qwen3_5/T$T
    if [ ! -f $D2/softloop.npz ]; then
      echo "=== grid qwen3_5 T=$T $(date)"
      ~/LLM/venv311/bin/python -m chaos_llm.run_softloop \
        --model ~/LLM/models--Qwen--Qwen3.5-2B --steps 600 --temp $T \
        --dirs 16 --r 0.1 --save-layers 23 --out $D2 --window 4096 \
        || echo "FAILED qwen3_5 T=$T"
    fi
  done
else
  echo "venv311 still broken $(date)"
fi

echo "FIXUP DONE $(date)"
