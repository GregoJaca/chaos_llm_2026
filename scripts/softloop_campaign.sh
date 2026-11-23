#!/bin/bash
# ============================================================================
# SOFTLOOP CAMPAIGN v2 (x11, Tesla T4) — gates for all small models, loop
# validation, extra architectures, dense directions, attractor states,
# sampling (CRN / independent) statistics.  Resume-safe: skips existing
# outputs.  Run detached: nohup setsid bash scripts/softloop_campaign.sh
# ============================================================================
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DS=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
PHI=~/LLM/models--microsoft--Phi-4-mini-instruct
Q25=~/LLM/models--Qwen--Qwen2.5-3B-Instruct
Q34=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507
RAND=~/LLM/models--randqwen-g1.0
PYT=~/LLM/models--EleutherAI--pythia-410m
MM=~/LLM/models--state-spaces--mamba-130m-hf
echo "CAMPAIGN START $(date)"

gates_a1 () {  # model  tag
  for p in p1 p2 p3 p4 p5; do
    D=results/softloop_gates/$2/$p
    if [ ! -f $D/gates.npz ] && [ -f results/rstar/$2/$p/a1.npz ]; then
      echo "=== gates $2/$p $(date)"
      python3 -m chaos_llm.run_gates --model $1 \
        --a1 results/rstar/$2/$p/a1.npz --out $D || echo "FAILED $2/$p"
    fi
  done
}

# ---- A. gate curves from the existing 2560-direction rstar corpora ----
gates_a1 $DS   deepseek
gates_a1 $PHI  phi4mini
gates_a1 $Q25  qwen25_3b
gates_a1 $Q34  qwen3_4b

# ---- A2. control models: greedy orbit (argmax feedback) then gates ----
for M in "$RAND randqwen" "$PYT pythia410"; do
  set -- $M
  D=results/softloop_greedy/$2
  if [ ! -f $D/softloop.npz ]; then
    echo "=== greedy orbit $2 $(date)"
    python3 -m chaos_llm.run_softloop --model $1 --steps 300 --temp 1.0 \
      --feedback argmax --dirs 8 --r 0.1 --save-layers 27 --out $D --window 4096
  fi
  if [ -f $D/softloop.npz ] && [ ! -f results/softloop_gates/$2/loop0/gates.npz ]; then
    echo "=== gates $2 $(date)"
    python3 -m chaos_llm.run_gates --model $1 --loop $D/softloop.npz \
      --out results/softloop_gates/$2/loop0
  fi
done
# mamba greedy-orbit gates (loop T=1e-9 already has top2)
if [ -f results/softloop/mamba130/T1e-9/softloop.npz ] && \
   [ ! -f results/softloop_gates/mamba130/loop0/gates.npz ]; then
  python3 -m chaos_llm.run_gates --model $MM \
    --loop results/softloop/mamba130/T1e-9/softloop.npz \
    --out results/softloop_gates/mamba130/loop0
fi

# ---- B. loop validation of the gate law per small model (fp16 -> r=0.3) ----
for M in "$PHI phi4mini 0.3" "$Q25 qwen25_3b 0.3" "$Q34 qwen3_4b 0.3"; do
  set -- $M
  for T in 0.05 0.1 0.2 1.0; do
    D=results/softloop_val/$2/T$T
    if [ ! -f $D/softloop.npz ]; then
      echo "=== loopval $2 T=$T $(date)"
      python3 -m chaos_llm.run_softloop --model $1 --steps 300 --temp $T \
        --dirs 16 --r $3 --save-layers 27 --out $D --window 4096 \
        || echo "FAILED $2 T=$T"
    fi
  done
done

# ---- C. full loop grids for the remaining architectures ----
TEMPS="1e-9,0.05,0.1,0.15,0.2,0.3,0.5,0.7,1.0,1.4,2.0"
for M in "$Q34 qwen3_4b 0.3 27" "$PHI phi4mini 0.3 31" "$RAND randqwen 0.1 27" "$PYT pythia410 0.1 23"; do
  set -- $M
  for T in $(echo $TEMPS | tr ',' ' '); do
    D=results/softloop/$2/T$T
    if [ ! -f $D/softloop.npz ]; then
      echo "=== grid $2 T=$T $(date)"
      python3 -m chaos_llm.run_softloop --model $1 --steps 600 --temp $T \
        --dirs 16 --r $3 --save-layers 0,$4 --out $D --window 4096 \
        || echo "FAILED $2 T=$T"
    fi
    if [ ! -f $D/replay/softloop.npz ] && [ -f $D/softloop.npz ]; then
      echo "=== replay $2 T=$T $(date)"
      python3 -m chaos_llm.run_softloop --model $1 --steps 600 --temp $T \
        --mode replay --replay-file $D/softloop.npz \
        --dirs 16 --r $3 --save-layers 0,$4 --out $D/replay --window 4096
    fi
  done
done

# ---- D. dense direction grids (survival statistics) ----
for P in prompts.txt prompts_childhood.txt prompts_cyber.txt; do
  for T in 0.2 1.0; do
    D=results/softloop_dense/ds15/p${P%.txt}/T$T
    if [ ! -f $D/softloop.npz ]; then
      echo "=== dense ds15 p=$P T=$T $(date)"
      python3 -m chaos_llm.run_softloop --model $DS --steps 600 --temp $T \
        --prompt @$P --dirs 128 --r 0.03 --save-layers 27 \
        --out $D --window 4096 || echo "FAILED dense $P T=$T"
    fi
  done
done
for T in 0.2 0.5 1.0 1.4; do
  D=results/softloop_dense/qwen25_3b/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== dense qwen25 T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $Q25 --steps 600 --temp $T \
      --dirs 64 --r 0.3 --save-layers 35 --out $D --window 4096
  fi
done

# ---- E. attractor states for the D2 / TwoNN analysis, per model ----
st () {  # model tag T out_extra
  D=results/softloop_states/$2/T$3
  if [ ! -f $D/softloop.npz ]; then
    echo "=== states $2 T=$3 $(date)"
    python3 -m chaos_llm.run_softloop --model $1 --steps 3000 --temp $3 \
      --dirs 2 --r 0.1 --save-layers 27 --save-states --out $D --window 4096 \
      || echo "FAILED states $2 T=$3"
  fi
}
st $DS ds15 0.2; st $DS ds15 0.3; st $DS ds15 0.7; st $DS ds15 1.4
st $MM mamba130 1e-9; st $MM mamba130 0.5; st $MM mamba130 1.0
st $Q25 qwen25_3b 1e-9; st $Q25 qwen25_3b 1.0
st $RAND randqwen 1.0
st $PYT pythia410 1.0

# ---- F. sampling statistics (random dynamical system) ----
for T in 0.3 0.5 1.0; do
  D=results/softloop_sample/ds15/crn_T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== sample_crn ds15 T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $DS --steps 600 --temp $T \
      --dirs 16 --r 0.03 --feedback sample_crn --save-layers 27 --out $D
  fi
done
for T in 0.5 1.0; do
  D=results/softloop_sample/ds15/ind_T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== sample_ind ds15 T=$T $(date)"
    python3 -m chaos_llm.run_softloop --model $DS --steps 600 --temp $T \
      --dirs 16 --r 0.03 --feedback sample_ind --save-layers 27 --out $D
  fi
done
# CRN radius grid: is the sampled-path conditional response contracting at every r?
for R in 0.003 0.01 0.1 0.3; do
  D=results/softloop_sample/ds15/crn_T1.0_r$R
  if [ ! -f $D/softloop.npz ]; then
    echo "=== sample_crn ds15 T=1.0 r=$R $(date)"
    python3 -m chaos_llm.run_softloop --model $DS --steps 600 --temp 1.0 \
      --dirs 16 --r $R --feedback sample_crn --save-layers 27 --out $D
  fi
done
# sampling on a second architecture + the SSM
D=results/softloop_sample/qwen25_3b/crn_T1.0
if [ ! -f $D/softloop.npz ]; then
  python3 -m chaos_llm.run_softloop --model $Q25 --steps 600 --temp 1.0 \
    --dirs 16 --r 0.3 --feedback sample_crn --save-layers 35 --out $D
fi
D=results/softloop_sample/qwen25_3b/ind_T1.0
if [ ! -f $D/softloop.npz ]; then
  python3 -m chaos_llm.run_softloop --model $Q25 --steps 600 --temp 1.0 \
    --dirs 16 --r 0.3 --feedback sample_ind --save-layers 35 --out $D
fi
D=results/softloop_sample/mamba130/crn_T1.0
if [ ! -f $D/softloop.npz ]; then
  python3 -m chaos_llm.run_softloop --model $MM --steps 600 --temp 1.0 \
    --dirs 16 --r 0.1 --feedback sample_crn --save-layers 23 --out $D --window 0
fi
D=results/softloop_sample/mamba130/ind_T1.0
if [ ! -f $D/softloop.npz ]; then
  python3 -m chaos_llm.run_softloop --model $MM --steps 600 --temp 1.0 \
    --dirs 16 --r 0.1 --feedback sample_ind --save-layers 23 --out $D --window 0
fi

# ---- G. Qwen3.5-2B (only if venv311 torch got fixed) ----
if ~/LLM/venv311/bin/python -c "import torch" 2>/dev/null; then
  echo "venv311 torch OK $(date)"
  D=results/softloop_greedy/qwen3_5
  if [ ! -f $D/softloop.npz ]; then
    ~/LLM/venv311/bin/python -m chaos_llm.run_softloop \
      --model ~/LLM/models--Qwen--Qwen3.5-2B --steps 300 --temp 1.0 \
      --feedback argmax --dirs 8 --r 0.1 --save-layers 23 --out $D --window 4096
  fi
  if [ -f $D/softloop.npz ] && [ ! -f results/softloop_gates/qwen3_5/loop0/gates.npz ]; then
    ~/LLM/venv311/bin/python -m chaos_llm.run_gates \
      --model ~/LLM/models--Qwen--Qwen3.5-2B --loop $D/softloop.npz \
      --out results/softloop_gates/qwen3_5/loop0
  fi
  for T in 0.1 0.2 0.5 1.0; do
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
  echo "venv311 still broken; skipping qwen3_5 $(date)"
fi

echo "CAMPAIGN DONE $(date)"
