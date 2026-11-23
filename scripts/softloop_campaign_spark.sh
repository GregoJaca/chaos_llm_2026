#!/bin/bash
# ============================================================================
# SOFTLOOP CAMPAIGN (spark, DGX GB10) — big-model gates, Qwen3.5-9B hybrid
# loop grid, and a conditional DS-14B fp32 loop grid (waits for free memory;
# colleagues must not be disturbed).  Resume-safe.
# ============================================================================
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv5/bin/python
DS14=~/LLM/models/DeepSeek-R1-Distill-Qwen-14B
Q314=~/LLM/models/Qwen3-14B
PHI4=~/LLM/models/phi-4
Q35=~/LLM/models/Qwen3.5-9B
echo "SPARK CAMPAIGN START $(date)"

free_gb () { $PY -c "import torch; print(round(torch.cuda.mem_get_info()[0]/1e9))"; }

# ---- S1. gates from the existing big-model a1 corpora (fp16 loads) ----
gg () {  # model  a1dir  tag
  for A in a1_inter_w100 a1_cyber_w100 a1_childhood_w100 a1_inter_w0 a1_inter_full; do
    SRC=results_big/$2/$A/a1.npz
    D=results_big/$2/softloop_gates/$A
    if [ -f $SRC ] && [ ! -f $D/gates.npz ]; then
      echo "=== gates $2/$A $(date)"
      $PY -m chaos_llm.run_gates --model $1 --a1 $SRC --out $D \
        || echo "FAILED gates $2/$A"
    fi
  done
}
gg $DS14 ds14
gg $Q314 qwen314
gg $PHI4 phi4
gg $Q35  qwen35

# ---- S2. Qwen3.5-9B hybrid loop grid (fp16, K=32, r=0.3) ----
TEMPS="1e-9,0.05,0.1,0.15,0.2,0.3,0.5,0.7,1.0,1.4,2.0"
for T in $(echo $TEMPS | tr ',' ' '); do
  D=results_big/softloop/qwen35_9b/T$T
  if [ ! -f $D/softloop.npz ]; then
    echo "=== q35 loop T=$T $(date)"
    $PY -m chaos_llm.run_softloop --model $Q35 --steps 600 --temp $T \
      --dirs 32 --r 0.3 --save-layers 31 --out $D --window 0 \
      || echo "FAILED q35 T=$T"
  fi
  if [ ! -f $D/replay/softloop.npz ] && [ -f $D/softloop.npz ]; then
    echo "=== q35 replay T=$T $(date)"
    $PY -m chaos_llm.run_softloop --model $Q35 --steps 600 --temp $T \
      --mode replay --replay-file $D/softloop.npz \
      --dirs 32 --r 0.3 --save-layers 31 --out $D/replay --window 0
  fi
done

# ---- S3. DS-14B fp32 loop grid: wait for >= 64 GB free (up to 10 h) ----
WAITED=0
while [ "$(free_gb)" -lt 64 ] && [ $WAITED -lt 36000 ]; do
  sleep 600; WAITED=$((WAITED+600))
done
if [ "$(free_gb)" -ge 64 ]; then
  echo "=== 14B window open: free=$(free_gb) GB $(date)"
  for T in 1e-9 0.02 0.05 0.1 0.2 0.5 1.0; do
    D=results_big/softloop/ds14/T$T
    if [ ! -f $D/softloop.npz ]; then
      echo "=== ds14 loop T=$T $(date)"
      $PY -m chaos_llm.run_softloop --model $DS14 --dtype float32 \
        --steps 600 --temp $T --dirs 32 --r 0.03 --save-layers 47 \
        --out $D --window 4096 || echo "FAILED ds14 T=$T"
    fi
    if [ ! -f $D/replay/softloop.npz ] && [ -f $D/softloop.npz ]; then
      echo "=== ds14 replay T=$T $(date)"
      $PY -m chaos_llm.run_softloop --model $DS14 --dtype float32 \
        --steps 600 --temp $T --mode replay --replay-file $D/softloop.npz \
        --dirs 32 --r 0.03 --save-layers 47 --out $D/replay --window 4096
    fi
  done
else
  echo "=== 14B window never opened in 10 h; giving up $(date)"
fi

echo "SPARK CAMPAIGN DONE $(date)"
