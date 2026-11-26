#!/bin/bash
set -u
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv_tf/bin/python
Q25=~/LLM/models--Qwen--Qwen2.5-3B-Instruct
PHI=~/LLM/models--microsoft--Phi-4-mini-instruct
Q34=~/LLM/models--Qwen--Qwen3-4B-Instruct-2507

# depth with 32 directions for two more models
if [ ! -f results/update/depth_qwen25_k32/depth.npz ]; then
  $PY -m chaos_llm.run_depth2 --model "$Q25" --run results/rstar/qwen25_3b/p1 \
    --tokens 0,4,8,12,16,20,30,40,50,70,100,123,143,180,200,250 --k 32 \
    --dtype float32 --out results/update/depth_qwen25_k32 || echo FAILED q25depth
fi
if [ ! -f results/update/depth_phi_k32/depth.npz ]; then
  $PY -m chaos_llm.run_depth2 --model "$PHI" --run results/rstar/phi4mini/p1 \
    --tokens 0,4,8,12,16,20,30,40,50,70,100,123,143,180,200,250 --k 32 \
    --dtype float16 --out results/update/depth_phi_k32 || echo FAILED phidepth
fi

# sampled feedback for Qwen3-4B (the missing panel in F9)
for mode in sample_crn sample_ind; do
  tag=${mode#sample_}
  D=results/softloop_sample/qwen3_4b/${tag}_T1.0
  if [ ! -f "$D/softloop.npz" ]; then
    $PY -m chaos_llm.run_softloop --model "$Q34" --steps 600 --temp 1.0 \
      --dirs 16 --r 0.1 --feedback "$mode" --save-layers 35 --out "$D" || echo FAILED $tag q34
  fi
done
echo "EXTRA X11 DONE $(date)"
