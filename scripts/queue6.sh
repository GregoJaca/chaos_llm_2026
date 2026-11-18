#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
while pgrep -f "queue5.sh" > /dev/null; do sleep 20; done
echo "QUEUE6 START $(date)"

# Continuous depth profile: all 28 layers, 3 radii, same fixed tokens
python3 -m chaos_llm.run_a1 --model $M --prompt @prompts.txt --window 100 \
  --max-new-tokens 300 --fixed-tokens-from results/long_traj/interstellar_travel.npz \
  --num-directions 8 --radii 0.01,0.1,0.3 \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27 \
  --batch-dirs 1 --out results/depth_profile_w100 || echo "FAILED depth_profile"

# Same, at the 500-token horizon, to check the layer ordering does not depend on length
python3 -m chaos_llm.run_a1 --model $M --prompt @prompts.txt --window 100 \
  --max-new-tokens 450 --fixed-tokens-from results/long_traj/interstellar_travel.npz \
  --num-directions 6 --radii 0.01,0.1 \
  --layers 0,2,4,7,10,14,18,21,24,27 \
  --batch-dirs 1 --out results/depth_profile_w100_450 || echo "FAILED depth_profile_450"

echo "QUEUE6 DONE $(date)"
