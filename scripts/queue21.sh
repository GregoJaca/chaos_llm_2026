#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
while pgrep -f "queue20.sh" > /dev/null; do sleep 30; done
echo "QUEUE21 START $(date)"

# 1. Extend four 1000-token trajectories to 3000 (source has prompt_ids)
for n in ocean_currents history_of_computing urban_transportation astronomy_observations; do
  out=results/big/long3k/${n}.npz
  [ -f "$out" ] && continue
  python3 -m chaos_llm.extend_trajectories --model $M \
    --src results/long_traj/${n}.npz --out "$out" --extra 2000 --window 4096 \
    || echo "FAILED extend $n"
done
echo "extend done $(date)"

# 2. Long-horizon A1 at 1000 tokens (inference_mode fix)
python3 -m chaos_llm.run_a1 --model $M --prompt @prompts_long2.txt --window 100 \
  --max-new-tokens 1000 --fixed-tokens-from results/long_traj_2k/ocean_currents.npz \
  --num-directions 4 --radii 0.1 --layers 7,14,21,27 --batch-dirs 1 \
  --out results/big/long1000 || echo "FAILED long1000"
echo "long1000 done $(date)"

# 3. If 1000 worked, try 1500
if [ -f results/big/long1000/a1.npz ]; then
  python3 -m chaos_llm.run_a1 --model $M --prompt @prompts_long2.txt --window 100 \
    --max-new-tokens 1500 --fixed-tokens-from results/long_traj_2k/ocean_currents.npz \
    --num-directions 3 --radii 0.1 --layers 7,14,21,27 --batch-dirs 1 \
    --out results/big/long1500 || echo "FAILED long1500"
fi

# 4. 500-token A1 for the fifth regime prompt
sed -n "5p" prompts_more.txt > /tmp/lh5.txt
python3 -m chaos_llm.run_a1 --model $M --prompt @/tmp/lh5.txt --window 100 \
  --max-new-tokens 500 --num-directions 8 --radii 0.1 --layers 7,14,21,27 \
  --batch-dirs 1 --out results/big/long500/p5 || echo "FAILED long p5"

echo "QUEUE21 DONE $(date)"
