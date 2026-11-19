#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv_tf/bin/python
while pgrep -f "chain_models.sh" > /dev/null; do sleep 20; done
echo "QUEUE26 START $(date)"

# 1. Qwen3-4B: three more prompts, A1 all layers
for i in 1 2 3; do
  sed -n "${i}p" prompts_more2.txt > /tmp/Q4_$i.txt
  out=results/models/qwen3_4b/a1_p$i
  [ -f "$out/a1.npz" ] && continue
  $PY -m chaos_llm.run_a1 --model ~/LLM/models--Qwen--Qwen3-4B-Instruct-2507 \
    --prompt @/tmp/Q4_$i.txt --window 100 --max-new-tokens 300 --num-directions 12 \
    --radii 0.1 --layers 0,4,8,12,16,20,24,28,32,35 --batch-dirs 1 --out "$out" || echo "FAILED q4 p$i"
done

# 2. Phi-4-mini: three more prompts
for i in 1 2 3; do
  sed -n "${i}p" prompts_more2.txt > /tmp/PH_$i.txt
  out=results/models/phi4mini/a1_p$i
  [ -f "$out/a1.npz" ] && continue
  $PY -m chaos_llm.run_a1 --model ~/LLM/models--microsoft--Phi-4-mini-instruct \
    --prompt @/tmp/PH_$i.txt --window 100 --max-new-tokens 300 --num-directions 12 \
    --radii 0.3 --layers 0,4,8,12,16,20,24,28,31 --batch-dirs 1 --out "$out" || echo "FAILED phi p$i"
done

# 3. DeepSeek: 4000-token response for four more trajectories
for f in results/big4/traj4k/*.npz; do
  [ -f "$f" ] || continue
  b=$(basename "$f" .npz)
  out=results/big4/a1_4k_more/$b
  [ -f "$out/a1.npz" ] && continue
  python3 -m chaos_llm.run_a1_long --prompt @prompts_long2.txt --window 100 \
    --max-new-tokens 4000 --fixed-tokens-from "$f" --num-directions 4 \
    --radii 0.1 --layers 7,14,21,27 --out "$out" || echo "FAILED a1_4k $b"
done

# 4. DeepSeek: flush law at large windows (needs 1000-token runs)
for w in 600 800; do
  out=results/big4/flush_big/w$w
  [ -f "$out/a1.npz" ] && continue
  python3 -m chaos_llm.run_a1_long --prompt @prompts_long2.txt --window $w \
    --max-new-tokens 1000 --fixed-tokens-from results/big4/traj4k/L2_1.npz \
    --num-directions 4 --radii 0.1 --layers 1,2,3,4,5,6 --out "$out" || echo "FAILED flush $w"
done

echo "QUEUE26 DONE $(date)"
