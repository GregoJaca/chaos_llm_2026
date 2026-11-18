#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=~/LLM/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B
echo "QUEUE16 START $(date)"

# 1. Window-flush robustness: same tokens, five windows, shallow layers, 500 tokens
for w in 100 150 200 300 500; do
  echo "=== flush $w $(date)"
  python3 -m chaos_llm.run_a1 --model $M --prompt @prompts.txt --window $w \
    --max-new-tokens 500 --fixed-tokens-from results/long_traj/interstellar_travel.npz \
    --num-directions 4 --radii 0.1 --layers 1,2,3,4,6,7 --batch-dirs 1 \
    --out results/flush_sweep/w$w || echo "FAILED flush $w"
done

# 2. Flush on the cyber prompt
for w in 100 200; do
  python3 -m chaos_llm.run_a1 --model $M --prompt @prompts_cyber.txt --window $w \
    --max-new-tokens 400 --fixed-tokens-from results/long_traj/cybersecurity.npz \
    --num-directions 4 --radii 0.1 --layers 1,2,3,6 --batch-dirs 1 \
    --out results/flush_cyber/w$w || echo "FAILED flushcyber $w"
done

# 3. Layer probes at more positions
for p in 16 32 75 100 150; do
  python3 -m chaos_llm.run_layer_probe --run results/a1_ds_w100_layers \
    --layers 0,4,7,10,14,18,21,24,27 --dirs 4 --r 0.05 --pos $p \
    --out results/layer_probe_p$p || echo "FAILED probe $p"
done

# 4. Window regimes with SINGLE-POSITION perturbations
for w in 50 200 4096; do
  python3 -m chaos_llm.run_state2 --run results/a1_ds_w100_layers \
    --steps 16,50,100 --dirs 12 --radii 0.03,0.1 \
    --layers 7,14,21,27 --inject-layer 0 --batch-dirs 2 --window $w \
    --out results/state_window/w$w || echo "FAILED statew $w"
done

echo "QUEUE16 DONE $(date)"
