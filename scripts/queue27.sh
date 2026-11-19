#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "chain_q35.sh" > /dev/null; do sleep 30; done
echo "QUEUE27 START $(date)"
ALL=$(seq -s, 0 27)

for pf in prompts.txt prompts_cyber.txt; do
  base=$(basename "$pf" .txt)
  out=results/big4/a1_all_layers/$base
  [ -f "$out/a1.npz" ] && continue
  python3 -m chaos_llm.run_a1 --prompt @$pf --window 100 --max-new-tokens 300 \
    --num-directions 20 --radii 0.03,0.1,0.3 --layers $ALL --batch-dirs 1 \
    --out "$out" || echo "FAILED all layers $base"
done

echo "QUEUE27 DONE $(date)"
