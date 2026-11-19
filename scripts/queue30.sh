#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
export CHAOS_DTYPE=float16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/LLM/venv311b/bin/python
M=~/LLM/models--Qwen--Qwen3.5-2B
echo "QUEUE30 START $(date)"

for p in 2 3; do
  for r in 0.1 0.3 0.6 1.0 1.5; do
    out=results/models/qwen3_5/chaos/p${p}_r${r}
    [ -f "$out/free_run.npz" ] && continue
    $PY -m chaos_llm.run_thresholds --model $M --run results/models/qwen3_5/a1_p$p \
      --radii $r --chunk 4 --out "$out" || echo "FAILED q35 p$p r$r"
  done
done

# DeepSeek: second prompt per prompt-length (flush law)
for i in 1 2 3 4; do
  sed -n "${i}p" prompts_lengths.txt > /tmp/LB_$i.txt
  src=results/big4/len_traj/p$i.npz
  [ -f "$src" ] || continue
  for w in 100 200; do
    out=results/big4/flush_len2/p${i}_w${w}
    [ -f "$out/a1.npz" ] && continue
    python3 -m chaos_llm.run_a1_long --prompt @/tmp/LB_$i.txt --window $w \
      --max-new-tokens 600 --fixed-tokens-from "$src" --num-directions 4 \
      --radii 0.1 --layers 1,2,3,4,5,6 --out "$out" || echo "FAILED len2 $i $w"
  done
done

echo "QUEUE30 DONE $(date)"
