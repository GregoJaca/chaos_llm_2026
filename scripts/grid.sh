#!/bin/bash
cd ~/LLM/chaos_llm_2026
echo "--- grid"
for m in deepseek qwen3_4b phi4mini qwen3_5 qwen25_3b; do
  line="$m:"
  for p in 1 2 3 4 5 6 7 8; do
    n=0
    for r in 0.1 0.3 0.6 1.0; do
      [ -f "results/sens/${m}_p$p/r$r/free_run.npz" ] && n=$((n+1))
    done
    line="$line p$p=$n"
  done
  echo "$line"
done
echo "--- transport"
for m in deepseek qwen3_4b phi4mini qwen3_5 qwen25_3b; do
  echo -n "$m:"
  for p in 1 2 3 4 5 6 7 8; do
    [ -f "results/transport/${m}_p$p/transport.npz" ] && echo -n " $p" || true
  done
  echo
done
