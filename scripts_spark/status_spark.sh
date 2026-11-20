#!/bin/bash
# One-call status of the spark campaign.
echo "=== $(date -u) ==="
echo "--- gpu/mem ---"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
free -g | sed -n 2p | awk '{print "Mem avail:", 7, "GB"}'
echo "--- active jobs ---"
pgrep -af 'chaos_llm|collect_traj|run_a1|run_free|run_state|run_depth|run_nonnorm|smoke_big' | grep -v pgrep | head -3 | cut -c1-140
echo "--- watchers ---"
pgrep -af 'phase1b|queue_' | grep -v pgrep | cut -c1-70
echo "--- queue tails ---"
for f in ~/LLM/queue_p1.log ~/LLM/phase1b.log ~/LLM/queue_p2.log ~/LLM/queue_p3.log ~/LLM/queue_p4.log ~/LLM/queue_p5.log; do
  if [ -f "$f" ]; then
    echo "[$f]"
    tail -1 "$f" | cut -c1-100
  fi
done
echo "--- results inventory ---"
for d in ~/LLM/chaos_llm_2026/results_big/*; do
  echo "$(basename $d): $(find $d -name '*.npz' | wc -l) npz"
done
