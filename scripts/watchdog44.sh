#!/bin/bash
# Keep the queue alive: restart it if it dies, stop when the sentinel exists.
cd ~/LLM/chaos_llm_2026
for i in $(seq 1 200); do
  if [ -f results/QUEUE44_DONE ]; then echo "watchdog: queue done"; break; fi
  if ! pgrep -f "bash scripts/queue44.sh" > /dev/null; then
    echo "watchdog: restarting queue44 $(date)"
    bash scripts/queue44.sh >> results/queue44.log 2>&1
  fi
  sleep 120
done
