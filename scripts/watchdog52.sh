#!/bin/bash
cd ~/LLM/chaos_llm_2026
for i in $(seq 1 400); do
  if [ -f results/QUEUE52_DONE ]; then echo "watchdog: done"; break; fi
  if ! pgrep -f "bash scripts/queue52.sh" > /dev/null; then
    echo "watchdog: restarting $(date)"
    bash scripts/queue52.sh >> results/queue52.log 2>&1
  fi
  sleep 120
done
