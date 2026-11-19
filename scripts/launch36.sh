#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
nohup setsid bash scripts/queue36.sh > results/queue36.log 2>&1 < /dev/null &
disown
sleep 3
echo "launched"
cat results/queue36.log 2>/dev/null | head -5
pgrep -af queue36 | head -3
