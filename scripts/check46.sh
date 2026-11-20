#!/bin/bash
cd ~/LLM/chaos_llm_2026
echo "=== queue46 blocksigma traceback"
grep -n -A14 "run_block_sigma.py\", line" results/queue46.log | head -30
echo "=== queue46 k64 q4 traceback"
grep -n -B2 -A12 "run_depth2.py" results/queue46.log | sed -n "1,60p"
