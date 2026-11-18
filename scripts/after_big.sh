#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
export PYTHONPATH=src
while pgrep -f "big_queue.sh" > /dev/null; do sleep 60; done
echo "AFTER_BIG START $(date)"
python3 scripts/flush_big_analysis.py || echo "FAILED flush analysis"
python3 scripts/big_analysis_all.py || echo "FAILED big analysis"
touch results/big/ANALYSIS_DONE
echo "AFTER_BIG DONE $(date)"
