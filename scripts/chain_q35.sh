#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
while pgrep -f "queue26.sh" > /dev/null; do sleep 30; done
bash scripts/qwen35_suite.sh > results/qwen35_suite.log 2>&1
echo "Q35 chain done $(date)"
python3 scripts/models_compare.py >> results/qwen35_suite.log 2>&1 || true
