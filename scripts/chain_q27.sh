#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
while pgrep -f "qwen35_suite.sh" > /dev/null; do sleep 30; done
bash scripts/queue27.sh > results/queue27.log 2>&1
echo "QUEUE27 chain done $(date)"
