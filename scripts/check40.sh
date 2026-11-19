#!/bin/bash
cd ~/LLM/chaos_llm_2026
echo "--- FAILED lines"
grep -n "FAILED" results/queue40.log | head -10
echo "--- around first failure"
grep -n -A12 "Traceback" results/queue40.log | head -40
