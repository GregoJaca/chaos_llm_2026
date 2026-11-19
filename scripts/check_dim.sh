#!/bin/bash
cd ~/LLM/chaos_llm_2026
echo "--- dim log lines"
grep -n -E "FAILED|Error|Traceback|prompt [0-9]|DONE|vocab saved|loaded" results/queue34.log | head -40
echo "--- files"
ls -la results/dim/deepseek results/dim/qwen3_4b results/dim/phi4mini results/dim/qwen3_5
