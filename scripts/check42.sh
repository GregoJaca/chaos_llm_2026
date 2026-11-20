#!/bin/bash
cd ~/LLM/chaos_llm_2026
echo "--- queue42 FAILED/tracebacks"
grep -n -B2 -A10 "Traceback" results/queue42.log | head -40
echo "--- files"
ls -la results/depth2/*/ 2>/dev/null | head -30
