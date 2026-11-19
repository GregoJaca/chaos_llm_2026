#!/bin/bash
cd ~/LLM/chaos_llm_2026 || exit 1
while pgrep -f "model_suite2.sh" > /dev/null; do sleep 30; done
echo "CHAIN: phi start $(date)"
bash scripts/model_suite2.sh ~/LLM/models--microsoft--Phi-4-mini-instruct phi4mini 0.3 32
echo "CHAIN: phi done $(date)"
python3 scripts/models_compare.py || echo "FAILED compare"
echo "CHAIN DONE $(date)"
