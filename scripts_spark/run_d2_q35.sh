#!/bin/bash
cd ~/LLM/chaos_llm_2026
PYTHONPATH=src ~/LLM/venv5/bin/python scripts_spark/d2_distances.py \
  --run-dir results_big/qwen35/traj --states h_last --out results_big/qwen35/d2 \
  > ~/LLM/d2_dist_q35.log 2>&1
echo D2_Q35_DONE >> ~/LLM/d2_dist_q35.log
